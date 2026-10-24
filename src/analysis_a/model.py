"""Qwen2.5-Omni thinker (text output only) + single-forward-pass option scoring."""
import threading
import time

import numpy as np
import torch

from src.data import LETTERS


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class PeakMemory:
    """Peak memory over a code block, in MB.

    CUDA: exact, via the allocator's peak stats.
    MPS: torch has no peak counter, so a background thread polls the allocated memory
    every `interval` seconds; the reported peak is an approximation.
    """

    def __init__(self, device, interval=0.002):
        self.device, self.interval = device, interval
        self.peak_mb = self.baseline_mb = None

    def _current(self):
        if self.device.type == "mps":
            return torch.mps.current_allocated_memory()
        return 0

    def _poll(self):
        while not self._stop.is_set():
            self._max = max(self._max, self._current())
            time.sleep(self.interval)

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.baseline_mb = torch.cuda.memory_allocated() / 2**20
        elif self.device.type == "mps":
            self._max = self._current()
            self.baseline_mb = self._max / 2**20
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            self.peak_mb = torch.cuda.max_memory_allocated() / 2**20
        elif self.device.type == "mps":
            self._stop.set()
            self._thread.join()
            self._max = max(self._max, self._current())
            self.peak_mb = self._max / 2**20
        return False


class OmniScorer:
    def __init__(self, cfg, device="auto", dtype=None):
        from qwen_omni_utils import fetch_image  # noqa: F401  (import check)
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        mcfg = cfg["model"]
        self.cfg = cfg
        self.device = pick_device(device if device != "auto" else mcfg["device"])
        self.dtype = getattr(torch, dtype or mcfg["dtype"])
        self.min_pixels, self.max_pixels = mcfg["min_pixels"], mcfg["max_pixels"]
        self.sr = cfg["dataset"]["audio_sr"]

        self.processor = Qwen2_5OmniProcessor.from_pretrained(mcfg["hf_id"])
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            mcfg["hf_id"], dtype=self.dtype, attn_implementation=mcfg["attn_implementation"]
        ).to(self.device).eval()

        tok = self.processor.tokenizer
        self.image_token_id = tok.convert_tokens_to_ids(self.processor.image_token)
        self.audio_token_id = tok.convert_tokens_to_ids(self.processor.audio_token)
        self.letter_ids = []
        for L in LETTERS:
            ids = tok.encode(L, add_special_tokens=False)
            assert len(ids) == 1, f"letter {L!r} is not a single token: {ids}"
            self.letter_ids.append(ids[0])

    def letter_token_report(self):
        tok = self.processor.tokenizer
        rows = []
        for s in ["A", "B", "C", "D", " A", " B", "A.", "(A)"]:
            ids = tok.encode(s, add_special_tokens=False)
            rows.append((s, ids, [tok.decode([i]) for i in ids]))
        return rows

    def prepare(self, messages):
        """Chat template + processor. Images are resized with Qwen's smart_resize to the pixel budget."""
        from qwen_omni_utils import fetch_image

        images, audios = [], []
        for m in messages:
            for c in m["content"]:
                if c["type"] == "image":
                    max_px = min(c.get("max_pixels", self.max_pixels), self.max_pixels)
                    images.append(fetch_image({"image": c["image"], "min_pixels": self.min_pixels,
                                               "max_pixels": max(max_px, self.min_pixels)}))
                elif c["type"] == "audio":
                    audios.append(c["audio"])
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kwargs = dict(text=[text], return_tensors="pt", padding=True)
        if images:
            kwargs["images"] = images
        if audios:
            kwargs["audio"] = audios
            kwargs["sampling_rate"] = self.sr
        inputs = self.processor(**kwargs)
        return text, inputs

    def token_counts(self, inputs, surrogate_parts=()):
        ids = inputs["input_ids"][0]
        n_img = int((ids == self.image_token_id).sum())
        n_aud = int((ids == self.audio_token_id).sum())
        n_sur = sum(len(self.processor.tokenizer.encode(p, add_special_tokens=False)) for p in surrogate_parts)
        return {"n_total_tok": int(ids.numel()), "n_image_tok": n_img, "n_audio_tok": n_aud,
                "n_text_tok": int(ids.numel()) - n_img - n_aud, "n_surrogate_tok": n_sur}

    @torch.inference_mode()
    def score(self, inputs):
        """One forward pass. Returns (probs over A-D, latency_ms, peak_mem_mb, baseline_mb)."""
        inputs = inputs.to(self.device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        sync(self.device)
        with PeakMemory(self.device) as mem:
            t0 = time.perf_counter()
            out = self.model(**inputs, use_cache=False, logits_to_keep=1)
            sync(self.device)
            t1 = time.perf_counter()
        logits = out.logits[0, -1].float()
        letter_logits = logits[self.letter_ids]
        probs = torch.softmax(letter_logits, dim=-1).cpu().numpy()
        top_id = int(logits.argmax())
        return {
            "probs_display": probs.tolist(),
            "letter_logits": letter_logits.cpu().tolist(),
            "top_vocab_token": self.processor.tokenizer.decode([top_id]),
            "top_vocab_is_letter": top_id in self.letter_ids,
            "latency_ms": (t1 - t0) * 1000,
            "peak_mem_mb": mem.peak_mb,
            "baseline_mem_mb": mem.baseline_mb,
        }

    @torch.inference_mode()
    def generate_text(self, inputs, max_new_tokens=4):
        inputs = inputs.to(self.device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def map_back(probs_display, perm):
    """probs over displayed letters -> probs over original option order."""
    orig = np.zeros(len(perm))
    for j, o in enumerate(perm):
        orig[o] = probs_display[j]
    return orig.tolist()
