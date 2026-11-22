"""Idea 4: generate text surrogates with models instead of using OmniBench's human-written ones.

For every item in the chosen split, writes to results/midterm/generated_surrogates.json:
  asr             Whisper (large-v3-turbo) transcript of the audio
  gen_caption     Qwen2.5-Omni's own description of the image (question-agnostic)
  gen_audio_desc  Qwen2.5-Omni's own description of the audio (speech transcribed + other sounds)
plus generation cost: wall time, input tokens and generated tokens for each surrogate.
Resumable: items already in the output file are skipped.

    python -m src.midterm.surrogates                      # test split (default)
    python -m src.midterm.surrogates --limit 2 --max-new-tokens 16 --device cpu   # quick check
"""
import argparse
import json
import logging
import time

import torch

from src.analysis_a.model import OmniScorer, sync
from src.analysis_a.prompts import SYSTEM_PROMPT
from src.common import load_config, repo_path, seed_everything
from src.data import OmniBenchSubset

log = logging.getLogger("surrogates")

CAPTION_PROMPT = "Describe this image in detail."
AUDIO_PROMPT = ("You are describing an audio clip for someone who cannot hear it. Do not reply to the "
                "speakers. First, transcribe any speech word for word inside quotation marks. Then describe "
                "any other sounds or music you hear.")
WHISPER_ID = "openai/whisper-large-v3-turbo"


def generate(scorer, media_item, prompt, max_new_tokens):
    """Greedy generation from one image or audio clip; returns (text, ms, n_input_tokens, n_new_tokens)."""
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [media_item, {"type": "text", "text": prompt}]}]
    _, inputs = scorer.prepare(messages)
    inputs = inputs.to(scorer.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(scorer.dtype)
    n_in = int(inputs["input_ids"].shape[1])
    tok = scorer.processor.tokenizer
    # Stop at the end of the assistant turn. Without this the thinker keeps going and invents
    # "Human: ..." turns until max_new_tokens (observed on the first run).
    eos = [i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"),
                       tok.eos_token_id} if isinstance(i, int) and i >= 0]
    sync(scorer.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = scorer.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=eos, pad_token_id=eos[0])
    sync(scorer.device)
    ms = (time.perf_counter() - t0) * 1000
    new = out[0, n_in:]
    text = tok.decode(new, skip_special_tokens=True).strip()
    text = text.split("\nHuman:")[0].split("Human:")[0].strip()  # safety net if a fake turn still appears
    return text, ms, n_in, int(new.numel())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "test", "all"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=160, help="cap for caption / audio description")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    split = json.loads(repo_path(cfg["midterm"]["split"]).read_text())
    ids = split["train"] + split["test"] if args.split == "all" else split[args.split]
    if args.limit:
        ids = ids[: args.limit]
    out = repo_path(args.out or cfg["midterm"]["surrogates"])
    done = json.loads(out.read_text()) if out.exists() else {}
    todo = [i for i in ids if str(i) not in done]
    log.info("%d items, %d done, %d to generate -> %s", len(ids), len(done), len(todo), out)
    if not todo:
        return

    from transformers import pipeline
    scorer = OmniScorer(cfg, device=args.device, dtype=args.dtype or ("float32" if args.device == "cpu" else None))
    asr = pipeline("automatic-speech-recognition", model=WHISPER_ID, device=scorer.device,
                   dtype=torch.float16 if scorer.device.type != "cpu" else torch.float32)
    log.info("models ready on %s", scorer.device)

    data = OmniBenchSubset(cfg, ids=ids)
    sr = cfg["dataset"]["audio_sr"]
    t_start = time.time()
    for k, idx in enumerate(todo, 1):
        ex = data[idx]
        rec = {"audio_type": ex["audio type"]}

        t0 = time.perf_counter()
        res = asr({"raw": ex["audio"], "sampling_rate": sr}, generate_kwargs={"task": "transcribe"})
        rec["asr"] = res["text"].strip()
        rec["asr_ms"] = (time.perf_counter() - t0) * 1000

        rec["gen_caption"], rec["caption_ms"], rec["caption_in_tok"], rec["caption_new_tok"] = generate(
            scorer, {"type": "image", "image": ex["image"]}, CAPTION_PROMPT, args.max_new_tokens)
        rec["gen_audio_desc"], rec["desc_ms"], rec["desc_in_tok"], rec["desc_new_tok"] = generate(
            scorer, {"type": "audio", "audio": ex["audio"]}, AUDIO_PROMPT, args.max_new_tokens)

        done[str(idx)] = rec
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(done, indent=1))
        el = time.time() - t_start
        log.info("%d/%d id=%d (%s) asr %.0fms | caption %d tok %.1fs | audio desc %d tok %.1fs | ETA %.1f min",
                 k, len(todo), idx, rec["audio_type"], rec["asr_ms"], rec["caption_new_tok"],
                 rec["caption_ms"] / 1000, rec["desc_new_tok"], rec["desc_ms"] / 1000,
                 el / k * (len(todo) - k) / 60)
    log.info("done: %d items in %s", len(done), out)


if __name__ == "__main__":
    main()
