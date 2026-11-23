"""Idea 4 (final): generate every text surrogate the final experiments score, with per-step timing.

For each item in the Idea 4 subset (data/final_idea4_ids.json):
  midterm baselines   gen_caption, gen_audio_desc (Qwen, question-agnostic), asr (Whisper)
  audio tagging       AST (AudioSet tagger): top tags + P(Speech); a clip counts as speech if P(Speech) >= 0.5
  question-conditioned q_audio_desc: Qwen describes the audio given the question (never the options)
  ASR check           asr_status in {ok, empty, filler, loop}; asr_failed = classified as speech and not ok
  derived texts       route_text = asr if speech else tags; pipe_text = asr if speech else tags + q_audio_desc
  threshold sweep     tag_text = tags, tagsq_text = tags + q_audio_desc, for every item (the non-speech branches)
  masking test        masked_audio_content = the human audio description with every option word masked
  generation length   Qwen writes each text once, with at most max(final.idea4_gen_lens) = 256 new tokens: the
                      fields above. <field>_t<n> (+ _ms, _tok) for each n in idea4_gen_lens is that output cut to its
                      first n tokens, for gen_caption, gen_audio_desc, q_audio_desc (plus pipe_text_t<n> and
                      tagsq_text_t<n>). Decoding is greedy, so this is exactly what max_new_tokens=n produces; _ms is
                      when the n-th token was ready.
                      Scored as <config>_t<n> (e.g. img_qdesc_t80), the generation-length ablation.
Scored by src.analysis_a.run (configs img_route, img_route_fb, img_qdesc, img_pipe, gencap_pipe, img_masked,
cap_masked, plus the midterm img_asr, img_gendesc, gen_cap_desc, gen_cap_asr; for the threshold sweep img_tags,
img_tagsq, gencap_tagsq, gencap_aud). Resumable.

    python -m src.final.idea4_generate --make-subset      # once, locally: freeze the subset
    python -m src.final.idea4_generate --limit 2          # smoke test
    python -m src.final.idea4_generate --out /content/drive/MyDrive/mmml_final/idea4_surrogates.json
"""
import argparse
import json
import re
import time

import numpy as np
import torch

from src.analysis_a.model import OmniScorer, sync
from src.common import load_config, repo_path, seed_everything
from src.data import AUDIO_TYPES, OmniBenchSubset, load_omnibench, text_records
from src.midterm.idea4_surrogates import asr_status
from src.midterm.surrogates import AUDIO_PROMPT, CAPTION_PROMPT, STOP_STRINGS, describe, messages, transcribe

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
ASR_MODEL = "openai/whisper-large-v3-turbo"
SPEECH_THRESHOLD = 0.5
N_TAGS = 5
MIN_TAG_PROB = 0.05
Q_AUDIO_PROMPT = ("A question will be asked about this audio clip: \"{q}\"\n"
                  "Describe only what you hear that is relevant to this question. If there is speech, transcribe "
                  "the relevant words. Otherwise name the specific sound sources, instruments, events and actions. "
                  "Do not answer the question. Reply with the description only, in English, and do not ask any "
                  "questions.")
STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "his", "her", "their",
             "its", "into", "onto", "some", "any", "all", "not", "but", "has", "have", "had", "who", "what",
             "which", "when", "where", "while", "being", "been", "they", "them", "she", "him", "one", "two"}


def build_subset(cfg):
    """Midterm test items (40/type) + fresh, never-inspected speech and sound items (+ the few fresh music)."""
    fcfg = cfg["final"]
    groups = json.loads(repo_path(fcfg["folds"]).read_text())["groups"]
    records, _ = text_records(load_omnibench(cfg))
    by_id = {r["index"]: r for r in records}
    rng = np.random.default_rng(fcfg["fold_seed"] + 4)
    ids = list(groups["midterm_test"])
    for t in AUDIO_TYPES:
        fresh = sorted(i for i in groups["fresh"] if by_id[i]["audio type"] == t)
        take = min(len(fresh), fcfg["idea4_n_fresh_per_type"])
        ids += rng.choice(fresh, size=take, replace=False).tolist()
    ids = sorted(ids)
    out = repo_path(fcfg["idea4_ids"])
    out.write_text(json.dumps({"seed": fcfg["fold_seed"] + 4, "n": len(ids),
                               "per_audio_type": {t: sum(by_id[i]["audio type"] == t for i in ids) for t in AUDIO_TYPES},
                               "midterm_test": groups["midterm_test"], "ids": ids}, indent=1))
    print(f"wrote {len(ids)} ids -> {out}")


def mask_options(text, options):
    """Mask every content word that appears in any option (so the mask itself does not point at the answer)."""
    words = {w for o in options for w in re.findall(r"[a-z]+", o.lower()) if len(w) >= 3 and w not in STOPWORDS}
    forms = set(words)
    for w in words:  # singular forms too, so "lutes" in an option also masks "lute" in the text
        if w.endswith("s") and len(w) >= 5:
            forms.add(w[:-1])
        if w.endswith("es") and len(w) >= 6:
            forms.add(w[:-2])
    n = 0
    for w in sorted(forms, key=len, reverse=True):
        text, k = re.subn(rf"\b{re.escape(w)}(e?s)?\b", "[MASK]", text, flags=re.IGNORECASE)
        n += k
    return text, n


@torch.inference_mode()
def tag(ast, fe, wav, sr, device):
    sync(device)
    t0 = time.perf_counter()
    inputs = fe(wav, sampling_rate=sr, return_tensors="pt").to(device)
    probs = torch.sigmoid(ast(**inputs).logits[0]).float().cpu().numpy()
    sync(device)
    ms = (time.perf_counter() - t0) * 1000
    labels = ast.config.id2label
    top = [(labels[int(i)], float(probs[i])) for i in np.argsort(-probs)[:N_TAGS] if probs[i] >= MIN_TAG_PROB]
    speech_id = next(int(i) for i, name in labels.items() if name == "Speech")
    return top, float(probs[speech_id]), ms


@torch.inference_mode()
def generate_capped(scorer, media, prompt, max_new_tokens, lens):
    """Greedy generation with up to max_new_tokens (as src.midterm.surrogates.describe), plus the output at each
    shorter cap. Greedy decoding picks every token without looking at the limit, so the first n tokens of this
    output are exactly what max_new_tokens=n would give; a per-token timer gives how long that run would take.
    -> {n: (text, ms, n_tokens)} for every n in lens and for max_new_tokens."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StepTimer(StoppingCriteria):  # never stops generation; notes when each new token is ready
        def __init__(self):
            self.times = []

        def __call__(self, input_ids, scores, **kwargs):
            sync(scorer.device)
            self.times.append(time.perf_counter())
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    _, inputs = scorer.prepare(messages(media, prompt))
    inputs = inputs.to(scorer.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(scorer.dtype)
    n_in = inputs["input_ids"].shape[1]
    tok = scorer.processor.tokenizer
    eos = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    timer = StepTimer()
    sync(scorer.device)
    t0 = time.perf_counter()
    out = scorer.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos,
                                pad_token_id=eos[1], stop_strings=STOP_STRINGS, tokenizer=tok,
                                stopping_criteria=StoppingCriteriaList([timer]))
    sync(scorer.device)
    total_ms = (time.perf_counter() - t0) * 1000
    gen = out[0, n_in:]
    res = {}
    for n in sorted(set(lens) | {max_new_tokens}):
        k = min(n, int(gen.numel()))
        text = tok.decode(gen[:k], skip_special_tokens=True)
        for s in STOP_STRINGS:
            text = text.split(s)[0]
        # cap not reached (or the full run): the capped run is this run; otherwise the time the k-th token was ready
        ms = total_ms if k == gen.numel() or k > len(timer.times) else (timer.times[k - 1] - t0) * 1000
        res[n] = (text.strip(), ms, k)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--make-subset", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="explicit OmniBench indices (testing)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: the largest of --gen-lens")
    ap.add_argument("--gen-lens", type=int, nargs="*", default=None,
                    help="generation caps for the length ablation (default: final.idea4_gen_lens in the config)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    if args.make_subset:
        build_subset(cfg)
        return
    lens = sorted(set(args.gen_lens or cfg["final"]["idea4_gen_lens"]))
    max_new = args.max_new_tokens or lens[-1]
    if max_new < lens[-1]:
        raise SystemExit(f"--max-new-tokens {max_new} is below the largest generation length {lens[-1]}")

    ids = args.ids or json.loads(repo_path(cfg["final"]["idea4_ids"]).read_text())["ids"]
    ids = ids[:args.limit] if args.limit else ids
    out = repo_path(args.out or cfg["final"]["idea4_surrogates"])
    res = json.loads(out.read_text()) if out.exists() and out.stat().st_size else {}
    required = ["tag_text", "tagsq_text"] + [f"{f}_t{n}" for n in lens for f in
                                             ("gen_caption", "gen_audio_desc", "q_audio_desc", "pipe_text", "tagsq_text")]
    # items made with other generation settings, or by an older version missing some fields, are redone
    todo = [i for i in ids if str(i) not in res or res[str(i)].get("gen_max_new_tokens") != max_new
            or any(f not in res[str(i)] for f in required)]
    print(f"{len(ids)} items, {len(ids) - len(todo)} done, {len(todo)} to run -> {out}", flush=True)
    if not todo:
        return

    from transformers import ASTForAudioClassification, AutoFeatureExtractor, pipeline
    sr = cfg["dataset"]["audio_sr"]
    data = OmniBenchSubset(cfg, ids=ids)
    scorer = OmniScorer(cfg, device=args.device, dtype=args.dtype)
    half = torch.float32 if scorer.device.type == "cpu" else torch.float16
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=scorer.device, dtype=half)
    fe = AutoFeatureExtractor.from_pretrained(AST_MODEL)
    ast = ASTForAudioClassification.from_pretrained(AST_MODEL).to(scorer.device).eval()
    print(f"models on {scorer.device}", flush=True)

    ex = data[todo[0]]  # warm up every model once so the first item's timings are not inflated
    describe(scorer, {"type": "image", "image": ex["image"]}, CAPTION_PROMPT, 8)
    describe(scorer, {"type": "audio", "audio": ex["audio"]}, AUDIO_PROMPT, 8)
    transcribe(asr, ex["audio"], scorer.device)
    tag(ast, fe, ex["audio"], sr, scorer.device)

    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for n, idx in enumerate(todo, 1):
        ex = data[idx]
        img, aud = {"type": "image", "image": ex["image"]}, {"type": "audio", "audio": ex["audio"]}
        gens = {"gen_caption": generate_capped(scorer, img, CAPTION_PROMPT, max_new, lens),
                "gen_audio_desc": generate_capped(scorer, aud, AUDIO_PROMPT, max_new, lens),
                "q_audio_desc": generate_capped(scorer, aud, Q_AUDIO_PROMPT.format(q=ex["question"].strip()),
                                                max_new, lens)}
        (cap, cap_ms, cap_tok), (desc, desc_ms, desc_tok), (qdesc, qdesc_ms, qdesc_tok) = (
            g[max_new] for g in gens.values())
        asr_text, asr_ms, asr_tok = transcribe(asr, ex["audio"], scorer.device)
        tags, p_speech, tag_ms = tag(ast, fe, ex["audio"], sr, scorer.device)

        is_speech = p_speech >= SPEECH_THRESHOLD
        status = asr_status(asr_text, asr_tok)
        tag_text = "Audio tags: " + (", ".join(f"{name} ({p:.2f})" for name, p in tags) or "none")
        masked, n_masked = mask_options(ex["audio content"], ex["options"])
        res[str(idx)] = {
            "audio_type": ex["audio type"],
            "gen_caption": cap, "gen_caption_ms": cap_ms, "gen_caption_tok": cap_tok,
            "gen_audio_desc": desc, "gen_audio_desc_ms": desc_ms, "gen_audio_desc_tok": desc_tok,
            "q_audio_desc": qdesc, "q_audio_desc_ms": qdesc_ms, "q_audio_desc_tok": qdesc_tok,
            "asr": asr_text, "asr_ms": asr_ms, "asr_tok": asr_tok, "asr_status": status,
            "ast_tags": tags, "ast_p_speech": p_speech, "ast_ms": tag_ms,
            "pred_speech": is_speech, "asr_failed": bool(is_speech and status != "ok"),
            "route_text": asr_text if is_speech else tag_text,
            "pipe_text": asr_text if is_speech else f"{tag_text}\n{qdesc}",
            "tag_text": tag_text, "tagsq_text": f"{tag_text}\n{qdesc}",  # every item, for the threshold sweep
            "masked_audio_content": masked, "n_masked": n_masked,
            "gen_device": str(scorer.device), "asr_model": ASR_MODEL, "ast_model": AST_MODEL,
            "gen_max_new_tokens": max_new, "gen_lens": lens,
        }
        rec = res[str(idx)]
        for lim in lens:  # length ablation: each generated text cut to its first `lim` tokens, and the time that takes
            for name, g in gens.items():
                rec[f"{name}_t{lim}"], rec[f"{name}_t{lim}_ms"], rec[f"{name}_t{lim}_tok"] = g[lim]
            rec[f"tagsq_text_t{lim}"] = f"{tag_text}\n{gens['q_audio_desc'][lim][0]}"
            rec[f"pipe_text_t{lim}"] = asr_text if is_speech else rec[f"tagsq_text_t{lim}"]
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(out)
        eta = (time.time() - t_start) / n * (len(todo) - n) / 60
        print(f"{n}/{len(todo)} id={idx} [{ex['audio type']}] P(speech)={p_speech:.2f} asr={status} | "
              f"cap {cap_ms / 1000:.1f}s desc {desc_ms / 1000:.1f}s qdesc {qdesc_ms / 1000:.1f}s "
              f"asr {asr_ms / 1000:.1f}s ast {tag_ms / 1000:.2f}s | ETA {eta:.1f} min", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
