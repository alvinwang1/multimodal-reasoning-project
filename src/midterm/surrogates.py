import argparse
import json
import time

import torch

from src.analysis_a.model import OmniScorer, sync
from src.analysis_a.prompts import SYSTEM_PROMPT
from src.common import load_config, repo_path, seed_everything
from src.data import OmniBenchSubset

CAPTION_PROMPT=("Describe this image in detail. Include the setting, objects, people, actions and any visible text. "
                "Reply with the description only, in English, and do not ask any questions.")
AUDIO_PROMPT=("Describe this audio clip. Do not reply to or answer anyone in it. If there is speech, give a word-for-word "
              "transcript and describe the speakers. Describe any other sounds or music: their sources, events, "
              "instruments and mood. Reply with the description only, in English, and do not ask any questions.")
STOP_STRINGS=["\nHuman:", "Human:", "\nUser:"]
ASR_MODEL="openai/whisper-large-v3-turbo"


def messages(media, prompt):
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [media, {"type": "text", "text": prompt}]}]


@torch.inference_mode()
def describe(scorer, media, prompt, max_new_tokens):
    _, inputs=scorer.prepare(messages(media, prompt))
    inputs=inputs.to(scorer.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k]=v.to(scorer.dtype)
    n_in=inputs["input_ids"].shape[1]
    sync(scorer.device)
    t0=time.perf_counter()
    tok=scorer.processor.tokenizer
    eos=[tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    out=scorer.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos,
                              pad_token_id=eos[1], stop_strings=STOP_STRINGS, tokenizer=tok)
    sync(scorer.device)
    ms=(time.perf_counter()-t0)*1000
    gen=out[0, n_in:]
    text=tok.decode(gen, skip_special_tokens=True)
    for s in STOP_STRINGS:
        text=text.split(s)[0]
    return text.strip(), ms, int(gen.numel()), n_in


def transcribe(asr, wav, device):
    sync(device)
    t0=time.perf_counter()
    text=asr(wav.copy(), return_timestamps=True, generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
    sync(device)
    return text, (time.perf_counter()-t0)*1000, len(asr.tokenizer.encode(text, add_special_tokens=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "test", "all"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None)
    args=ap.parse_args()
    cfg=load_config(args.config)
    seed_everything(cfg["seed"])

    split=json.loads(repo_path(cfg["midterm"]["split"]).read_text())
    ids=split["train"]+split["test"] if args.split=="all" else split[args.split]
    ids=ids[:args.limit] if args.limit else ids
    out=repo_path(args.out or cfg["midterm"]["surrogates"])
    res=json.loads(out.read_text()) if out.exists() else {}
    todo=[i for i in ids if str(i) not in res]
    print(f"{len(ids)} items, {len(ids)-len(todo)} done, {len(todo)} to run -> {out}", flush=True)
    if not todo:
        return

    from transformers import pipeline
    data=OmniBenchSubset(cfg, ids=ids)
    scorer=OmniScorer(cfg, device=args.device, dtype=args.dtype)
    asr=pipeline("automatic-speech-recognition", model=ASR_MODEL, device=scorer.device,
                 dtype=torch.float32 if scorer.device.type=="cpu" else torch.float16)
    print(f"models on {scorer.device}", flush=True)

    ex=data[todo[0]]
    describe(scorer, {"type": "image", "image": ex["image"]}, CAPTION_PROMPT, 8)
    describe(scorer, {"type": "audio", "audio": ex["audio"]}, AUDIO_PROMPT, 8)
    transcribe(asr, ex["audio"], scorer.device)

    out.parent.mkdir(parents=True, exist_ok=True)
    t_start=time.time()
    for n, idx in enumerate(todo, 1):
        ex=data[idx]
        cap, cap_ms, cap_tok, cap_in=describe(scorer, {"type": "image", "image": ex["image"]}, CAPTION_PROMPT, args.max_new_tokens)
        desc, desc_ms, desc_tok, desc_in=describe(scorer, {"type": "audio", "audio": ex["audio"]}, AUDIO_PROMPT, args.max_new_tokens)
        asr_text, asr_ms, asr_tok=transcribe(asr, ex["audio"], scorer.device)
        res[str(idx)]={
            "gen_caption": cap, "gen_caption_ms": cap_ms, "gen_caption_tok": cap_tok, "gen_caption_in_tok": cap_in,
            "gen_audio_desc": desc, "gen_audio_desc_ms": desc_ms, "gen_audio_desc_tok": desc_tok,
            "gen_audio_desc_in_tok": desc_in,
            "asr": asr_text, "asr_ms": asr_ms, "asr_tok": asr_tok,
            "audio_type": ex["audio type"], "gen_device": str(scorer.device), "asr_model": ASR_MODEL,
        }
        tmp=out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(out)
        eta=(time.time()-t_start)/n*(len(todo)-n)/60
        print(f"{n}/{len(todo)} id={idx} [{ex['audio type']}] cap {cap_tok}tok {cap_ms/1000:.1f}s | "
              f"desc {desc_tok}tok {desc_ms/1000:.1f}s | asr {asr_tok}tok {asr_ms/1000:.1f}s | ETA {eta:.1f} min", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
