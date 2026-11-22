import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis_a.prompts import config_spec
from src.common import (AUDIO_TYPE_COLORS, COL_W, CONFIG_LABELS, INK, INK_2, PALETTE, load_config, read_jsonl,
                        repo_path, save_fig, set_plot_style)
from src.data import AUDIO_TYPES, LETTERS

GOLD=["full", "img_trans", "cap_aud", "cap_trans", "img_only", "aud_only", "q_only"]
GEN=["gen_cap_desc", "gen_cap_asr", "img_gendesc", "img_asr", "gencap_aud"]
PAIRS=[("cap_trans", "gen_cap_desc"), ("cap_trans", "gen_cap_asr"), ("img_trans", "img_asr"),
       ("img_trans", "img_gendesc"), ("cap_aud", "gencap_aud")]
LABELS={**CONFIG_LABELS, "gen_cap_desc": "GenCaption+GenDesc", "gen_cap_asr": "GenCaption+ASR",
        "img_gendesc": "Image+GenDesc", "img_asr": "Image+ASR", "gencap_aud": "GenCaption+Audio"}
SURR_FIELDS={"gen": "gen_caption", "gen_desc": "gen_audio_desc", "asr": "asr"}
GOLD_FIELDS={"gen_caption": "image content", "gen_audio_desc": "audio content", "asr": "audio content"}
ASR_STATUS=["ok", "empty", "filler", "loop"]
ASR_LABELS={"ok": "Plausible", "empty": "Empty", "filler": "Known filler", "loop": "Repetition loop"}
LABEL_OFFSETS={"cap_trans": (-4, 2, "right"), "aud_only": (-4, 3, "right"), "cap_aud": (4, -9, "left"),
               "img_only": (4, -3, "left"), "q_only": (4, -3, "left")}
FILLER={"thank you", "thanks for watching", "thank you for watching", "thank you so much for watching",
        "thank you so much for joining us", "you", "orchestra plays", "piano plays", "music"}
FILLER_PREFIX=("transcription by", "subtitles by")
N_BOOT=2000
N_PERM=10000


def gen_fields(config):
    spec=config_spec(config)
    return [SURR_FIELDS[r] for r in (spec["img"], spec["aud"]) if r in SURR_FIELDS]


def boot_ci(x, rng):
    idx=rng.integers(0, len(x), (N_BOOT, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def signflip_p(d, rng):
    s=rng.choice([-1, 1], (N_PERM, len(d)))
    return float((np.abs((s*d).mean(1))>=abs(d.mean())-1e-12).mean())


def words(t):
    return re.sub(r"[^a-z0-9' ]", " ", t.lower()).replace("'", "").split()


def asr_status(text, tok):
    w=words(text)
    if not w:
        return "empty"
    if tok>=400 or (len(w)>=8 and len(set(w))/len(w)<0.3):
        return "loop"
    n=" ".join(w)
    return "filler" if n in FILLER or n.startswith(FILLER_PREFIX) else "ok"


def wer(ref, hyp):
    r, h=words(ref), words(hyp)
    d=np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        prev, d[0]=d.copy(), i
        for j in range(1, len(h)+1):
            d[j]=min(prev[j]+1, d[j-1]+1, prev[j-1]+(r[i-1]!=h[j-1]))
    return d[len(h)]/max(len(r), 1)


def plot_gaps(pairs, out):
    fig, ax=plt.subplots(figsize=(COL_W, 2.5))
    scopes=[("all", INK, "All")]+[(t, AUDIO_TYPE_COLORS[t], t.capitalize()) for t in AUDIO_TYPES]
    for k, (scope, color, label) in enumerate(scopes):
        p=pairs[pairs.scope==scope].reset_index(drop=True)
        y=np.arange(len(p))+(k-1.5)*0.17
        ax.errorbar(p.gap*100, y, xerr=[(p.gap-p.ci_lo)*100, (p.ci_hi-p.gap)*100], fmt="o", ms=3.5,
                    color=color, ecolor=color, elinewidth=0.8, capsize=0, label=label)
    p=pairs[pairs.scope=="all"].reset_index(drop=True)
    ax.set_yticks(np.arange(len(p)), [f"{LABELS[g]}\nvs {LABELS[n]}" for g, n in zip(p.gold, p.generated)])
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("Accuracy gap, human-written − generated (points)")
    ax.legend(loc="lower right", ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    return save_fig(fig, out, "idea4_paired_gaps")


def plot_cost(table, out):
    fig, ax=plt.subplots(figsize=(COL_W, 2.6))
    gen=table[table.kind=="generated"]
    gold=table[table.kind!="generated"]
    for r in gen.itertuples():
        ax.plot([r.answer_ms/1000, r.total_ms/1000], [r.acc*100, r.acc*100], color=PALETTE[1], lw=0.8, alpha=0.5)
    ax.scatter(gen.answer_ms/1000, gen.acc*100, s=14, facecolor="white", edgecolor=PALETTE[1], lw=1,
               label="Generated, answer time only", zorder=3)
    for g, color, label in [(gold, PALETTE[0], "Raw / human-written"), (gen, PALETTE[1], "Generated, incl. generation")]:
        ax.errorbar(g.total_ms/1000, g.acc*100, yerr=[(g.acc-g.ci_lo)*100, (g.ci_hi-g.acc)*100], fmt="o", ms=4,
                    color=color, ecolor=color, elinewidth=0.7, capsize=0, label=label, zorder=4)
    for r in table.itertuples():
        dx, dy, ha=LABEL_OFFSETS.get(r.config, (4, 2, "left"))
        ax.annotate(LABELS[r.config], (r.total_ms/1000, r.acc*100), xytext=(dx, dy), textcoords="offset points",
                    fontsize=5.5, color=INK_2, ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.015, 30)
    ax.set_xlabel("Seconds per question (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.5)
    return save_fig(fig, out, "idea4_acc_vs_total_cost")


def plot_asr(counts, out):
    fig, ax=plt.subplots(figsize=(COL_W, 1.4))
    left=np.zeros(len(counts))
    for k, s in enumerate(ASR_STATUS):
        v=counts[s].values
        ax.barh(counts.index, v, left=left, color=PALETTE[k], edgecolor="white", lw=1, label=ASR_LABELS[s])
        for y, (l, n) in enumerate(zip(left, v)):
            if n>=3:
                ax.text(l+n/2, y, str(int(n)), ha="center", va="center", fontsize=6, color="white")
        left+=v
    ax.invert_yaxis()
    ax.set_xlabel("Test questions")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=4, handlelength=1, columnspacing=1)
    ax.grid(axis="y", visible=False)
    return save_fig(fig, out, "idea4_asr_status")


def example_md(r, gold, gen, sur, text, raw):
    s=sur[str(r.id)]
    preds={c: raw[(raw.id==r.id)&(raw.config==c)].sort_values("perm").pred_idx.tolist() for c in (gold, gen)}
    lines=[f"### id {r.id} ({r.audio_type}, {r.task_type})",
           f"- {LABELS[gold]}: {round(r.acc_gold*3)}/3 correct, predictions {[LETTERS[p] for p in preds[gold]]}",
           f"- {LABELS[gen]}: {round(r.acc_gen*3)}/3 correct, predictions {[LETTERS[p] for p in preds[gen]]}",
           f"- answer: {LETTERS[r.answer_idx]}"]
    t=text.get(r.id)
    if t:
        lines.insert(1, f"- question: {t['question'].strip()}")
        lines.insert(2, "- options: "+" | ".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(t["options"])))
    for f in gen_fields(gen):
        lines.append(f"- generated `{f}`: {s[f][:500]}")
        if t:
            lines.append(f"- human-written `{GOLD_FIELDS[f]}`: {t[GOLD_FIELDS[f]][:500]}")
    if "asr" in gen_fields(gen):
        lines.append(f"- ASR status: {s['asr_status']}")
    return "\n".join(lines)+"\n"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--surrogate-raw", default=None)
    ap.add_argument("--base-raw", default="results/midterm/raw_base_test.jsonl")
    ap.add_argument("--surrogates", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--cap-limit", type=int, default=256)
    args=ap.parse_args()
    cfg=load_config(args.config)
    mcfg=cfg["midterm"]
    rng=np.random.default_rng(cfg["seed"])
    out=repo_path(args.out or f"{mcfg['out_dir']}/idea4")
    (out/"tables").mkdir(parents=True, exist_ok=True)
    set_plot_style()

    raw=pd.DataFrame(read_jsonl(repo_path(args.surrogate_raw or mcfg["surrogate_out"]))+read_jsonl(repo_path(args.base_raw)))
    raw=raw[raw.config.isin(GOLD+GEN)].drop_duplicates(["id", "config", "perm"])
    n_perms=raw.perm.nunique()
    complete=raw.groupby("id").size()==len(GOLD+GEN)*n_perms
    raw=raw[raw.id.isin(complete[complete].index)]
    sur=json.loads(repo_path(args.surrogates or mcfg["surrogates"]).read_text())
    text={}
    if args.text:
        from src.data import load_omnibench, text_records
        text={r["index"]: r for r in text_records(load_omnibench(cfg))[0]}

    for k, s in sur.items():
        s["asr_status"]=asr_status(s["asr"], s["asr_tok"])

    item=raw.groupby(["id", "config"]).agg(acc=("correct", "mean"), tok=("n_total_tok", "mean"),
                                            ms=("latency_ms", "mean"), audio_type=("audio_type", "first"),
                                            task_type=("task_type", "first"),
                                            answer_idx=("answer_idx", "first")).reset_index()
    for col, suffix in [("gen_ms", "_ms"), ("gen_out_tok", "_tok"), ("gen_in_tok", "_in_tok")]:
        item[col]=[sum(sur[str(i)].get(f+suffix, 0) for f in gen_fields(c)) for i, c in zip(item.id, item.config)]
    item["total_ms"]=item.ms+item.gen_ms
    item["total_tok"]=item.tok+item.gen_in_tok+item.gen_out_tok
    item.to_csv(out/"tables"/"items.csv", index=False)

    rows=[]
    for c in GOLD+GEN:
        g=item[item.config==c]
        lo, hi=boot_ci(g.acc.values, rng)
        rows.append({"config": c, "label": LABELS[c], "kind": "generated" if c in GEN else "raw/gold",
                     "n_items": len(g), "acc": g.acc.mean(), "ci_lo": lo, "ci_hi": hi,
                     "acc_perm0": raw[(raw.config==c)&(raw.perm==0)].correct.mean(),
                     **{f"acc_{t.replace(' ', '_')}": g[g.audio_type==t].acc.mean() for t in AUDIO_TYPES},
                     "answer_tok": g.tok.mean(), "answer_ms": g.ms.mean(), "gen_in_tok": g.gen_in_tok.mean(),
                     "gen_out_tok": g.gen_out_tok.mean(), "gen_ms": g.gen_ms.mean(),
                     "total_tok": g.total_tok.mean(), "total_ms": g.total_ms.mean()})
    table=pd.DataFrame(rows)
    table.to_csv(out/"tables"/"configs.csv", index=False)

    wide=item.pivot(index="id", columns="config", values="acc")
    meta=item.drop_duplicates("id").set_index("id")[["audio_type", "task_type", "answer_idx"]]
    prow, examples=[], ["# Idea 4 failure examples\n",
                        "Items where the human-written config is correct in at least 2 of 3 option orders and the "
                        "generated config in at most 1 of 3.\n"]
    for gold, gen in PAIRS:
        for scope in ["all"]+AUDIO_TYPES:
            ids=meta.index if scope=="all" else meta.index[meta.audio_type==scope]
            d=(wide.loc[ids, gold]-wide.loc[ids, gen]).values
            lo, hi=boot_ci(d, rng)
            prow.append({"gold": gold, "generated": gen, "scope": scope, "n_items": len(d),
                         "acc_gold": wide.loc[ids, gold].mean(), "acc_generated": wide.loc[ids, gen].mean(),
                         "gap": d.mean(), "ci_lo": lo, "ci_hi": hi, "p_signflip": signflip_p(d, rng),
                         "n_gold_only": int(((wide.loc[ids, gold]>=2/3)&(wide.loc[ids, gen]<=1/3)).sum()),
                         "n_generated_only": int(((wide.loc[ids, gen]>=2/3)&(wide.loc[ids, gold]<=1/3)).sum())})
        f=meta.assign(acc_gold=wide[gold], acc_gen=wide[gen]).reset_index()
        f=f[(f.acc_gold>=2/3)&(f.acc_gen<=1/3)].sort_values(["acc_gen", "audio_type", "id"])
        f=f.groupby("audio_type").head(max(1, args.n_examples//len(AUDIO_TYPES)))
        examples.append(f"\n## {LABELS[gold]} vs {LABELS[gen]} ({len(f)} shown)\n")
        examples+=[example_md(r, gold, gen, sur, text, raw) for r in f.itertuples()]
    pairs=pd.DataFrame(prow)
    pairs.to_csv(out/"tables"/"paired.csv", index=False)
    (out/"examples.md").write_text("\n".join(examples))

    sdf=pd.DataFrame([{"id": int(k), **v} for k, v in sur.items()]).set_index("id")
    sdf=sdf.loc[sdf.index.isin(meta.index)]
    counts=pd.crosstab(sdf.audio_type, sdf.asr_status).reindex(index=AUDIO_TYPES, columns=ASR_STATUS, fill_value=0)
    counts.to_csv(out/"tables"/"asr_status.csv")
    by_status=sdf[["asr_status", "audio_type"]].join(wide[["img_asr", "img_trans", "gen_cap_asr", "cap_trans"]])
    by_status.groupby(["audio_type", "asr_status"]).agg(n=("img_asr", "size"), acc_img_asr=("img_asr", "mean"),
                                                         acc_img_trans=("img_trans", "mean"),
                                                         acc_gen_cap_asr=("gen_cap_asr", "mean"),
                                                         acc_cap_trans=("cap_trans", "mean")
                                                         ).to_csv(out/"tables"/"acc_by_asr_status.csv")
    limits={f: {t: int((sdf[sdf.audio_type==t][f+"_tok"]>=args.cap_limit).sum()) for t in AUDIO_TYPES}
            for f in ["gen_caption", "gen_audio_desc"]}
    gen_cost={f: {"mean_ms": float(sdf[f+"_ms"].mean()), "mean_out_tok": float(sdf[f+"_tok"].mean())}
              for f in ["gen_caption", "gen_audio_desc", "asr"]}
    if text:
        sp=sdf[sdf.audio_type=="speech"]
        sdf.loc[sp.index, "asr_wer"]=[wer(text[i]["audio content"], a) for i, a in zip(sp.index, sp.asr)]
    sdf.drop(columns=["gen_caption", "gen_audio_desc", "asr"]).to_csv(out/"tables"/"surrogates.csv")

    figs=[plot_gaps(pairs, out), plot_cost(table, out), plot_asr(counts, out)]
    summary={"n_items": int(len(meta)), "n_perms": int(n_perms),
             "configs": table.set_index("config")[["acc", "ci_lo", "ci_hi", "answer_ms", "total_ms",
                                                   "total_tok"]].round(4).to_dict("index"),
             "paired_all": pairs[pairs.scope=="all"].round(4).to_dict("records"),
             "asr_status": counts.to_dict("index"), "hit_token_limit": limits, "generation_cost": gen_cost}
    if text:
        summary["asr_wer_speech_median"]=float(sdf.asr_wer.median())
    (out/"summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print(table[["config", "acc", "ci_lo", "ci_hi", "answer_ms", "gen_ms", "total_ms"]].round(3).to_string(index=False))
    print(pairs[pairs.scope=="all"][["gold", "generated", "gap", "ci_lo", "ci_hi", "p_signflip",
                                     "n_gold_only", "n_generated_only"]].round(3).to_string(index=False))
    print(counts.to_string())
    print("hit token limit:", limits)
    print("wrote", out, *[f.name for f in figs])


if __name__ == "__main__":
    main()
