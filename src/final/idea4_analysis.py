"""Idea 4 (final report): generated text in place of the raw audio (and image), on the 242-item Idea 4 subset.

Reads the score bank (base + idea4 sources), the generated texts and their timings (results/final/idea4_surrogates.json)
and, when present, OmniBench's text fields (results/final/omnibench_text.json, from `--sections text` on Modal).
No model is run here.

    python -m src.final.idea4_analysis          # -> results/final/idea4/ (report.md, CSVs, figures)

Report sections:
  main table           every Idea 4 config and the base references: accuracy (95% CI) overall and by audio type;
                       answering cost (tokens, ms) and generation cost (ms per item on the L4), end to end
  research questions   paired differences for RQ4.1 (routing + fallback), RQ4.2 (question conditioning), RQ4.3 (masking)
  speech threshold     routing on AST P(Speech) >= theta, theta swept from 0 to 1. Rebuilt per item from the branch
                       configs that were scored on every item (notebook §9c); theta also chosen inside the folds; and
                       routing on the true audio-type label
  generation length    80 / 120 / 256 tokens: accuracy, generation time, how often the cap is hit
  break-even k         generate once, then answer k questions about the same clip: C(k) = C_answer + C_generate / k,
                       against answering from the raw input every time
  translation quality  (needs the text export) how often each text names the correct option or a wrong one, its word
                       overlap with the human description, and accuracy when the answer is or is not named
  leakage, all items   (needs the text export) how often the human transcript names the answer, on all 1,141 items,
                       and the image+transcript vs image+audio gap with and without that
  midterm test set     the 120 midterm test items: this run next to the midterm's numbers
  coding sheet         (needs the text export) transcript_coding.csv: every item where the raw audio beats the human
                       transcript or the reverse, to hand-code what the transcript loses (plan §6.3c)
"""
import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.analysis_a.prompts import config_spec
from src.common import load_config, repo_path
from src.data import AUDIO_TYPES
from src.final.bank import load_bank, load_folds, load_groups, matrix
from src.final.stats import Resampler, holm, md_table, pct, with_ci

OUT_DIR = "results/final/idea4"
TEXT = "results/final/omnibench_text.json"
MIDTERM = "results/midterm/idea4/summary.json"

BASE_REFS = ["full", "img_trans", "cap_aud", "cap_trans", "img_only", "aud_only", "q_only"]
MAIN = ["img_pipe", "gencap_pipe", "img_route", "img_route_fb", "img_qdesc", "img_masked", "cap_masked",
        "img_asr", "img_gendesc", "gen_cap_desc", "gen_cap_asr"]
BRANCHES = ["img_tags", "img_tagsq", "gencap_tagsq", "gencap_aud"]  # scored on every item for the threshold sweep
LEN_BASES = ["img_qdesc", "img_pipe", "gencap_pipe", "img_gendesc", "gen_cap_desc", "gen_cap_asr"]
LENS = (80, 120, 256)  # 256 is the unsuffixed config
NO_MEDIA = {"text", "masked", None}  # audio/image representations that need no media file
LABELS = {
    "full": "image + audio", "img_trans": "image + human transcript", "cap_aud": "human caption + audio",
    "cap_trans": "human caption + transcript", "img_only": "image only", "aud_only": "audio only",
    "q_only": "question only", "img_pipe": "image + pipeline text", "gencap_pipe": "gen. caption + pipeline text",
    "img_route": "image + routed text (no fallback)", "img_route_fb": "image + routed text (fallback)",
    "img_qdesc": "image + question-conditioned description", "img_masked": "image + masked human transcript",
    "cap_masked": "human caption + masked transcript", "img_asr": "image + Whisper transcript",
    "img_gendesc": "image + generic description", "gen_cap_desc": "gen. caption + generic description",
    "gen_cap_asr": "gen. caption + Whisper transcript", "img_tags": "image + AST tags",
    "img_tagsq": "image + tags + question-conditioned description", "gencap_tagsq": "gen. caption + tags + q-cond. desc.",
    "gencap_aud": "gen. caption + audio"}

# (question, A, B, what A - B measures)
COMPARISONS = [
    ("RQ4.1", "img_pipe", "img_asr", "the pipeline (AST routing, fallback, q-conditioned text) vs Whisper alone"),
    ("RQ4.1", "img_pipe", "img_gendesc", "the pipeline vs the midterm's generic audio description"),
    ("RQ4.1", "img_route_fb", "img_route", "falling back to raw audio when the transcript is broken"),
    ("RQ4.1", "img_tagsq", "img_tags", "non-speech: tags + q-conditioned description vs tags alone"),
    ("RQ4.1", "img_pipe", "full", "the pipeline vs the raw audio"),
    ("RQ4.1", "img_pipe", "img_trans", "the pipeline vs the human transcript"),
    ("RQ4.1", "img_pipe", "img_masked", "the pipeline vs the human transcript with option words masked"),
    ("RQ4.1", "gencap_pipe", "cap_trans", "all generated text vs all human text"),
    ("RQ4.1", "gencap_aud", "full", "a generated caption instead of the raw image (raw audio kept)"),
    ("RQ4.2", "img_qdesc", "img_gendesc", "question-conditioned vs generic audio description"),
    ("RQ4.3", "img_masked", "img_trans", "masking the option words in the human transcript"),
    ("RQ4.3", "cap_masked", "cap_trans", "masking the option words (human caption + transcript)"),
    ("RQ4.3", "img_masked", "full", "the masked human transcript vs the raw audio"),
]


# ----------------------------------------------------------------------------- data
class Data:
    """Per-item tables on the Idea 4 subset (rows in the order of data/final_idea4_ids.json)."""

    def __init__(self, cfg):
        self.ids = json.loads(repo_path(cfg["final"]["idea4_ids"]).read_text())["ids"]
        bank = load_bank(cfg, sources=("base", "idea4"))
        if bank.empty or not bank.rep.str.startswith("i4:").any():
            raise SystemExit("score bank without Idea 4 results: unpack final_results.tar.gz in the repo root first")
        bank = bank.assign(rep=bank.rep.str.replace("i4:", "", regex=False))

        def m(value, items=self.ids):
            return matrix(bank, value, complete=False).reindex(items)
        self.acc, self.logp = m("acc"), m("logp")
        self.answer_s, self.tokens = m("latency_s"), m("cost")
        self.decode_s = m("decode_s").mean(axis=1)
        reps = BASE_REFS + MAIN + BRANCHES + [f"{b}_t{n}" for b in LEN_BASES for n in LENS[:-1]]
        missing = [r for r in reps if r not in self.acc.columns or self.acc[r].isna().any()]
        if missing:
            raise SystemExit(f"score bank: Idea 4 items without scores for {missing[:5]}")
        first = bank.drop_duplicates("id").set_index("id")
        self.audio_type = first.audio_type.reindex(self.ids)
        self.task_type = first.task_type.reindex(self.ids)
        sur = json.loads(repo_path(cfg["final"]["idea4_surrogates"]).read_text())
        self.sur = pd.DataFrame.from_dict({int(k): v for k, v in sur.items()}, orient="index").reindex(self.ids)
        if self.sur.isna().all(axis=1).any():
            raise SystemExit("idea4_surrogates.json does not cover every Idea 4 item")
        fold = {i: k for k, f in enumerate(load_folds(cfg)) for i in f}
        self.fold = pd.Series([fold[i] for i in self.ids], index=self.ids)
        self.midterm_test = pd.Series(np.isin(self.ids, load_groups(cfg)["midterm_test"]), index=self.ids)
        # all 1,141 items, for the leakage section and the coding sheet
        self.acc_all = matrix(bank[bank.source == "base"], "acc")
        self.audio_type_all = first.audio_type.reindex(self.acc_all.index)
        self.task_type_all = first.task_type.reindex(self.acc_all.index)
        p = repo_path(TEXT)
        self.text = {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else None


def gen_seconds(D, config):
    """Seconds per item to produce the generated text a config reads (0 when it reads none). The pipeline runs AST
    on every clip, then Whisper on clips it calls speech, else the question-conditioned description."""
    spec, S = config_spec(config), D.sur
    t = f"_t{spec['gen_len']}" if "gen_len" in spec else ""
    ms = np.zeros(len(S))
    if spec["img"] == "gen":
        ms += S[f"gen_caption{t}_ms"].astype(float)
    aud, speech = spec["aud"], S.pred_speech.astype(bool).to_numpy()
    ast, asr = S.ast_ms.astype(float).to_numpy(), S.asr_ms.astype(float).to_numpy()
    q = S[f"q_audio_desc{t}_ms"].astype(float).to_numpy() if f"q_audio_desc{t}_ms" in S else None
    ms += {"gen_desc": lambda: S[f"gen_audio_desc{t}_ms"].astype(float).to_numpy(), "asr": lambda: asr,
           "route": lambda: ast + np.where(speech, asr, 0.0), "route_fb": lambda: ast + np.where(speech, asr, 0.0),
           "qdesc": lambda: q, "pipe": lambda: ast + np.where(speech, asr, q), "tags": lambda: ast,
           "tagsq": lambda: ast + q}.get(aud, lambda: 0.0)()
    return pd.Series(ms / 1000, index=D.ids)


def end_to_end(D, config):
    """Seconds per item: read the media (if anything needs them), generate, answer."""
    spec = config_spec(config)
    media = spec["img"] not in NO_MEDIA or spec["aud"] not in NO_MEDIA
    return D.decode_s * media + gen_seconds(D, config) + D.answer_s[config]


# ----------------------------------------------------------------------------- text helpers
STOP = set("""a an the of in on at to for with and or is are was were be been being it its this that these those he she
they his her their them him by from as about into not no so but has have had do does did i you we me my your our there
what which who whom how why when where will would can could should may might must also very just than then out up down
over some any all each more most other such only own same too one two""".split())


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


def words(s):
    """Content words, with a plain plural -s removed (cars -> car), as the masking in Idea 4 also matches singulars."""
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in norm(s).split() if w not in STOP}


def option_overlap(text, options, j):
    """Share of option j's distinctive content words (in no other option) that appear in the text; NaN if none."""
    own = words(options[j]) - set().union(*(words(o) for k, o in enumerate(options) if k != j))
    return len(own & words(text)) / len(own) if own else np.nan


def names(text, options, j, share=0.5):
    """Whether the text contains at least `share` of option j's distinctive words (a lexical check)."""
    o = option_overlap(text, options, j)
    return bool(o >= share) if not np.isnan(o) else False


def token_f1(a, b):
    ta, tb = Counter(norm(a).split()), Counter(norm(b).split())
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


# ----------------------------------------------------------------------------- analyses
class Stats:
    def __init__(self, n_boot, seed):
        self.n_boot, self.seed, self.cache = n_boot, seed, {}

    def ci(self, v):
        v = np.asarray(v, float)
        if len(v) not in self.cache:
            self.cache[len(v)] = Resampler(len(v), self.n_boot, [self.seed, len(v)])
        lo, hi = self.cache[len(v)].ci(v)
        return float(v.mean()), float(lo), float(hi)

    def test(self, d):
        d = np.asarray(d, float)
        return (*self.ci(d), self.cache[len(d)].p_signflip(d))


def routed(D, theta, speech_branch, other_branch, fallback=None, label=False):
    """Per-item accuracy when clips called speech (AST P(Speech) >= theta, or the true label) go to speech_branch
    (or to `fallback` when Whisper's output was flagged as broken) and the rest to other_branch."""
    speech = (D.audio_type == "speech").to_numpy() if label else (D.sur.ast_p_speech.astype(float) >= theta).to_numpy()
    sp = D.acc[speech_branch].to_numpy()
    if fallback is not None:
        sp = np.where(D.sur.asr_failed.astype(bool).to_numpy(), D.acc[fallback].to_numpy(), sp)
    return pd.Series(np.where(speech, sp, D.acc[other_branch].to_numpy()), index=D.ids)


SYSTEMS = {  # name -> (speech branch, other branch, fallback for broken transcripts, the config it reproduces at 0.5)
    "pipeline": ("img_asr", "img_tagsq", "full", "img_pipe"),
    "route + fallback": ("img_asr", "img_tags", "full", "img_route_fb"),
    "route, no fallback": ("img_asr", "img_tags", None, "img_route"),
    "pipeline, generated caption": ("gen_cap_asr", "gencap_tagsq", "gencap_aud", "gencap_pipe"),
}
THETAS = np.round(np.r_[np.arange(0, 1.0001, 0.02), 1.01], 3)  # 1.01: nothing is called speech


def threshold_sweep(D):
    rows, check = [], {}
    y = (D.audio_type == "speech").to_numpy()
    p = D.sur.ast_p_speech.astype(float).to_numpy()
    for name, (sb, ob, fb, cfg) in SYSTEMS.items():
        rebuilt = routed(D, 0.5, sb, ob, fb)
        check[name] = float(np.abs(rebuilt - D.acc[cfg]).max())  # must be 0: the same inputs were scored
        for th in THETAS:
            a = routed(D, th, sb, ob, fb)
            called = p >= th
            rows.append({"system": name, "theta": th, "acc": a.mean(), "called_speech": called.mean(),
                         "speech_recall": called[y].mean(), "speech_precision": y[called].mean() if called.any() else np.nan})
        rows.append({"system": name, "theta": "true label", "acc": routed(D, 0, sb, ob, fb, label=True).mean(),
                     "called_speech": y.mean(), "speech_recall": 1.0, "speech_precision": 1.0})
    return pd.DataFrame(rows), check


def cv_threshold(D, system):
    """theta chosen on the other folds' Idea 4 items (highest accuracy; ties: closest to 0.5), applied to each fold."""
    sb, ob, fb, _ = SYSTEMS[system]
    acc = {th: routed(D, th, sb, ob, fb) for th in THETAS}
    out, chosen = pd.Series(np.nan, index=D.ids), []
    for k in sorted(D.fold.unique()):
        tr, te = D.fold != k, D.fold == k
        th = max(THETAS, key=lambda t: (round(acc[t][tr].mean(), 12), -abs(t - 0.5)))
        chosen.append(th)
        out[te] = acc[th][te]
    return out, chosen


def translation_quality(D):
    """Per text: how often it names the correct option / a wrong option, word overlap with the human description,
    and the accuracy of the config that reads it when the answer is or is not named."""
    fields = [("human transcript", "audio content", "img_trans"), ("masked human transcript", "masked_audio_content", "img_masked"),
              ("Whisper transcript", "asr", "img_asr"), ("generic description", "gen_audio_desc", "img_gendesc"),
              ("question-conditioned description", "q_audio_desc", "img_qdesc"), ("AST tags", "tag_text", "img_tags"),
              ("tags + q-cond. description", "tagsq_text", "img_tagsq"), ("routed text", "route_text", "img_route"),
              ("pipeline text", "pipe_text", "img_pipe")]
    rows = []
    for label, field, config in fields:
        named, wrong, f1, share = [], [], [], []
        for i in D.ids:
            t = D.text[i]
            txt = t["audio content"] if field == "audio content" else D.sur.at[i, field]
            a, opts = t["answer_idx"], t["options"]
            share.append(option_overlap(txt, opts, a))
            named.append(names(txt, opts, a))
            wrong.append(any(names(txt, opts, j) for j in range(len(opts)) if j != a))
            f1.append(np.nan if field == "audio content" else token_f1(txt, t["audio content"]))
        named, wrong = np.array(named), np.array(wrong)
        acc = D.acc[config].to_numpy()
        rows.append({"text": label, "config": config, "answer words in text": np.nanmean(share),
                     "names answer": named.mean(), "names a wrong option": wrong.mean(),
                     "word F1 vs human": np.nanmean(f1) if field != "audio content" else np.nan,
                     "accuracy": acc.mean(), "accuracy | answer named": acc[named].mean() if named.any() else np.nan,
                     "accuracy | not named": acc[~named].mean() if (~named).any() else np.nan, "items named": int(named.sum())})
    return pd.DataFrame(rows)


def leakage_all(D, st):
    """All 1,141 items: does the human transcript name the correct option, and how much does image+transcript gain
    over image+audio with and without that."""
    ids = D.acc_all.index
    named = np.array([names(D.text[i]["audio content"], D.text[i]["options"], D.text[i]["answer_idx"]) for i in ids])
    gain = (D.acc_all["img_trans"] - D.acc_all["full"]).to_numpy()
    rows = []
    for t in AUDIO_TYPES + ["all"]:
        m = np.ones(len(ids), bool) if t == "all" else (D.audio_type_all == t).to_numpy()
        row = {"audio type": t, "items": int(m.sum()), "transcript names the answer": named[m].mean()}
        for lab, sel in (("named", m & named), ("not named", m & ~named)):
            row[f"I+T - I+A, {lab}"] = st.ci(gain[sel]) if sel.sum() > 1 else "-"
        rows.append(row)
    return rows, named


def coding_sheet(D, named_all, seed):
    """Items where the raw audio clearly beats the human transcript (>= 2/3 of orders vs <= 1/3) or the reverse,
    for both pairs: image + audio vs image + transcript, and audio only vs caption + transcript."""
    a = D.acc_all
    hi, lo = a.ge(0.6), a.le(0.4)
    pairs = {"I+A beats I+T": hi.full & lo.img_trans, "I+T beats I+A": hi.img_trans & lo.full,
             "A beats C+T": hi.aud_only & lo.cap_trans, "C+T beats A": hi.cap_trans & lo.aud_only}
    keep = pd.concat(pairs, axis=1)
    keep = keep[keep.any(axis=1)]
    named = pd.Series(named_all, index=a.index)
    rows = []
    for i, flags in keep.iterrows():
        t = D.text[i]
        audio_wins = bool(flags["I+A beats I+T"] or flags["A beats C+T"])
        rows.append({"id": i, "direction": "raw audio wins" if audio_wins else "transcript wins",
                     "pairs": "; ".join(k for k, v in flags.items() if v), "audio type": D.audio_type_all[i],
                     "task type": D.task_type_all[i], "transcript names the answer": bool(named[i]),
                     "answer words in transcript": round(option_overlap(t["audio content"], t["options"], t["answer_idx"]), 2),
                     **{f"acc {c}": round(a.at[i, c], 2) for c in ("full", "img_trans", "aud_only", "cap_trans")},
                     "question": t["question"], "options": " | ".join(t["options"]), "answer": t["answer"],
                     "human transcript": t["audio content"], "category": "", "notes": ""})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    # a random coding order within each direction (fixed seed): code from the top and stop at any point, and the
    # rows coded so far are a random sample of that direction
    rng = np.random.default_rng(seed)
    df["coding order"] = 0
    for d in df.direction.unique():
        m = df.direction == d
        df.loc[m, "coding order"] = rng.permutation(int(m.sum())) + 1
    cols = ["coding order"] + [c for c in df.columns if c != "coding order"]
    return df[cols].sort_values(["direction", "coding order"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    D, st = Data(cfg), Stats(args.n_boot, cfg["seed"])
    out = repo_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(D.ids)
    tables, md = {}, [f"# Idea 4: generated text in place of the raw input ({n} items)\n",
                      f"Accuracy is the mean over the 3 option orders; 95% CIs from {args.n_boot:,} bootstrap resamples "
                      "over items; paired tests are sign-flip tests, Holm-adjusted. Times are per item on the L4: "
                      "*answer* = the scoring pass (forward + preprocessing), *generate* = producing the text it reads "
                      "(AST, Whisper, Qwen captions/descriptions), *end to end* adds reading the media files. Tokens: "
                      "answering input tokens relative to the full input. Generated by `python -m src.final.idea4_analysis`.\n"]

    # ---- main table
    rows = []
    for c in BASE_REFS + MAIN + BRANCHES:
        a = st.ci(D.acc[c])
        row = {"config": c, "input": LABELS[c], "acc": a[0], "lo": a[1], "hi": a[2]}
        for t in AUDIO_TYPES:
            row[t] = D.acc[c][D.audio_type == t].mean()
        row["macro"] = np.mean([row[t] for t in AUDIO_TYPES])
        row.update({"tokens": D.tokens[c].mean(), "answer_ms": 1000 * D.answer_s[c].mean(),
                    "generate_ms": 1000 * gen_seconds(D, c).mean(), "end_to_end_ms": 1000 * end_to_end(D, c).mean()})
        rows.append(row)
    tables["main"] = df = pd.DataFrame(rows)
    counts = D.audio_type.value_counts()
    md += ["## Accuracy and cost\n", md_table(pd.DataFrame({
        "config": df.config, "input": df.input, "accuracy": [with_ci(*v) for v in zip(df.acc, df.lo, df.hi)],
        **{f"{t} (n={counts.get(t, 0)})": df[t].map(pct) for t in AUDIO_TYPES}, "macro": df.macro.map(pct),
        "tokens": df.tokens.map(lambda v: f"{v:.2f}"), "answer ms": df.answer_ms.map(lambda v: f"{v:.0f}"),
        "generate ms": df.generate_ms.map(lambda v: f"{v:.0f}"), "end to end ms": df.end_to_end_ms.map(lambda v: f"{v:.0f}")})),
        "", "The last four rows are the routing branches scored on every item for the threshold sweep.", ""]

    # ---- research questions
    rows = []
    for q, a, b, what in COMPARISONS:
        m, lo, hi, p = st.test(D.acc[a] - D.acc[b])
        rows.append({"question": q, "A": a, "B": b, "what": what, "acc_A": D.acc[a].mean(), "acc_B": D.acc[b].mean(),
                     "diff": m, "lo": lo, "hi": hi, "p": p, "items_differ": int((D.acc[a] != D.acc[b]).sum())})
    tables["comparisons"] = df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p)
    md += ["## Research questions (A - B, paired over items; Holm over the whole table)\n", md_table(pd.DataFrame({
        "": df.question, "A - B": df.A + " - " + df.B, "what": df.what, "A": df.acc_A.map(pct), "B": df.acc_B.map(pct),
        "difference (pts)": [with_ci(*v, signed=True) for v in zip(df["diff"], df.lo, df.hi)],
        "items differ": df.items_differ, "p_holm": df.p_holm.map(lambda v: f"{v:.4f}")})), ""]

    # ---- speech threshold
    sweep, check = threshold_sweep(D)
    if max(check.values()) > 1e-9:
        print(f"WARNING: rebuilt routing at theta=0.5 differs from the scored configs: {check}")
    tables["threshold_sweep"] = sweep
    rows = []
    for name in SYSTEMS:
        s = sweep[(sweep.system == name) & (sweep.theta != "true label")].astype({"theta": float})
        best = s.loc[s.acc.idxmax()]
        cv, chosen = cv_threshold(D, name)
        at05 = s[np.isclose(s.theta, 0.5)].iloc[0]
        lab = sweep[(sweep.system == name) & (sweep.theta == "true label")].iloc[0]
        rows.append({"system": name, "theta 0.5": pct(at05.acc), "best theta (in-sample)": f"{pct(best.acc)} at {best.theta:g}",
                     "theta chosen in the folds": f"{with_ci(*st.ci(cv))} (thetas {', '.join(f'{c:g}' for c in chosen)})",
                     "true audio-type label": pct(lab.acc), "all called speech (theta 0)": pct(s.iloc[0].acc),
                     "none called speech": pct(s.iloc[-1].acc)})
    y = (D.audio_type == "speech").to_numpy()
    called = D.sur.pred_speech.astype(bool).to_numpy()
    md += ["## Speech threshold (AST P(Speech) >= theta)\n",
           f"At theta = 0.5, AST calls {called.mean():.0%} of clips speech: recall {called[y].mean():.0%} of the "
           f"{y.sum()} speech clips, precision {y[called].mean():.0%}. Rebuilding each system from its branches "
           f"reproduces the scored configs exactly at 0.5 (max per-item difference {max(check.values()):.1e}). "
           "*theta chosen in the folds*: picked on the other folds' items, so it is an honest estimate; *best theta "
           "(in-sample)* is optimistic.\n", md_table(pd.DataFrame(rows)), ""]

    # ---- generation length
    rows = []
    for b in LEN_BASES:
        for L in LENS:
            c = b if L == 256 else f"{b}_t{L}"
            spec, sfx = config_spec(c), f"_t{L}"
            caps = []
            for fld in (["gen_caption"] if spec["img"] == "gen" else []) + \
                       {"qdesc": ["q_audio_desc"], "gen_desc": ["gen_audio_desc"], "pipe": ["q_audio_desc"]}.get(spec["aud"], []):
                tok = D.sur[f"{fld}{sfx}_tok"].astype(float)
                if spec["aud"] == "pipe" and fld == "q_audio_desc":
                    tok = tok[~D.sur.pred_speech.astype(bool)]  # the pipeline only describes non-speech clips
                caps.append(f"{fld} {np.mean(tok >= L):.0%}")
            d = st.ci(D.acc[c] - D.acc[b]) if L != 256 else (0.0, 0.0, 0.0)
            rows.append({"config": b, "max tokens": L, "acc": D.acc[c].mean(), "vs 256": d,
                         "generate_ms": 1000 * gen_seconds(D, c).mean(), "hit the cap": ", ".join(caps)})
    tables["generation_length"] = df = pd.DataFrame(rows)
    md += ["## Generation length\n", "*hit the cap*: share of generated texts that used every allowed token.\n",
           md_table(pd.DataFrame({"config": df.config, "max tokens": df["max tokens"], "accuracy": df.acc.map(pct),
                                  "vs 256 (pts)": [with_ci(*v, signed=True) if L != 256 else "-" for v, L in zip(df["vs 256"], df["max tokens"])],
                                  "generate ms": df.generate_ms.map(lambda v: f"{v:.0f}"), "hit the cap": df["hit the cap"]})), ""]

    # ---- break-even k
    raw = D.answer_s["full"].mean()
    rows = []
    for c in MAIN + BRANCHES:
        g = gen_seconds(D, c).mean()
        if g == 0:
            continue
        saving = raw - D.answer_s[c].mean()
        rows.append({"config": c, "acc": D.acc[c].mean(), "vs I+A (pts)": st.ci(D.acc[c] - D.acc["full"]),
                     "generate_ms": 1000 * g, "answer_ms": 1000 * D.answer_s[c].mean(), "raw_answer_ms": 1000 * raw,
                     "break_even_k": float(np.ceil(g / saving)) if saving > 0 else np.inf})
    tables["break_even"] = df = pd.DataFrame(rows).sort_values("break_even_k")
    md += ["## Break-even k: generate the text once, then answer k questions about the same clip\n",
           f"Cost per question C(k) = answer + generate / k, against answering from the raw image + audio every time "
           f"({1000 * raw:.0f} ms). Break-even k = the smallest k with C(k) <= the raw cost, i.e. generate / (raw answer "
           "- text answer) rounded up; *never* when answering from the text is not faster. Reading the media is needed "
           "either way and cancels out. Accuracy is not part of k: see *vs I+A*.\n",
           md_table(pd.DataFrame({"config": df.config, "accuracy": df.acc.map(pct),
                                  "vs I+A (pts)": [with_ci(*v, signed=True) for v in df["vs I+A (pts)"]],
                                  "generate ms": df.generate_ms.map(lambda v: f"{v:.0f}"),
                                  "answer ms": df.answer_ms.map(lambda v: f"{v:.0f}"),
                                  "break-even k": df.break_even_k.map(lambda v: "never" if np.isinf(v) else f"{v:.0f}")})), ""]

    # ---- text-based sections
    if D.text is None:
        md += ["## Translation quality, leakage, coding sheet\n",
               f"Skipped: {TEXT} not found. Export it on Modal (`modal run scripts/modal_run.py --sections text`), "
               "download it, and run this again.", ""]
    else:
        tables["translation_quality"] = tq = translation_quality(D)
        md += ["## Translation quality (does the text name the answer?)\n",
               "*Answer words in text*: share of the correct option's distinctive content words (words in no other "
               "option; stop words and plural -s ignored) that appear in the text. *Names* an option: at least half of "
               "its distinctive words appear. A lexical check: it misses paraphrases. The masked transcript is a "
               "built-in control: its option words were removed, so it should name almost nothing. *Word F1*: "
               "bag-of-words overlap with the human description.\n",
               md_table(pd.DataFrame({
                   "text": tq.text, "config": tq.config, "answer words in text": tq["answer words in text"].map(pct),
                   "names the answer": tq["names answer"].map(pct),
                   "names a wrong option": tq["names a wrong option"].map(pct),
                   "word F1 vs human": tq["word F1 vs human"].map(lambda v: "-" if np.isnan(v) else f"{v:.2f}"),
                   "accuracy": tq.accuracy.map(pct), "acc, answer named": tq["accuracy | answer named"].map(
                       lambda v: "-" if np.isnan(v) else pct(v)), "acc, not named": tq["accuracy | not named"].map(
                       lambda v: "-" if np.isnan(v) else pct(v)), "items named": tq["items named"]})), ""]
        leak, named_all = leakage_all(D, st)
        tables["leakage_all_items"] = pd.DataFrame([{k: (v[0] if isinstance(v, tuple) else (np.nan if v == "-" else v))
                                                     for k, v in r.items()} for r in leak])
        md += ["## Leakage in the human transcripts (all 1,141 items)\n",
               "How often OmniBench's human audio description names the correct option, and how much image + transcript "
               "gains over image + audio on those items and on the rest.\n",
               md_table(pd.DataFrame([{k: (with_ci(*v, signed=True) if isinstance(v, tuple) else (pct(v) if isinstance(v, float) else v))
                                       for k, v in r.items()} for r in leak])), "",
               "*Names the answer*: at least half of the correct option's distinctive words appear in the transcript.", ""]
        sheet = coding_sheet(D, named_all, cfg["seed"])
        sheet.to_csv(out / "transcript_coding.csv", index=False)
        cnt = sheet.groupby(["direction", "audio type"]).size().unstack(fill_value=0) if len(sheet) else pd.DataFrame()
        md += ["## Coding sheet: what does the transcript lose? (plan §6.3c)\n",
               f"`transcript_coding.csv` lists the {len(sheet)} items where one side is right in >= 2 of 3 option "
               "orders and the other in <= 1: image + audio vs image + transcript, and audio only vs caption + "
               "transcript. Fill in *category* by hand, one of: tone/emotion, speaker identity, non-speech sound, "
               "instrument/timbre, background event, timing/order, answer leakage, other. *transcript names the "
               "answer* is filled in automatically (lexical). Rows are in a random *coding order* within each "
               "direction: code from the top (e.g. the first 50 of each) and the coded rows are a random sample.\n",
               md_table(cnt.reset_index()) if len(cnt) else "(no items)", ""]

    # ---- midterm test set
    mt = D.midterm_test.to_numpy()
    p = repo_path(MIDTERM)
    old = json.loads(p.read_text())["configs"] if p.exists() else {}
    rows = []
    for c in BASE_REFS + ["img_asr", "img_gendesc", "gen_cap_desc", "gen_cap_asr", "img_pipe", "img_route_fb", "img_qdesc", "gencap_pipe", "img_masked"]:
        rows.append({"config": c, "input": LABELS[c], "midterm": pct(old[c]["acc"]) if c in old else "-",
                     "this run": with_ci(*st.ci(D.acc[c][mt]))})
    tables["midterm_test"] = pd.DataFrame(rows)
    md += [f"## The midterm test set ({mt.sum()} items)\n",
           "*midterm*: the midterm report's numbers (results/midterm/idea4/summary.json); *this run*: the same items in "
           "the final L4 run, where every text was generated again (greedy decoding, same prompts). Configs new in the "
           "final have no midterm number.\n", md_table(tables["midterm_test"]), ""]

    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    figures(D, tables, out)
    md += ["## Files\n", "CSVs: " + ", ".join(f"`{k}.csv`" for k in tables)
           + (", `transcript_coding.csv`" if D.text is not None else "") + ". Figures: `threshold_sweep.pdf/png`, "
           "`break_even.pdf/png`.", ""]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md[:4]))
    print(f"written: {out}/report.md, CSVs and figures")


def figures(D, tables, out):
    import matplotlib.pyplot as plt

    from src.common import COL_W, INK_2, PALETTE, save_fig, set_plot_style
    set_plot_style()
    sw = tables["threshold_sweep"]
    sw = sw[sw.theta != "true label"].astype({"theta": float})
    sw = sw[sw.theta <= 1.0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for (name, color) in zip(["pipeline", "route + fallback", "route, no fallback"], PALETTE):
        s = sw[sw.system == name]
        ax.plot(s.theta, 100 * s.acc, color=color, lw=1.4, label=name)
    for c, ls, lab in (("full", "--", "image + audio"), ("img_masked", ":", "image + masked transcript")):
        ax.axhline(100 * D.acc[c].mean(), color=INK_2, ls=ls, lw=0.8)
        ax.annotate(lab, (1.0, 100 * D.acc[c].mean()), xytext=(0, 2), textcoords="offset points", ha="right",
                    va="bottom", fontsize=5.5, color=INK_2)
    ax.axvline(0.5, color=INK_2, lw=0.6, ls="-", alpha=0.5)
    ax.set_xlabel("Speech threshold on AST P(Speech)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    save_fig(fig, out, "threshold_sweep")

    be = tables["break_even"]
    shown = [c for c in ("img_pipe", "img_qdesc", "img_asr", "gen_cap_desc") if c in set(be.config)]
    ks = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    raw = be.raw_answer_ms.iloc[0]
    ax.axhline(raw, color=INK_2, ls="--", lw=0.9)
    ax.annotate("raw image + audio, every question", (100, raw), xytext=(0, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize=5.5, color=INK_2)
    for c, color in zip(shown, PALETTE):
        r = be[be.config == c].iloc[0]
        ax.plot(ks, r.answer_ms + r.generate_ms / ks, color=color, lw=1.4, label=LABELS[c])
    ax.set_yscale("log")
    ax.set_xlabel("Questions asked about the same clip (k)")
    ax.set_ylabel("ms per question (log)")
    ax.legend(loc="upper right", fontsize=5.5)
    save_fig(fig, out, "break_even")


if __name__ == "__main__":
    main()
