"""Idea 3 (final report): an item-level allocator of image and audio budgets, and the remaining Idea 3 analyses.

The allocator runs a cheap first pass at b0 = (64 image tokens, first half of the audio). From its option
distribution (and, optionally, two more cheap passes for the KL features) a regressor g(z, b) predicts the accuracy
gain of each budget cell b over b0, and the policy answers with argmax_b g(z(x), b) - lambda * c(x; b). The first
pass is always paid; answering with b0 itself costs nothing extra. Everything comes from the score bank; no model is
run. It reuses the Idea 1 machinery (src/final/idea1_router.py): the 5 frozen folds, choices made inside the training
folds only (ridge strength by validation error, lambda on cross-fitted predictions), the cost accounting and the hull.

    python -m src.final.idea3_allocator            # -> results/final/idea3/ (report.md, CSVs, figures), a few minutes

Report sections:
  headline / target cost  the allocator against the midterm's baselines (the best single cell, and the best cell per
                          audio type, chosen in the training folds under a token budget), fixed cells and the oracle
  research questions      RQ3.1 allocator vs the baselines, RQ3.2 the KL features, and the first pass, the regressor
                          and the regression target (paired over items, cost-adjusted)
  variants and sweeps     every allocator, plus the first-pass budget b0 and the ridge strength
  cheap-pass failures     (plan §6.4a) how often each cheap pass is confidently wrong, by audio type, task type and
                          the cross-modal subset, before and after temperature scaling; a reliability diagram
  time per cell           every representation: tokens and milliseconds (forward, preprocessing, reading media)
  audio selection         (RQ3.3) centered and loudest windows vs the start of the clip, overall and by audio type
  midterm test set        the 90 Idea 3 test items, trained on the other items, next to the midterm's numbers
"""
import argparse
import json
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.common import load_config, read_jsonl, repo_path
from src.data import AUDIO_TYPES
from src.final.bank import load_bank, load_groups, matrix
from src.final.idea1_router import (REPORT_TAUS, TAUS, Bank, Frontier, Policy, Stats, added_area, choose,
                                    dist_features, fit_temperature, kl, macro, oracle_policies, pay, select_lambdas,
                                    short, spearman_rows, upper_hull)
from src.final.stats import holm, md_table, pct, with_ci

OUT_DIR = "results/final/idea3"
MIDTERM = "results/idea3/summary.json"
# the 16 budget cells (image tokens x leading audio fraction); the base configs are its corners
GRID = ["q_only", "i3:v0_a0.25", "i3:v0_a0.5", "aud_only", "i3:v64_a0", "i3:v64_a0.25", "i3:v64_a0.5", "i3:v64_a1",
        "i3:v256_a0", "i3:v256_a0.25", "i3:v256_a0.5", "i3:v256_a1", "img_only", "i3:v768_a0.25", "i3:v768_a0.5",
        "full"]
CORNER = {"q_only": "v0_a0", "aud_only": "v0_a1", "img_only": "v768_a0", "full": "v768_a1"}
KL_PASSES = ("i3:v64_a0", "i3:v0_a0.5")  # image only (64 tokens), audio only (first half)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
MIDTERM_BETAS = (0.25, 0.4, 0.6, 0.8)
SELECTION = [("i3:v0_a0.25c", "i3:v0_a0.25"), ("i3:v0_a0.25e", "i3:v0_a0.25"), ("i3:v0_a0.5c", "i3:v0_a0.5"),
             ("i3:v0_a0.5e", "i3:v0_a0.5"), ("i3:v64_a0.25e", "i3:v64_a0.25"), ("i3:v64_a0.5e", "i3:v64_a0.5")]
WINDOW = {"c": "centered", "e": "loudest"}


def cell(rep):
    return CORNER.get(rep, rep.replace("i3:", ""))


# ----------------------------------------------------------------------------- specification
@dataclass(frozen=True)
class ASpec:
    regressor: str = "ridge"        # ridge | gbt
    b0: object = "i3:v64_a0.5"      # the first pass (None: no first pass)
    kl: bool = True                 # two more cheap passes, image only and audio only, and their KL features
    question: bool = False          # SBERT question embedding (PCA 16)
    audio_type: bool = False        # one-hot audio type (needs a classifier at deployment)
    target: str = "acc"             # acc | logp: the gain over b0 the regressor predicts
    alpha: object = "auto"          # ridge strength: "auto" (chosen inside the training folds) or a number

    @property
    def passes(self):
        return tuple(dict.fromkeys(([self.b0] if self.b0 else []) + (list(KL_PASSES) if self.kl else [])))

    @property
    def label(self):
        f = [f"first pass {cell(self.b0)}" if self.b0 else "no first pass"] + (["KL"] if self.kl else [])
        f += (["question"] if self.question else []) + ["meta"] + (["audio type"] if self.audio_type else [])
        extra = [] if self.target == "acc" else ["target log p"]
        extra += [] if self.alpha == "auto" else [f"alpha={self.alpha:g}"]
        return f"{self.regressor} · " + "+".join(f) + (f" · {', '.join(extra)}" if extra else "")


def load_logits(cfg, B, reps):
    """Original-order letter logits of extra Idea 3 cells (the Idea 1 bank loads only its own passes)."""
    need = [r for r in reps if r not in B.logits]
    if not need:
        return
    pos = {i: k for k, i in enumerate(B.ids)}
    for r in need:
        B.logits[r] = np.full((B.n, 4), np.nan)
    for row in read_jsonl(repo_path(f"{cfg['idea3']['out_dir']}/raw_final.jsonl")):
        rep, k = "i3:" + row["cell"], pos.get(row["id"])
        if rep in need and row["perm"] == 0 and k is not None:
            B.logits[rep][k] = row["letter_logits"]
    if any(np.isnan(B.logits[r]).any() for r in need):
        raise SystemExit(f"score bank: original-order passes missing for {need}")


# ----------------------------------------------------------------------------- features and regressors
def build_features(B, s, tr, te, seed):
    """Standardized features for tr (fitting rows) and te; every fitted step uses tr only."""
    cols = []
    if s.question:
        from sklearn.decomposition import PCA
        cols.append(PCA(n_components=16, random_state=seed).fit(B.emb[tr]).transform(B.emb))
    cols.append(B.meta)
    if s.audio_type:
        cols.append(np.stack([B.audio_type == t for t in AUDIO_TYPES], axis=1).astype(float))
    dist = {}
    for p in s.passes:
        dist[p] = softmax(B.logits[p] / fit_temperature(B.logits[p][tr], B.gold[tr]), axis=1)
        cols.append(dist_features(dist[p]))
    if s.kl:
        pv, pa = dist[KL_PASSES[0]], dist[KL_PASSES[1]]
        cols.append(np.c_[kl(pv, pa), kl(pa, pv), pv.argmax(axis=1) == pa.argmax(axis=1)])
        if s.b0 and s.b0 not in KL_PASSES:
            pva = dist[s.b0]
            cols.append(np.c_[kl(pva, pa), kl(pva, pv)])
    X = np.hstack(cols).astype(float)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    return X[tr], X[te]


class Ridge:
    """Multi-output ridge regression, closed form; the intercept is not penalized."""

    def __init__(self, alpha, seed):
        self.alpha = alpha

    def fit(self, X, Y):
        xm, ym = X.mean(axis=0), Y.mean(axis=0)
        Xc = X - xm
        self.W = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (Y - ym))
        self.b = ym - xm @ self.W
        return self

    def predict(self, X):
        return X @ self.W + self.b


class GBT:
    """One gradient-boosted regressor per cell (fixed, conservative settings); constant targets are kept constant."""

    def __init__(self, alpha, seed):
        self.seed = seed

    def fit(self, X, Y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.models = []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if np.ptp(y) == 0:
                self.models.append(float(y[0]))
            else:
                self.models.append(HistGradientBoostingRegressor(
                    max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
                    random_state=self.seed).fit(X, y))
        return self

    def predict(self, X):
        return np.column_stack([np.full(len(X), m) if isinstance(m, float) else m.predict(X) for m in self.models])


REGRESSORS = {"ridge": Ridge, "gbt": GBT}


def targets(B, s):
    """Accuracy (or log p(gold)) of every cell, minus that of b0: the gain the allocator predicts."""
    T = (B.acc if s.target == "acc" else B.logp)[:, [B.col[r] for r in GRID]]
    return T - T[:, [GRID.index(s.b0)]] if s.b0 else T


# ----------------------------------------------------------------------------- fitting and policies
def fit_split(B, s, tr, te, taus, seed):
    """Train on tr, score te; the ridge strength and lambda are chosen on predictions cross-fitted inside tr."""
    Y, A = targets(B, s), B.acc[:, [B.col[r] for r in GRID]]
    alphas = (ALPHAS if s.alpha == "auto" else (float(s.alpha),)) if s.regressor == "ridge" else (None,)
    inner, Reg = B.fold[tr], REGRESSORS[s.regressor]
    oof = {a: np.empty((len(tr), len(GRID))) for a in alphas}
    err = dict.fromkeys(alphas, 0.0)
    for f in np.unique(inner):
        fit_pos, val_pos = np.flatnonzero(inner != f), np.flatnonzero(inner == f)
        Xa, Xv = build_features(B, s, tr[fit_pos], tr[val_pos], seed)
        for a in alphas:
            pred = Reg(a, seed).fit(Xa, Y[tr[fit_pos]]).predict(Xv)
            oof[a][val_pos] = pred
            err[a] += float(((pred - Y[tr[val_pos]]) ** 2).sum())
    a = min(alphas, key=lambda v: (err[v], -(v or 0)))
    Xtr, Xte = build_features(B, s, tr, te, seed)
    model = Reg(a, seed).fit(Xtr, Y[tr])
    tok, _ = pay(B, GRID, s.passes)
    lam_max, lam_tau = select_lambdas(oof[a], tok[tr], A[tr], taus)
    t0 = time.perf_counter()
    S = model.predict(Xte)
    infer_s = (time.perf_counter() - t0) / len(te)
    return {"te": te, "S": S, "lam_max": lam_max, "lam_tau": lam_tau, "alpha": a, "infer_s": infer_s}


def splits(B, kind):
    if kind == "folds":
        return [(np.flatnonzero(B.fold != k), np.flatnonzero(B.fold == k)) for k in np.unique(B.fold)]
    test = np.isin(B.ids, B.idea3_test)
    return [(np.flatnonzero(~test), np.flatnonzero(test))]  # "midterm": the 90 Idea 3 test items


def run_spec(B, s, taus, seed, split="folds"):
    parts = [fit_split(B, s, tr, te, taus, seed) for tr, te in splits(B, split)]
    S = np.full((B.n, len(GRID)), np.nan)
    lam = {p: np.full(B.n, np.nan) for p in ("max",) + tuple(taus)}
    for f in parts:
        S[f["te"]] = f["S"]
        lam["max"][f["te"]] = f["lam_max"]
        for t in taus:
            lam[t][f["te"]] = f["lam_tau"][t]
    return {"spec": s, "S": S, "lam": lam, "parts": parts, "infer_s": float(np.mean([f["infer_s"] for f in parts]))}


def allocator_policies(B, run, name):
    s, S = run["spec"], run["S"]
    idx = np.flatnonzero(~np.isnan(S[:, 0]))
    tok, sec = pay(B, GRID, s.passes, run["infer_s"] + (B.embed_s if s.question else 0.0))
    J, rows = np.array([B.col[r] for r in GRID]), np.arange(len(idx))
    out = {}
    for point, lam in run["lam"].items():
        c = choose(S[idx], tok[idx], lam[idx])
        out[point] = Policy(name, point, idx, B.acc[idx, J[c]], tok[idx][rows, c], sec[idx][rows, c], np.array(GRID)[c])
    return out


def table_policies(B, taus, per_type, split="folds", name=None):
    """The midterm baselines: the most accurate cell on the training items whose mean token cost is within the
    budget (or simply the most accurate, for "max"), chosen for all items or per audio type."""
    tok, sec = pay(B, GRID)
    A = B.acc[:, [B.col[r] for r in GRID]]
    points = ("max",) + tuple(taus)
    pick = {p: np.full(B.n, -1) for p in points}
    for tr, te in splits(B, split):
        groups = [(t, B.audio_type == t) for t in AUDIO_TYPES] if per_type else [("all", np.ones(B.n, bool))]
        for _, g in groups:
            trg, teg = tr[g[tr]], te[g[te]]
            if not len(teg):
                continue
            acc, cost = A[trg].mean(axis=0), tok[trg].mean(axis=0)
            for p in points:
                ok = np.ones(len(GRID), bool) if p == "max" else cost <= p + 1e-9
                if not ok.any():
                    ok = cost == cost.min()
                cand = np.flatnonzero(ok)
                best = cand[np.lexsort((cost[cand], -np.round(acc[cand], 12)))[0]]
                pick[p][teg] = best
    idx = np.flatnonzero(pick["max"] >= 0)
    name = name or ("Best cell per audio type" if per_type else "Best single cell")
    rows = np.arange(len(idx))
    return {p: Policy(name, p, idx, A[idx][rows, pick[p][idx]], tok[idx][rows, pick[p][idx]],
                      sec[idx][rows, pick[p][idx]], np.array(GRID)[pick[p][idx]]) for p in points}


# ----------------------------------------------------------------------------- the other Idea 3 analyses
def cheap_pass_failures(B, reps):
    """Original-order single pass: accuracy, calibration (ECE) before and after temperature scaling (fitted on the
    other folds), and the share of items answered wrong with top probability >= 0.8, by subgroup."""
    rows, rel = [], {}
    for p in reps:
        z, conf_t = B.logits[p], np.empty(B.n)
        for k in np.unique(B.fold):
            te = B.fold == k
            conf_t[te] = softmax(z[te] / fit_temperature(z[~te], B.gold[~te]), axis=1).max(axis=1)
        conf, right = softmax(z, axis=1).max(axis=1), z.argmax(axis=1) == B.gold

        def ece(c, m=None):
            m = np.ones(B.n, bool) if m is None else m
            bins = np.minimum((c[m] * 10).astype(int), 9)
            return sum(abs(c[m][bins == b].mean() - right[m][bins == b].mean()) * (bins == b).mean()
                       for b in range(10) if (bins == b).any())
        groups = [("all", np.ones(B.n, bool))] + [(t, B.audio_type == t) for t in AUDIO_TYPES] + \
                 [("cross-modal", B.cross_modal)] + [(f"task: {t}", B.task_type == t) for t in sorted(set(B.task_type))]
        for gname, m in groups:
            rows.append({"pass": cell(p), "group": gname, "items": int(m.sum()), "accuracy": right[m].mean(),
                         "mean max prob": conf[m].mean(), "ECE": ece(conf, m), "ECE scaled": ece(conf_t, m),
                         "confidently wrong": ((conf[m] >= 0.8) & ~right[m]).mean(),
                         "confidently wrong, scaled": ((conf_t[m] >= 0.8) & ~right[m]).mean()})
        rel[p] = (conf, conf_t, right)
    return pd.DataFrame(rows), rel


def time_per_cell(cfg):
    bank = load_bank(cfg, sources=("base", "idea3"))
    g = bank.groupby("rep").agg(accuracy=("acc", "mean"), tokens=("cost", "mean"), forward_ms=("model_s", "mean"),
                                preprocess_ms=("preproc_s", "mean"), read_media_ms=("decode_s", "mean"),
                                end_to_end_ms=("e2e_s", "mean"))
    for c in ("forward_ms", "preprocess_ms", "read_media_ms", "end_to_end_ms"):
        g[c] *= 1000
    g.index = [cell(r) if r in CORNER or r.startswith("i3:") else r for r in g.index]
    return g.sort_values("tokens")


# ----------------------------------------------------------------------------- report
def experiments():
    main = ASpec()
    specs = {"Allocator": main,
             "Allocator, no KL passes": replace(main, kl=False),
             "Allocator, no first pass (question + meta)": replace(main, b0=None, kl=False, question=True),
             "Allocator, no first pass (meta only)": replace(main, b0=None, kl=False),
             "Allocator, boosted trees": replace(main, regressor="gbt"),
             "Allocator, log p target": replace(main, target="logp"),
             "Allocator, + audio type": replace(main, audio_type=True)}
    sweeps = [replace(main, kl=False, b0=b) for b in ("i3:v64_a0.25", "i3:v0_a0.5", "i3:v256_a0.5")]
    sweeps += [replace(main, alpha=a) for a in (0.1, 1.0, 10.0, 100.0, 1000.0)]
    return specs, sweeps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    cfg, t_start = load_config(args.config), time.time()
    seed = cfg["seed"]
    B = Bank(cfg, question=True)
    B.task_type = load_bank(cfg, sources=("base",)).drop_duplicates("id").set_index("id").task_type.reindex(B.ids).to_numpy()
    B.idea3_test = load_groups(cfg)["idea3_test"]
    load_logits(cfg, B, ["i3:v64_a0.25", "i3:v256_a0.5"])
    st, out = Stats(args.n_boot, seed), repo_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs, sweeps = experiments()
    all_specs = list(dict.fromkeys(list(specs.values()) + sweeps))
    print(f"fitting {len(all_specs)} allocators x 5 folds (+ 1 on the midterm test split)", flush=True)
    runs = {s: run_spec(B, s, TAUS, seed) for s in all_specs}
    mid_runs = {s: run_spec(B, s, MIDTERM_BETAS, seed, split="midterm") for s in (specs["Allocator"],)}
    idx = np.arange(B.n)
    pols = {s: allocator_policies(B, r, s.label) for s, r in runs.items()}
    glob, ptype = table_policies(B, TAUS, False), table_policies(B, TAUS, True)
    oracle, _ = oracle_policies(B, GRID, idx, TAUS)
    F, F_sec = Frontier(B, GRID, idx), Frontier(B, GRID, idx, "sec")
    tables, md = {}, []

    def summary(name, pol, own=F):
        a, g = st.ci(pol.acc), st.ci(own.gain(pol)) if own is not None else (np.nan,) * 3
        return {"method": name, "point": pol.point, "acc": a[0], "acc_lo": a[1], "acc_hi": a[2],
                "macro_acc": macro(pol, B), "cross_modal_acc": pol.acc[B.cross_modal[pol.idx]].mean(),
                "tokens": pol.tok.mean(), "ms_per_item": 1000 * pol.sec.mean(), "gain": g[0], "gain_lo": g[1],
                "gain_hi": g[2], "oracle_gap_closed": own.gap_closed(pol) if own is not None else np.nan}

    def fmt(df):
        return pd.DataFrame({
            "method": df.method, "accuracy": [with_ci(*v) for v in zip(df.acc, df.acc_lo, df.acc_hi)],
            "macro": df.macro_acc.map(pct), "cross-modal": df.cross_modal_acc.map(pct),
            "tokens": df.tokens.map(lambda v: f"{v:.2f}"), "ms/item": df.ms_per_item.map(lambda v: f"{v:.0f}"),
            "vs hull at same cost (pts)": ["-" if np.isnan(v[0]) else with_ci(*v, signed=True)
                                           for v in zip(df.gain, df.gain_lo, df.gain_hi)],
            "oracle gap closed": df.oracle_gap_closed.map(lambda v: "-" if np.isnan(v) else pct(v, signed=True))})

    b0 = specs["Allocator"].b0
    md += [f"# Idea 3: item-level budget allocation ({B.n} items)\n",
           "Every choice (ridge strength, lambda, the baselines' cells, temperatures) is made inside the training folds; "
           "every decision is out of fold. Accuracy is the mean over the 3 option orders; *tokens*: input tokens "
           "relative to the full input, including the cheap passes; *ms/item*: end to end on the L4. *vs hull*: "
           "accuracy minus random mixing of the 16 budget cells at the same mean token cost (paired over items). "
           f"95% CIs: {args.n_boot:,} bootstrap resamples over items. The first pass b0 = {cell(b0)} costs "
           f"{B.tok[:, B.col[b0]].mean():.2f} tokens; the two KL passes ({', '.join(cell(p) for p in KL_PASSES)}) "
           f"{B.tok[:, [B.col[p] for p in KL_PASSES]].sum(axis=1).mean():.2f} more. The baselines are the midterm's: "
           "the most accurate cell in the training folds within the budget, for all items or per audio type (which "
           "needs the audio type at test time).\n"]

    # ---- headline (accuracy-tuned) and target costs
    rows = [summary(f"{cell(r)} (fixed)", Policy(cell(r), "fixed", idx, B.acc[:, B.col[r]], *(x[:, 0] for x in pay(B, [r]))), None)
            for r in ("full", "i3:v256_a1", "i3:v64_a1", b0)]
    rows += [summary(glob["max"].name, glob["max"]), summary(ptype["max"].name, ptype["max"])]
    rows += [summary(n, pols[s]["max"]) for n, s in specs.items()]
    rows.append(summary("Oracle (sees the answers)", oracle["max"]))
    tables["headline"] = pd.DataFrame(rows)
    md += ["## Headline (operating point chosen for accuracy)\n", md_table(fmt(tables["headline"])), ""]
    rows = []
    for t in REPORT_TAUS:
        rows += [summary(glob[t].name, glob[t]), summary(ptype[t].name, ptype[t])]
        rows += [summary(n, pols[specs[n]][t]) for n in ("Allocator", "Allocator, no KL passes",
                                                         "Allocator, no first pass (question + meta)")]
    tables["matched_cost"] = df = pd.DataFrame(rows)
    md += ["## At a token budget (chosen in the training folds)\n",
           "The allocator cannot cost less than its cheap passes; the baselines take the best cell within the budget, "
           "so they can come in under it.\n",
           md_table(pd.concat([df.point.map(lambda v: f"{v:g}").rename("budget"), fmt(df)], axis=1)), ""]

    # ---- research questions (cost-adjusted against the grid hull, paired)
    main_s = specs["Allocator"]
    cmp = [("RQ3.1", pols[main_s], glob, "allocator vs the best single cell"),
           ("RQ3.1", pols[main_s], ptype, "allocator vs the best cell per audio type"),
           ("RQ3.1", ptype, glob, "per audio type vs single cell (the midterm comparison)"),
           ("RQ3.2", pols[main_s], pols[specs["Allocator, no KL passes"]], "adding the KL passes and features"),
           ("first pass", pols[specs["Allocator, no KL passes"]], pols[specs["Allocator, no first pass (question + meta)"]],
            "a first pass vs question + meta features only"),
           ("regressor", pols[specs["Allocator, boosted trees"]], pols[main_s], "boosted trees vs ridge"),
           ("target", pols[specs["Allocator, log p target"]], pols[main_s], "predicting the log p gain vs the accuracy gain")]
    rows = []
    for fam, a, b, what in cmp:
        for p in ("max", 0.5):
            m, lo, hi, pv = st.test(F.gain(a[p]) - F.gain(b[p]))
            rows.append({"question": fam, "comparison": what, "point": "accuracy-tuned" if p == "max" else f"budget {p:g}",
                         "acc_A": a[p].acc.mean(), "tok_A": a[p].tok.mean(), "acc_B": b[p].acc.mean(),
                         "tok_B": b[p].tok.mean(), "diff": m, "lo": lo, "hi": hi, "p": pv})
    df = pd.DataFrame(rows)
    df["p_holm"] = df.groupby("point").p.transform(holm)
    tables["comparisons"] = df
    md += ["## Research questions (A - B, each side minus the hull at its own cost; paired; Holm per point)\n",
           md_table(pd.DataFrame({
               "": df.question, "comparison": df.comparison, "point": df.point,
               "A: acc / tokens": [f"{pct(a)} / {t:.2f}" for a, t in zip(df.acc_A, df.tok_A)],
               "B: acc / tokens": [f"{pct(a)} / {t:.2f}" for a, t in zip(df.acc_B, df.tok_B)],
               "A - B (pts)": [with_ci(*v, signed=True) for v in zip(df["diff"], df.lo, df.hi)],
               "p_holm": df.p_holm.map(lambda v: f"{v:.3f}")})), ""]

    # ---- variants and sweeps
    rows = []
    for s, run in runs.items():
        g = st.ci(F.gain(pols[s]["max"]))
        row = {"allocator": s.label, "acc": pols[s]["max"].acc.mean(), "tokens": pols[s]["max"].tok.mean(),
               "gain_max": g[0], "gain_lo": g[1], "gain_hi": g[2], "oracle_gap_closed": F.gap_closed(pols[s]["max"])}
        for t in REPORT_TAUS:
            row[f"gain@{t:g}"], row[f"tokens@{t:g}"] = F.gain(pols[s][t]).mean(), pols[s][t].tok.mean()
        row["added_area"] = added_area(F.vertices, [(p.tok.mean(), p.acc.mean()) for p in pols[s].values()])
        row["spearman"] = spearman_rows(run["S"], targets(B, s) if s.b0 else B.acc[:, [B.col[r] for r in GRID]])
        row["alpha"] = ", ".join(f"{k:g}x{v}" for k, v in pd.Series([f["alpha"] for f in run["parts"]]).value_counts().items()) \
            if s.regressor == "ridge" else "-"
        row["sweep"] = s in sweeps
        rows.append(row)
    tables["variants"] = var = pd.DataFrame(rows)

    def var_md(d):
        return md_table(pd.DataFrame({
            "allocator": d.allocator, "acc (tokens), accuracy-tuned": [f"{pct(a)} ({t:.2f})" for a, t in zip(d.acc, d.tokens)],
            "vs hull, accuracy-tuned": [with_ci(*v, signed=True) for v in zip(d.gain_max, d.gain_lo, d.gain_hi)],
            **{f"vs hull @ {t:g} (tokens)": [f"{pct(g, True)} ({c:.2f})" for g, c in zip(d[f"gain@{t:g}"], d[f"tokens@{t:g}"])]
               for t in REPORT_TAUS},
            "added area": d.added_area.map(lambda v: pct(v, True)), "oracle gap closed": d.oracle_gap_closed.map(lambda v: pct(v, True)),
            "Spearman": d.spearman.map(lambda v: f"{v:.3f}"), "alpha (folds)": d.alpha}))
    md += ["## Allocator variants\n", "*Added area*: accuracy (pts) the allocator's operating points add to the cell "
           "frontier over costs 0.2-0.8. *Spearman*: per item, predicted vs actual gain over the cells.\n",
           var_md(var[~var.sweep]), "", "### Sweeps: the first-pass budget b0 (no KL passes) and the ridge strength\n",
           var_md(var[var.sweep]), ""]

    # ---- cheap-pass failures
    reps = list(dict.fromkeys([b0, *KL_PASSES, "i3:v64_a0.25", "i3:v256_a0.5"]))
    fail, rel = cheap_pass_failures(B, reps)
    tables["cheap_pass_failures"] = fail
    show = fail[fail.group.isin(["all", *AUDIO_TYPES, "cross-modal"])]
    md += ["## The cheap passes: when are they confidently wrong? (plan §6.4a)\n",
           "One pass in the original option order. *Confidently wrong*: top probability >= 0.8 and wrong. *Scaled*: "
           "temperature fitted on the other folds; the fitted temperatures are large (the passes are overconfident), "
           "so after scaling no answer reaches 0.8. By task type: cheap_pass_failures.csv.\n",
           md_table(pd.concat([show[["pass", "group", "items"]],
                               show[["accuracy", "mean max prob", "ECE", "ECE scaled", "confidently wrong",
                                     "confidently wrong, scaled"]].map(pct)], axis=1)), ""]
    tt = fail[(fail["pass"] == cell(b0)) & fail.group.str.startswith("task: ")].sort_values("confidently wrong", ascending=False)
    md += [f"**{cell(b0)} by task type**\n", md_table(pd.concat([tt[["group", "items"]],
                                                              tt[["accuracy", "confidently wrong"]].map(pct)], axis=1)), ""]

    # ---- time per cell
    tables["time_per_cell"] = tpc = time_per_cell(cfg).reset_index(names="representation")
    md += ["## Time per representation (L4, mean per item)\n", md_table(pd.concat([
        tpc[["representation"]], tpc[["accuracy"]].map(pct), tpc[["tokens"]].map(lambda v: f"{v:.2f}"),
        tpc[["forward_ms", "preprocess_ms", "read_media_ms", "end_to_end_ms"]].map(lambda v: f"{v:.0f}")], axis=1)), ""]

    # ---- audio selection
    rows = []
    for a, b in SELECTION:
        d = B.acc[:, B.col[a]] - B.acc[:, B.col[b]]
        m, lo, hi, pv = st.test(d)
        row = {"window": f"{WINDOW[a[-1]]} vs leading", "cell": cell(b), "diff": m, "lo": lo, "hi": hi, "p": pv}
        row.update({t: d[B.audio_type == t].mean() for t in AUDIO_TYPES})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p)
    tables["audio_selection"] = df
    md += ["## Which part of the audio to keep (RQ3.3)\n", md_table(pd.DataFrame({
        "window": df.window, "budget cell": df.cell,
        "difference (pts)": [with_ci(*v, signed=True) for v in zip(df["diff"], df.lo, df.hi)],
        **{t: df[t].map(lambda v: pct(v, True)) for t in AUDIO_TYPES}, "p_holm": df.p_holm.map(lambda v: f"{v:.3f}")})), ""]

    # ---- the midterm test set
    te = np.flatnonzero(np.isin(B.ids, B.idea3_test))
    mg, mp = table_policies(B, MIDTERM_BETAS, False, "midterm"), table_policies(B, MIDTERM_BETAS, True, "midterm")
    ma = allocator_policies(B, mid_runs[specs["Allocator"]], "Allocator")
    p = repo_path(MIDTERM)
    old = pd.DataFrame(json.loads(p.read_text())["policies"]) if p.exists() else pd.DataFrame()
    rows = []
    for beta in MIDTERM_BETAS:
        for pol_name, pol in (("global", mg[beta]), ("per audio type", mp[beta]), ("allocator", ma[beta])):
            o = old[(old.get("candidates") == "raw grid") & np.isclose(old.get("beta", np.nan), beta) & (old.get("policy") == pol_name)] \
                if len(old) else old
            rows.append({"budget": beta, "policy": pol_name, "midterm acc": pct(o.test_acc.iloc[0]) if len(o) else "-",
                         "midterm tokens": f"{o.test_cost.iloc[0]:.2f}" if len(o) else "-",
                         "this run": with_ci(*st.ci(pol.acc)), "tokens": f"{pol.tok.mean():.2f}"})
    tables["midterm_test"] = pd.DataFrame(rows)
    md += [f"## The midterm test set ({len(te)} items; trained on the other {B.n - len(te)})\n",
           "*midterm*: results/idea3/summary.json (dev 90 / test 90, Mac); *this run*: the L4 bank, trained on every "
           "other item. The allocator cannot go below its cheap passes.\n", md_table(tables["midterm_test"]), ""]

    # ---- every operating point, for the combined-system table in src/final/stats.py
    rows = [{"method": f"{cell(r)} (fixed)", "point": "fixed", "acc": B.acc[:, B.col[r]].mean(),
             "tokens": pay(B, [r])[0].mean(), "seconds": pay(B, [r])[1].mean()} for r in GRID]
    for name, pol in [(n, pols[s]) for n, s in specs.items()] + [(glob["max"].name, glob), (ptype["max"].name, ptype)]:
        rows += [{"method": name, "point": p, **dict(zip(("acc", "tokens", "seconds"), q.means))} for p, q in pol.items()]
    tables["curves"] = pd.DataFrame(rows)

    # ---- files and figures
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    figures(B, pols, specs, glob, ptype, rel, b0, out)
    md += ["## Files\n", "CSVs: " + ", ".join(f"`{n}.csv`" for n in tables) + ". Figures: `allocator_curves.pdf/png`, "
           "`reliability.pdf/png`.", ""]
    (out / "report.md").write_text("\n".join(md))
    head = tables["headline"]
    print(md_table(pd.DataFrame({"method": head.method, "accuracy": head.acc.map(pct), "tokens": head.tokens.round(2),
                                 "vs hull": head.gain.map(lambda v: "-" if np.isnan(v) else pct(v, True))})))
    print(f"\nwritten: {out}/report.md, CSVs and figures ({(time.time() - t_start) / 60:.1f} min)")


def figures(B, pols, specs, glob, ptype, rel, b0, out):
    import matplotlib.pyplot as plt

    from src.common import COL_W, INK_2, PAGE_W, PALETTE, save_fig, set_plot_style
    set_plot_style()
    idx = np.arange(B.n)
    tok, sec = pay(B, GRID)
    fixed = [(tok[:, j].mean(), sec[:, j].mean(), B.acc[:, B.col[r]].mean(), cell(r)) for j, r in enumerate(GRID)]
    series = [("Allocator", pols[specs["Allocator"]], PALETTE[0], "-"),
              ("Allocator, no first pass", pols[specs["Allocator, no first pass (question + meta)"]], PALETTE[4], "-"),
              ("Best single cell", glob, PALETTE[1], "--"), ("Best cell per audio type", ptype, PALETTE[2], "--")]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.6), sharey=True)
    for ax, k, xlabel in [(axes[0], 0, "Cost (input tokens / full input, incl. cheap passes)"),
                          (axes[1], 1, "Time per item on the L4 (ms)")]:
        scale = 1000 if k == 1 else 1
        hull = upper_hull([(f[k], f[2], f[3]) for f in fixed])
        ax.plot([v[0] * scale for v in hull], [100 * v[1] for v in hull], color=INK_2, ls=":", lw=1.0,
                label="Random mixing of fixed cells (hull)", zorder=1)
        for name, pol, color, ls in series:
            pts = sorted({(p.tok.mean() if k == 0 else p.sec.mean(), p.acc.mean()) for q, p in pol.items() if q != "max"})
            ax.plot([c * scale for c, _ in pts], [100 * a for _, a in pts], color=color, ls=ls, lw=1.4, label=name, zorder=3)
            m = pol["max"]
            ax.scatter((m.tok.mean() if k == 0 else m.sec.mean()) * scale, 100 * m.acc.mean(), s=24, color=color,
                       edgecolors="white", linewidths=1.0, zorder=4)
        ax.scatter([f[k] * scale for f in fixed], [100 * f[2] for f in fixed], s=14, color="#8a8984",
                   edgecolors="white", linewidths=0.8, zorder=2)
        for v in hull:
            ax.annotate(v[2], (v[0] * scale, 100 * v[1]), xytext=(4, -3), textcoords="offset points", fontsize=5.5,
                        color=INK_2, ha="left", va="top")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=2.2)
    fig.subplots_adjust(wspace=0.06)
    save_fig(fig, out, "allocator_curves")

    conf, conf_t, right = rel[b0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    edges = np.linspace(0.25, 1.0, 7)
    for c, color, lab in ((conf, PALETTE[1], "raw"), (conf_t, PALETTE[0], "temperature-scaled")):
        b = np.clip(np.digitize(c, edges) - 1, 0, len(edges) - 2)
        xs = [c[b == i].mean() for i in range(len(edges) - 1) if (b == i).sum() >= 10]
        ys = [right[b == i].mean() for i in range(len(edges) - 1) if (b == i).sum() >= 10]
        ax.plot(np.array(xs) * 100, np.array(ys) * 100, marker="o", ms=3.5, color=color, lw=1.4, label=lab)
    ax.plot([25, 100], [25, 100], color=INK_2, ls=":", lw=0.8)
    ax.set_xlabel(f"Confidence of the {cell(b0)} pass (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper left")
    save_fig(fig, out, "reliability")


if __name__ == "__main__":
    from src.final.idea3_allocator import main as _main
    _main()
