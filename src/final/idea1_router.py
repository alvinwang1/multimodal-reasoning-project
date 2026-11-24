"""Idea 1 (final report): learned routing between input representations, evaluated on the shared score bank.

For each item, a router picks one representation from a menu (raw image and audio, the human caption or
transcript, Idea 3 budget cells, ...) and the model answers from it. The aim is high accuracy at low cost.
Everything comes from the score bank (src/final/bank.py): no model is run here, so the study takes minutes on a CPU.

Routers: linear in the features phi(x), with one weight vector per representation r.
  pointwise  s(x,r) = P(r answers x correctly): a logistic regression on the soft label (correctness averaged
             over the 3 option orders). This is the midterm router.
  pairwise   s(x,r) only ranks representations. For every pair (r, r') where r has the higher log p(gold) on
             item i, the loss is  w * log(1 + exp(s(x,r') - s(x,r))),  with w = the log p(gold) gap, capped.
Policy: answer with argmax_r s(x,r) - lambda * C_r(x), where C_r(x) is the cost actually paid. The cheap passes
that produce the features are always paid, and answering with one of them costs nothing extra.
lambda is chosen inside the training folds only, on cross-fitted scores: either to reach a target mean cost tau,
or to maximize accuracy. The L2 strength is chosen the same way (validation loss).

Features (all fitted on the training folds only):
  question   SBERT embedding of the question, PCA to 16 dims
  meta       log audio length, log question+options tokens, log image tokens; audio type (one-hot; ablated,
             since at deployment it needs an audio classifier)
  probe      the option distribution of the cheap first pass (64 image tokens, first half of the audio):
             max prob, margin, entropy, after temperature scaling
  disagree   two more cheap passes, image only (64 tokens) and audio only (first half): the same three numbers
             for each, and for every pair of passes KL both ways and whether their top answers agree
  midterm    the midterm probe, the caption+transcript pass (human-written text, so not deployable as is)

Menus: base7 (the 7 midterm formats), base7+cells (+ Idea 3 cells (256,1), (64,1), (64,1/2)), raw (no
human-written text: raw image/audio at several budgets, image only, audio only, question only).

Evaluation: the 5 frozen folds (data/final_folds.json); every item gets one out-of-fold decision. Cost is input
tokens relative to the full input (as in the midterm), and end-to-end seconds on the L4: reading the media
once, every pass, the question embedding and the router. Baselines: every fixed representation, the uniform random
router, random mixing of fixed representations at matched cost (the hull), confidence cascades, the per-item oracle.

    python -m src.final.idea1_router                 # everything (a few minutes)
    python -m src.final.idea1_router --quick         # headline methods only
    python -m src.final.idea1_router --no-question   # without question embeddings (no SBERT needed)
    python -m src.final.idea1_router --embed-only    # compute the question embeddings only
    On Modal (CPU): modal run scripts/modal_run.py --sections router

The question embeddings (results/final/question_emb.npz) need sentence-transformers and the OmniBench download; they
are computed on first use. Outputs: results/final/idea1/ (report.md with every table, CSVs, figures).
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_softmax, softmax
from scipy.stats import rankdata

from src.common import CONFIG_ORDER, load_config, read_jsonl, repo_path
from src.data import AUDIO_TYPES
from src.final.bank import fold_of, load_bank, load_groups, matrix
from src.final.stats import Resampler, holm, md_table, pct, with_ci

OUT_DIR = "results/final/idea1"
EMB_CACHE = "results/final/question_emb.npz"

BASE7 = list(CONFIG_ORDER)
CELLS = ["i3:v256_a1", "i3:v64_a1", "i3:v64_a0.5"]
RAW = ["full", "i3:v768_a0.5", "i3:v256_a1", "i3:v256_a0.5", "i3:v64_a1", "i3:v64_a0.5", "img_only", "aud_only",
       "q_only"]
MENUS = {"base7": BASE7, "base7+cells": BASE7 + CELLS, "raw": RAW}
PASSES = {"probe": ("i3:v64_a0.5",), "disagree": ("i3:v64_a0", "i3:v0_a0.5"), "midterm": ("cap_trans",)}
MEDIA_FREE = {"q_only", "cap_trans"}  # passes that read no image or audio file
SHORT = {"full": "I+A", "img_trans": "I+T", "cap_aud": "C+A", "cap_trans": "C+T", "img_only": "I", "aud_only": "A",
         "q_only": "Q"}
META = ["log audio seconds", "log question+options tokens", "log image tokens"]

TAUS = tuple(np.round(np.arange(0.15, 1.2001, 0.05), 3))  # target mean costs (tokens) for the curves
REPORT_TAUS = (0.35, 0.5, 0.7)
L2_GRID = {"pointwise": (1e-3, 1e-2, 1e-1, 1.0), "pairwise": (1e-1, 1.0, 10.0)}  # chosen inside each training fold
PCA_DIMS = 16
PAIR_CAP = 4.0  # cap on the log p(gold) gap used as a pair weight, so a few confident items do not dominate
LAMBDA_STEPS = 240


def short(rep):
    return SHORT.get(rep, rep.replace("i3:", ""))


# ----------------------------------------------------------------------------- experiment specification
@dataclass(frozen=True)
class Spec:
    model: str = "pairwise"                 # pointwise | pairwise
    menu: str = "base7+cells"               # a key of MENUS
    probes: tuple = ("probe", "disagree")   # cheap-pass feature groups (keys of PASSES)
    question: bool = True
    audio_type: bool = True
    temperature: bool = True
    l2: object = "auto"                     # "auto" (chosen inside the training folds) or a number
    pair_weight: str = "logp"               # pairwise only: logp (capped gap) | acc (accuracy gap) | uniform
    train_size: object = None               # learning curve: training items per fold (None = all)
    repeat: int = 0                         # learning curve: which random subsample

    @property
    def passes(self):
        return tuple(dict.fromkeys(p for g in self.probes for p in PASSES[g]))

    @property
    def reps(self):
        """Menu plus the cheap passes: answering with a pass that was run anyway is always possible."""
        menu = MENUS[self.menu]
        return menu + [p for p in self.passes if p not in menu]

    @property
    def features(self):
        f = (["question"] if self.question else []) + ["meta"] + (["audio type"] if self.audio_type else [])
        return "+".join(f + list(self.probes))

    @property
    def label(self):
        extra = [] if self.temperature or not self.passes else ["no temperature"]
        extra += [] if self.l2 == "auto" else [f"l2={self.l2:g}"]
        extra += [] if self.model == "pointwise" or self.pair_weight == "logp" else [f"weights={self.pair_weight}"]
        extra += [] if self.train_size is None else [f"n={self.train_size}", f"rep {self.repeat}"]
        return f"{self.model} · {self.menu} · {self.features}" + (f" · {', '.join(extra)}" if extra else "")


# ----------------------------------------------------------------------------- data
class Bank:
    """Everything the router study needs, as arrays with one row per item."""

    def __init__(self, cfg, question=True, emb_path=EMB_CACHE):
        bank = load_bank(cfg, sources=("base", "idea3"))
        if bank.empty:
            raise SystemExit("the score bank is empty: unpack final_results.tar.gz in the repo root first")
        acc = matrix(bank, "acc")  # items with every representation scored
        self.reps, self.ids = list(acc.columns), acc.index.to_numpy()
        self.n, self.col = len(self.ids), {r: j for j, r in enumerate(acc.columns)}

        def m(value):
            return matrix(bank, value, complete=False).reindex(index=self.ids, columns=self.reps).to_numpy(float)
        self.acc, self.logp = acc.to_numpy(float), m("logp")
        self.tok = m("cost")          # input tokens / the item's full-input tokens
        self.model_s = m("latency_s")  # forward pass + preprocessing on the L4
        self.decode_s = np.nanmean(m("decode_s"), axis=1)  # reading the media files: once per item
        for name in ("logp", "tok", "model_s"):
            if np.isnan(getattr(self, name)).any():
                raise SystemExit(f"score bank: missing {name} values")
        first = bank.drop_duplicates("id").set_index("id").reindex(self.ids)
        self.audio_type = first.audio_type.to_numpy()
        fold = fold_of(cfg)
        self.fold = np.array([fold[i] for i in self.ids])
        self.midterm_test = np.isin(self.ids, load_groups(cfg)["midterm_test"])
        a = {r: self.acc[:, self.col[r]] for r in ("full", "img_only", "aud_only")}
        # plan §6.3b: answered by the full input (>= 2/3 of orders) while image only and audio only both fail
        self.cross_modal = (a["full"] > 0.6) & (a["img_only"] < 0.4) & (a["aud_only"] < 0.4)
        self._load_passes(cfg)
        self.emb, self.embed_s = self._load_embeddings(emb_path) if question else (None, 0.0)

    def _load_passes(self, cfg):
        """Original-order (perm 0) letter logits of the cheap passes, the gold answer and the meta features."""
        need = {p for group in PASSES.values() for p in group}
        pos = {i: k for k, i in enumerate(self.ids)}
        self.logits = {p: np.full((self.n, 4), np.nan) for p in need}
        self.gold = np.full(self.n, -1)
        meta = np.full((self.n, 3), np.nan)
        sources = [(cfg["final"]["base_out"], "config", ""),
                   (f"{cfg['idea3']['out_dir']}/raw_final.jsonl", "cell", "i3:")]
        for path, key, prefix in sources:
            for r in read_jsonl(repo_path(path)):
                k = pos.get(r["id"])
                if r["perm"] != 0 or k is None:
                    continue
                if r["perm_order"] != [0, 1, 2, 3]:
                    raise SystemExit(f"order 0 of item {r['id']} is not the original option order")
                rep = prefix + r[key]
                if rep in need:
                    self.logits[rep][k] = r["letter_logits"]
                if rep == "full":
                    self.gold[k] = r["answer_idx"]
                    meta[k, 0], meta[k, 2] = r["audio_sec"], r["n_image_tok"]
                elif rep == "q_only":
                    meta[k, 1] = r["n_text_tok"]
        missing = [p for p, z in self.logits.items() if np.isnan(z).any()]
        if missing or (self.gold < 0).any() or np.isnan(meta).any():
            raise SystemExit(f"score bank: original-order passes missing ({missing or 'meta'})")
        self.meta = np.log(np.maximum(meta, 1e-3))

    def _load_embeddings(self, path):
        p = repo_path(path)
        if not p.exists():
            raise SystemExit(f"{path} not found. Compute the question embeddings first: python -m src.final.idea1_router "
                             "--embed-only (needs sentence-transformers and the OmniBench download; on Modal: modal run "
                             "scripts/modal_run.py --sections router), or run without them: --no-question")
        z = np.load(p)
        emb = dict(zip(z["ids"].tolist(), z["emb"]))
        missing = [int(i) for i in self.ids if int(i) not in emb]
        if missing:
            raise SystemExit(f"{path} has no embedding for {len(missing)} items (e.g. {missing[0]}); recompute it with "
                             "--embed-only")
        return np.stack([emb[int(i)] for i in self.ids]).astype(float), float(z["s_per_item"])


def compute_embeddings(cfg, path=EMB_CACHE):
    """SBERT embeddings of every question (the model of Analysis C), and the CPU time per question at batch size 1."""
    from sentence_transformers import SentenceTransformer

    from src.data import load_omnibench, text_records

    records, _ = text_records(load_omnibench(cfg))
    question = {r["index"]: r["question"] for r in records}
    ids = sorted(i for f in json.loads(repo_path(cfg["final"]["folds"]).read_text())["folds"] for i in f)
    model = SentenceTransformer(cfg["analysis_c"]["sbert_model"], device="cpu")
    model.encode([question[ids[0]]])  # warm-up
    sample = ids[:: max(1, len(ids) // 100)]
    t0 = time.perf_counter()
    for i in sample:  # deployment embeds one question at a time
        model.encode([question[i]], normalize_embeddings=True)
    s_per_item = (time.perf_counter() - t0) / len(sample)
    emb = model.encode([question[i] for i in ids], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    out = repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids), emb=emb.astype(np.float32), s_per_item=s_per_item,
             model=cfg["analysis_c"]["sbert_model"])
    print(f"question embeddings: {len(ids)} x {emb.shape[1]} -> {out} ({1000 * s_per_item:.1f} ms per question on CPU)")


# ----------------------------------------------------------------------------- features
def fit_temperature(z, gold):
    """Temperature T minimizing the NLL of the gold answer under softmax(z / T)."""
    rows = np.arange(len(z))
    res = minimize_scalar(lambda lt: -log_softmax(z / np.exp(lt), axis=1)[rows, gold].mean(),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def dist_features(q):
    s = np.sort(q, axis=1)[:, ::-1]
    return np.c_[s[:, 0], s[:, 0] - s[:, 1], -(q * np.log(np.clip(q, 1e-12, 1))).sum(axis=1)]


def kl(p, q):
    return (p * (np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(q, 1e-12, 1)))).sum(axis=1)


def build_features(B, spec, tr, te, seed):
    """Standardized features for the items tr (fitting rows) and te, with every fitted step fitted on tr only."""
    cols, names, temps = [], [], {}
    if spec.question:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=PCA_DIMS, random_state=seed).fit(B.emb[tr])
        cols.append(pca.transform(B.emb))
        names += [f"question pc{j + 1}" for j in range(PCA_DIMS)]
    cols.append(B.meta)
    names += META
    if spec.audio_type:
        cols.append(np.stack([B.audio_type == t for t in AUDIO_TYPES], axis=1).astype(float))
        names += [f"audio type: {t}" for t in AUDIO_TYPES]
    dist = {}
    for p in spec.passes:
        z = B.logits[p]
        temps[p] = fit_temperature(z[tr], B.gold[tr]) if spec.temperature else 1.0
        dist[p] = softmax(z / temps[p], axis=1)
        cols.append(dist_features(dist[p]))
        names += [f"{short(p)} {x}" for x in ("max prob", "margin", "entropy")]
    for p, q in combinations(spec.passes, 2):
        cols.append(np.c_[kl(dist[p], dist[q]), kl(dist[q], dist[p]),
                          dist[p].argmax(axis=1) == dist[q].argmax(axis=1)])
        names += [f"KL({short(p)}||{short(q)})", f"KL({short(q)}||{short(p)})", f"{short(p)}, {short(q)} agree"]
    X = np.hstack(cols).astype(float)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    return X[tr], X[te], names, temps


# ----------------------------------------------------------------------------- routers
class LinearRouter:
    """Scores S = X W + b (one column per representation), fitted with L-BFGS; the bias is not penalized."""

    def __init__(self, l2, spec):
        self.l2, self.spec = l2, spec

    def _fit(self, X, K, loss_grad, init=None):
        n, d = X.shape
        Xb = np.c_[X, np.ones(n)]
        pen = np.r_[np.ones(d), 0.0][:, None]

        def objective(w):
            W = w.reshape(d + 1, K)
            loss, G = loss_grad(Xb @ W)
            return loss + 0.5 * self.l2 * np.sum(pen * W ** 2), (Xb.T @ G + self.l2 * pen * W).ravel()

        w0 = np.zeros((d + 1) * K) if init is None else init.W.ravel()
        res = minimize(objective, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-6})
        self.W, self.converged = res.x.reshape(d + 1, K), bool(res.success)
        return self

    def raw(self, X):
        return X @ self.W[:-1] + self.W[-1]


class Pointwise(LinearRouter):
    @staticmethod
    def loss_grad(acc, logp):
        n = len(acc)

        def f(S):  # binary cross-entropy with soft labels, summed over representations, mean over items
            return (np.logaddexp(0, S) - acc * S).sum() / n, (expit(S) - acc) / n
        return f

    def fit(self, X, acc, logp, init=None):
        return self._fit(X, acc.shape[1], self.loss_grad(acc, logp), init)

    def score(self, X):
        return expit(self.raw(X))

    def val_loss(self, X, acc, logp):
        return self.loss_grad(acc, logp)(self.raw(X))[0]


def pairs(acc, logp, scheme, cap=PAIR_CAP):
    """(item, winner, loser, weight) for every pair of representations that differ on the item."""
    key = acc if scheme == "acc" else logp
    out = [[], [], [], []]
    for a, b in combinations(range(key.shape[1]), 2):
        d = key[:, a] - key[:, b]
        idx = np.flatnonzero(np.abs(d) > 1e-9)
        d = d[idx]
        out[0].append(idx)
        out[1].append(np.where(d > 0, a, b))
        out[2].append(np.where(d > 0, b, a))
        out[3].append(np.ones(len(idx)) if scheme == "uniform" else np.minimum(np.abs(d), cap))
    return [np.concatenate(x) for x in out]


class Pairwise(LinearRouter):
    def loss_grad(self, acc, logp):
        from scipy.sparse import csr_matrix

        n, K = acc.shape
        i, win, lose, w = pairs(acc, logp, self.spec.pair_weight)
        m = len(i)  # D maps the flattened scores to s_win - s_lose for every pair
        D = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)],
                                                          np.r_[i * K + win, i * K + lose])), shape=(m, n * K))
        Dt = D.T.tocsr()

        def f(S):  # sum over the item's pairs of w * log(1 + exp(-(s_win - s_lose))), mean over items
            d = D @ S.ravel()
            return (w * np.logaddexp(0, -d)).sum() / n, (Dt @ (-w * expit(-d) / n)).reshape(n, K)
        return f

    def fit(self, X, acc, logp, init=None):
        return self._fit(X, acc.shape[1], self.loss_grad(acc, logp), init)

    def score(self, X):
        return self.raw(X)

    def val_loss(self, X, acc, logp):
        return self.loss_grad(acc, logp)(self.raw(X))[0]


MODELS = {"pointwise": Pointwise, "pairwise": Pairwise}


# ----------------------------------------------------------------------------- costs and policies
def pay(B, reps, passes=(), overhead_s=0.0):
    """Cost actually paid per item for answering with each rep after running `passes` -> (tokens, seconds), n x K.
    Seconds: the media files are read once if any pass needs them, plus every forward pass and the router overhead."""
    J, P = [B.col[r] for r in reps], [B.col[p] for p in passes]
    in_passes = np.isin(reps, list(passes))
    tok = np.where(in_passes, 0.0, B.tok[:, J]) + B.tok[:, P].sum(axis=1, keepdims=True)
    sec = np.where(in_passes, 0.0, B.model_s[:, J]) + B.model_s[:, P].sum(axis=1, keepdims=True)
    media = np.array([r not in MEDIA_FREE for r in reps]) | any(p not in MEDIA_FREE for p in passes)
    return tok, sec + B.decode_s[:, None] * media + overhead_s


def choose(S, E, lam):
    """argmax_r S - lam * E per item (lam: scalar or one value per item); ties go to the cheaper rep."""
    lam = np.asarray(lam, float).reshape(-1, 1) if np.ndim(lam) else lam
    return np.argmax(S - (lam + 1e-9) * E, axis=1)


def lambda_grid(S):
    spread = max(float(np.ptp(S, axis=1).max()), 1e-6)
    return np.r_[0.0, np.geomspace(1e-4, 1e2, LAMBDA_STEPS) * spread]  # at the top, the cheapest rep always wins


def select_lambdas(S, E, A, taus):
    """On cross-fitted training scores: the lambda with the highest accuracy (ties: larger lambda, i.e. cheaper),
    and for each target mean cost tau the lambda whose mean cost is closest (ties: smaller lambda)."""
    grid, rows = lambda_grid(S), np.arange(len(S))
    cost, acc = np.empty(len(grid)), np.empty(len(grid))
    for j, lam in enumerate(grid):
        c = choose(S, E, lam)
        cost[j], acc[j] = E[rows, c].mean(), A[rows, c].mean()
    best = np.lexsort((-grid, -np.round(acc, 12)))[0]
    return grid[best], {t: grid[np.argmin(np.abs(cost - t))] for t in taus}


def splits(B, kind):
    if kind == "folds":
        return [(np.flatnonzero(B.fold != k), np.flatnonzero(B.fold == k)) for k in np.unique(B.fold)]
    return [(np.flatnonzero(~B.midterm_test), np.flatnonzero(B.midterm_test))]  # "midterm": the 120 test items


def fit_fold(B, spec, tr, te, taus, seed):
    """Train on tr and score te. The L2 strength and the lambdas are chosen on scores cross-fitted inside tr
    (inner folds = the frozen folds of the training items), so te is never used for any choice."""
    reps, Model = spec.reps, MODELS[spec.model]
    J = [B.col[r] for r in reps]
    A, L = B.acc[:, J], B.logp[:, J]
    if spec.train_size is not None:
        rng = np.random.default_rng([seed, int(B.fold[te][0]), spec.repeat, spec.train_size])
        tr = np.sort(rng.choice(tr, size=spec.train_size, replace=False))
    l2s = L2_GRID[spec.model] if spec.l2 == "auto" else (float(spec.l2),)
    inner = B.fold[tr]
    oof = {l2: np.empty((len(tr), len(reps))) for l2 in l2s}
    vloss = dict.fromkeys(l2s, 0.0)
    for f in np.unique(inner):
        fit_pos, val_pos = np.flatnonzero(inner != f), np.flatnonzero(inner == f)
        Xa, Xv, _, _ = build_features(B, spec, tr[fit_pos], tr[val_pos], seed)
        m = None
        for l2 in sorted(l2s, reverse=True):  # strongest penalty first; each fit starts from the previous one
            m = Model(l2, spec).fit(Xa, A[tr[fit_pos]], L[tr[fit_pos]], init=m)
            oof[l2][val_pos] = m.score(Xv)
            vloss[l2] += m.val_loss(Xv, A[tr[val_pos]], L[tr[val_pos]]) * len(val_pos)
    l2 = min(l2s, key=lambda v: (vloss[v], -v))  # ties: the stronger penalty
    Xtr, Xte, names, temps = build_features(B, spec, tr, te, seed)
    t0 = time.perf_counter()
    model = Model(l2, spec).fit(Xtr, A[tr], L[tr])
    fit_s = time.perf_counter() - t0
    tok, _ = pay(B, reps, spec.passes)
    lam_max, lam_tau = select_lambdas(oof[l2], tok[tr], A[tr], taus)
    t0 = time.perf_counter()
    S = model.score(Xte)
    choose(S, tok[te], lam_max)
    infer_s = (time.perf_counter() - t0) / len(te)
    return {"te": te, "S": S, "lam_max": lam_max, "lam_tau": lam_tau, "l2": l2, "temps": temps, "names": names,
            "W": model.W, "converged": model.converged, "fit_s": fit_s, "infer_s": infer_s, "n_train": len(tr)}


def run_spec(B, spec, taus, seed, split="folds"):
    """Out-of-fold scores and per-item lambdas for one router (each item scored by the model that did not see it)."""
    folds = [fit_fold(B, spec, tr, te, taus, seed) for tr, te in splits(B, split)]
    K = len(spec.reps)
    S = np.full((B.n, K), np.nan)
    lam = {p: np.full(B.n, np.nan) for p in ("max",) + tuple(taus)}
    for f in folds:
        S[f["te"]] = f["S"]
        lam["max"][f["te"]] = f["lam_max"]
        for t in taus:
            lam[t][f["te"]] = f["lam_tau"][t]
    return {"spec": spec, "split": split, "S": S, "lam": lam, "folds": folds,
            "infer_s": float(np.mean([f["infer_s"] for f in folds]))}


# ----------------------------------------------------------------------------- policies and baselines
@dataclass
class Policy:
    """Per-item outcome of one routing policy on the items idx: accuracy (mean over orders), tokens, seconds."""
    name: str
    point: object        # "max" (lambda/threshold chosen for accuracy), a target cost tau, or "fixed"
    idx: np.ndarray
    acc: np.ndarray
    tok: np.ndarray
    sec: np.ndarray
    choice: object = None  # the representation used per item (None for the expected random router)

    @property
    def means(self):
        return float(self.acc.mean()), float(self.tok.mean()), float(self.sec.mean())


def router_policies(B, run, name):
    spec, S = run["spec"], run["S"]
    idx = np.flatnonzero(~np.isnan(S[:, 0]))
    overhead = run["infer_s"] + (B.embed_s if spec.question else 0.0)
    tok, sec = pay(B, spec.reps, spec.passes, overhead)
    J, reps = np.array([B.col[r] for r in spec.reps]), np.array(spec.reps)
    out = {}
    for point, lam in run["lam"].items():
        c = choose(S[idx], tok[idx], lam[idx])
        rows = np.arange(len(idx))
        out[point] = Policy(name, point, idx, B.acc[idx, J[c]], tok[idx][rows, c], sec[idx][rows, c], reps[c])
    return out


def fixed_policy(B, rep, idx):
    tok, sec = pay(B, [rep])
    return Policy(short(rep), "fixed", idx, B.acc[idx, B.col[rep]], tok[idx, 0], sec[idx, 0], np.full(len(idx), rep))


def uniform_policy(B, reps, idx, name):
    """The uniform random router, in expectation: each item gets the mean over the menu."""
    tok, sec = pay(B, reps)
    J = [B.col[r] for r in reps]
    return Policy(name, "random", idx, B.acc[idx][:, J].mean(axis=1), tok[idx].mean(axis=1), sec[idx].mean(axis=1))


def cascade_policies(B, probe, target, split, taus, name):
    """Answer with the probe pass unless its top probability is below theta; then also run the target.
    theta is chosen on the training items of each split (for accuracy, or to match a target mean cost)."""
    conf = softmax(B.logits[probe], axis=1).max(axis=1)
    jp, jt = B.col[probe], B.col[target]
    thetas = np.r_[0.0, np.quantile(conf, np.linspace(0, 1, 201)), 1.01]

    def apply(theta, items):
        esc = conf[items] < theta
        media = (probe not in MEDIA_FREE) | (esc & (target not in MEDIA_FREE))
        return (np.where(esc, B.acc[items, jt], B.acc[items, jp]), B.tok[items, jp] + esc * B.tok[items, jt],
                B.model_s[items, jp] + esc * B.model_s[items, jt] + B.decode_s[items] * media,
                np.where(esc, target, probe))

    points = ("max",) + tuple(taus)
    res = {p: [np.full(B.n, np.nan) for _ in range(3)] + [np.full(B.n, None, dtype=object)] for p in points}
    for tr, te in splits(B, split):
        acc_tr = np.array([apply(t, tr)[0].mean() for t in thetas])
        tok_tr = np.array([apply(t, tr)[1].mean() for t in thetas])
        chosen = {"max": thetas[np.lexsort((thetas, -np.round(acc_tr, 12)))[0]]}
        chosen.update({t: thetas[np.argmin(np.abs(tok_tr - t))] for t in taus})
        for p, theta in chosen.items():
            for arr, v in zip(res[p], apply(theta, te)):
                arr[te] = v
    idx = np.flatnonzero(~np.isnan(res["max"][0]))
    return {p: Policy(name, p, idx, *(a[idx] for a in res[p][:3]), res[p][3][idx]) for p in points}


def oracle_policies(B, reps, idx, taus, name="Oracle"):
    """Per-item best representation (it sees the answers): an upper bound, not a method."""
    tok, sec = pay(B, reps)
    J = np.array([B.col[r] for r in reps])
    A, T = B.acc[idx][:, J], tok[idx]
    rows = np.arange(len(idx))
    grid = np.r_[0.0, np.geomspace(1e-4, 10, 400)]
    costs = np.array([T[rows, choose(A, T, lam)].mean() for lam in grid])
    out = {}
    for p, lam in [("max", 0.0)] + [(t, grid[np.argmin(np.abs(costs - t))]) for t in taus]:
        c = choose(A, T, lam)
        out[p] = Policy(name, p, idx, A[rows, c], T[rows, c], sec[idx][rows, c], np.array(reps)[c])
    curve = [(T[rows, c].mean(), A[rows, c].mean()) for c in (choose(A, T, lam) for lam in grid)]
    return out, curve


def upper_hull(points):
    """Increasing upper-left concave hull of (cost, acc, key) points: what random mixing of them can reach."""
    out = []
    for p in sorted(points, key=lambda q: (q[0], -q[1])):
        while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                 - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) >= 0:
            out.pop()
        out.append(p)
    mono, best = [], -np.inf
    for p in out:
        if p[1] > best:
            mono.append(p)
            best = p[1]
    return mono


def hull_value(vertices, c):
    xs, ys = [v[0] for v in vertices], [v[1] for v in vertices]
    return float(np.interp(c, xs, ys))  # flat beyond the ends


class Frontier:
    """Random mixing of fixed representations (the hull) on the items idx, in token or second costs.
    items(c) is the per-item accuracy of the mix of the two hull neighbours with mean cost c (for paired CIs)."""

    def __init__(self, B, reps, idx, kind="tok"):
        tok, sec = pay(B, reps)
        cost = (tok if kind == "tok" else sec)[idx].mean(axis=0)
        self.cols, self.idx, self.kind = B.acc[idx][:, [B.col[r] for r in reps]], idx, kind
        self.vertices = upper_hull([(cost[j], self.cols[:, j].mean(), j) for j in range(len(reps))])
        self.reps = [reps[v[2]] for v in self.vertices]
        self._B, self._all, self._oracle = B, list(reps), None

    def items(self, c):
        V = self.vertices
        if c <= V[0][0] or len(V) == 1:
            return self.cols[:, V[0][2]]
        if c >= V[-1][0]:
            return self.cols[:, V[-1][2]]
        j = max(k for k in range(len(V) - 1) if V[k][0] <= c)
        a = (V[j + 1][0] - c) / (V[j + 1][0] - V[j][0])
        return a * self.cols[:, V[j][2]] + (1 - a) * self.cols[:, V[j + 1][2]]

    def gap_closed(self, pol):
        """Share of the gap between the hull and the oracle (on the same options) that the policy closes,
        at its mean token cost."""
        if self._oracle is None:
            _, curve = oracle_policies(self._B, self._all, self.idx, ())
            self._oracle = upper_hull([(c, a, None) for c, a in curve])
        c = pol.tok.mean()
        hull, best = hull_value(self.vertices, c), hull_value(self._oracle, c)
        return (pol.acc.mean() - hull) / (best - hull) if best > hull + 1e-9 else np.nan

    def gain(self, pol):
        """Per-item accuracy of the policy minus the hull mix at the policy's mean cost."""
        assert np.array_equal(pol.idx, self.idx), "policy and frontier on different items"
        return pol.acc - self.items(pol.tok.mean() if self.kind == "tok" else pol.sec.mean())


def added_area(fixed_vertices, points, lo=0.2, hi=0.8):
    """Mean accuracy the router's operating points add to the fixed-representation frontier over costs [lo, hi]
    (0 = the router never beats random mixing of fixed representations)."""
    grid = np.linspace(lo, hi, 121)
    base = np.array([hull_value(fixed_vertices, c) for c in grid])
    both = upper_hull(list(fixed_vertices) + [(c, a, None) for c, a in points])
    return float(np.mean([hull_value(both, c) for c in grid] - base))


def spearman_rows(S, Y):
    """Mean over items of the Spearman correlation between router scores and log p(gold) across the menu."""
    rs, ry = rankdata(S, axis=1), rankdata(Y, axis=1)
    rs, ry = rs - rs.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rs ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    ok = den > 0
    return float(((rs * ry).sum(axis=1)[ok] / den[ok]).mean())


def macro(pol, B):
    return float(np.mean([pol.acc[B.audio_type[pol.idx] == t].mean() for t in AUDIO_TYPES
                          if (B.audio_type[pol.idx] == t).any()]))


# ----------------------------------------------------------------------------- experiments
LEARN_SIZES, LEARN_REPEATS = (180, 360, 540, 720), 3


def experiments(quick, question):
    """Named headline routers and the groups of router specs to run (deduplicated later)."""
    def q(s):
        return s if question else replace(s, question=False)
    main = q(Spec())
    pointwise_main = replace(main, model="pointwise")
    free = replace(main, probes=())  # no cheap passes: question + meta features only
    headline = {"Midterm router": q(Spec(model="pointwise", menu="base7", probes=("midterm",))),
                "New router": main,
                "New router, no cheap passes": free,
                "New router, no human text": replace(main, menu="raw")}
    groups = {"headline": list(headline.values()),
              "grid": [q(Spec(model=m, menu=menu, probes=p)) for menu in MENUS for m in MODELS
                       for p in ((), ("probe",), ("probe", "disagree"))]
              + [q(Spec(model=m, menu="base7", probes=("midterm",))) for m in MODELS]}
    if not quick:
        kws = [{"temperature": False}, {"audio_type": False}] + ([{"question": False}] if question else [])
        groups["ablations"] = ([replace(s, **kw) for s in (main, pointwise_main) for kw in kws]
                               + [replace(s, **kw) for s in (free, replace(free, model="pointwise")) for kw in kws[1:]])
        groups["sweeps"] = ([replace(main, l2=v) for v in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)]
                            + [replace(pointwise_main, l2=v) for v in (1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0)]
                            + [replace(main, pair_weight=w) for w in ("acc", "uniform")])
        groups["learning"] = [replace(s, train_size=n, repeat=r)
                              for s in (main, pointwise_main, free, replace(free, model="pointwise"))
                              for n in LEARN_SIZES for r in range(LEARN_REPEATS)]
    return headline, groups


def comparisons(headline, question):
    """(family, A, B, what A - B measures): paired, cost-adjusted comparisons for the research questions."""
    main, mid, raw = headline["New router"], headline["Midterm router"], headline["New router, no human text"]
    rows = [("RQ1.1", main, replace(main, model="pointwise"), "pairwise vs pointwise training"),
            ("RQ1.2", main, replace(main, probes=("probe",)), "adding the disagreement passes"),
            ("RQ1.2", replace(main, probes=("probe",)), replace(main, probes=()), "the cheap first pass vs no pass"),
            ("RQ1.3", main, replace(main, menu="base7"), "adding Idea 3 budget cells to the menu"),
            ("ablation", main, replace(main, temperature=False), "temperature scaling of the passes"),
            ("ablation", main, replace(main, audio_type=False), "the audio-type feature"),
            ("midterm", main, mid, "new router vs the midterm router"),
            ("midterm", raw, main, "no human-written text in the menu")]
    if question:
        rows.insert(6, ("ablation", main, replace(main, question=False), "the question embedding"))
    return rows


# ----------------------------------------------------------------------------- report helpers
class Stats:
    """Bootstrap CIs and sign-flip tests over items, with resamples shared per item-set size."""

    def __init__(self, n_boot, seed):
        self.n_boot, self.seed, self.cache = n_boot, seed, {}

    def r(self, n):
        if n not in self.cache:
            self.cache[n] = Resampler(n, self.n_boot, [self.seed, n])
        return self.cache[n]

    def ci(self, v):
        lo, hi = self.r(len(v)).ci(v)
        return float(np.mean(v)), float(lo), float(hi)

    def test(self, d):
        return (*self.ci(d), self.r(len(d)).p_signflip(d))


def point_name(p):
    return "accuracy-tuned" if p == "max" else f"cost {p:g}"


CASCADES = [("Cascade C+T -> I+T (midterm)", "cap_trans", "img_trans"),
            ("Cascade v64_a0.5 -> I+A", "i3:v64_a0.5", "full"),
            ("Cascade v64_a0.5 -> v256_a1", "i3:v64_a0.5", "i3:v256_a1")]
FIXED_SHOWN = ["full", "img_trans", "i3:v256_a1", "cap_trans", "i3:v64_a0.5", "q_only"]


# ----------------------------------------------------------------------------- the study
def run_specs(B, specs, taus, seed, n_jobs, split="folds"):
    from joblib import Parallel, delayed
    specs = list(dict.fromkeys(specs))
    t0 = time.time()
    try:
        runs = Parallel(n_jobs=n_jobs)(delayed(run_spec)(B, s, taus, seed, split) for s in specs)
    except (OSError, ImportError) as e:  # no worker processes on this machine: run them one at a time
        print(f"  parallel workers unavailable ({e}); running sequentially", flush=True)
        runs = [run_spec(B, s, taus, seed, split) for s in specs]
    print(f"  {len(specs)} routers on the {split} split in {time.time() - t0:.0f} s", flush=True)
    return dict(zip(specs, runs))


def latency_parts(B, pol, spec=None, run=None, passes=()):
    """Mean seconds per item: reading the media, cheap passes, question embedding, router, final pass."""
    passes = spec.passes if spec else passes
    P = [B.col[p] for p in passes]
    media = np.array([c not in MEDIA_FREE for c in pol.choice]) | any(p not in MEDIA_FREE for p in passes)
    final = np.array([0.0 if c in passes else B.model_s[i, B.col[c]] for i, c in zip(pol.idx, pol.choice)])
    parts = {"read media": float((B.decode_s[pol.idx] * media).mean()),
             "cheap passes": float(B.model_s[pol.idx][:, P].sum(axis=1).mean()) if P else 0.0,
             "question embedding": B.embed_s if spec is not None and spec.question else 0.0,
             "router": run["infer_s"] if run else 0.0, "final pass": float(final.mean())}
    parts["total"] = sum(parts.values())
    return parts


def calibration_table(B):
    """Cheap passes in the original option order: calibration before and after temperature scaling
    (temperature fitted on the other folds), and how often they are confidently wrong."""
    rows = []
    for p in dict.fromkeys(x for g in PASSES.values() for x in g):
        z, conf_t, temps = B.logits[p], np.empty(B.n), []
        for k in np.unique(B.fold):
            te = B.fold == k
            T = fit_temperature(z[~te], B.gold[~te])
            temps.append(T)
            conf_t[te] = softmax(z[te] / T, axis=1).max(axis=1)
        conf = softmax(z, axis=1).max(axis=1)
        right = z.argmax(axis=1) == B.gold

        def ece(c):
            bins = np.minimum((c * 10).astype(int), 9)
            return sum(abs(c[bins == b].mean() - right[bins == b].mean()) * (bins == b).mean()
                       for b in range(10) if (bins == b).any())
        row = {"pass": short(p), "accuracy": right.mean(), "mean max prob": conf.mean(), "ECE": ece(conf),
               "temperature": float(np.mean(temps)), "ECE scaled": ece(conf_t),
               "confidently wrong (p>=0.8)": ((conf >= 0.8) & ~right).mean(),
               "confidently wrong, scaled": ((conf_t >= 0.8) & ~right).mean()}
        for t in AUDIO_TYPES:
            m = B.audio_type == t
            row[f"confidently wrong: {t}"] = ((conf[m] >= 0.8) & ~right[m]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def study(B, cfg, args):
    seed, st = cfg["seed"], Stats(args.n_boot, cfg["seed"])
    question = B.emb is not None
    out = repo_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    headline, groups = experiments(args.quick, question)
    cmps = comparisons(headline, question)
    specs = [s for g in groups.values() for s in g] + [s for c in cmps for s in c[1:3]]
    print(f"fitting {len(dict.fromkeys(specs))} routers x 5 folds (+ 3 on the midterm split)", flush=True)
    runs = run_specs(B, specs, TAUS, seed, args.n_jobs)
    mid_runs = run_specs(B, list(headline.values()), REPORT_TAUS, seed, args.n_jobs, split="midterm")
    idx = np.arange(B.n)
    pols = {s: router_policies(B, r, s.label) for s, r in runs.items()}
    cache = {}

    def F(reps, kind="tok", items=idx):
        key = (tuple(reps), kind, tuple(items) if len(items) < B.n else "all")
        if key not in cache:
            cache[key] = Frontier(B, list(reps), items, kind)
        return cache[key]

    F_all = F(B.reps)
    main_menu = MENUS["base7+cells"]
    casc = {name: (cascade_policies(B, p, t, "folds", TAUS, name), [p, t]) for name, p, t in CASCADES}
    oracle, oracle_curve = oracle_policies(B, main_menu, idx, TAUS)
    uniform = uniform_policy(B, main_menu, idx, "Uniform random router (base7+cells)")
    tables, md = {}, []

    def summary(name, pol, own, kind="tok"):
        a = st.ci(pol.acc)
        g = st.ci(own.gain(pol)) if own is not None else (np.nan,) * 3
        cm = pol.acc[B.cross_modal[pol.idx]]
        return {"method": name, "point": pol.point, "items": len(pol.idx), "acc": a[0], "acc_lo": a[1], "acc_hi": a[2],
                "macro_acc": macro(pol, B), "cross_modal_acc": cm.mean() if len(cm) else np.nan,
                "tokens": pol.tok.mean(), "ms_per_item": 1000 * pol.sec.mean(),
                "gain_vs_hull": g[0], "gain_lo": g[1], "gain_hi": g[2],
                "oracle_gap_closed": own.gap_closed(pol) if own is not None else np.nan}

    def fmt(df):
        return pd.DataFrame({
            "method": df.method, "accuracy": [with_ci(*v) for v in zip(df.acc, df.acc_lo, df.acc_hi)],
            "macro": df.macro_acc.map(pct), "cross-modal": df.cross_modal_acc.map(pct),
            "tokens": df.tokens.map(lambda v: f"{v:.2f}"), "ms/item": df.ms_per_item.map(lambda v: f"{v:.0f}"),
            "vs hull at same cost (pts)": [with_ci(*v, signed=True) if not np.isnan(v[0]) else "-"
                                           for v in zip(df.gain_vs_hull, df.gain_lo, df.gain_hi)],
            "oracle gap closed": df.oracle_gap_closed.map(lambda v: "-" if np.isnan(v) else pct(v, signed=True))})

    # ---- headline: operating point tuned for accuracy inside the training folds
    rows = [summary(short(r), fixed_policy(B, r, idx), None) for r in FIXED_SHOWN]
    rows.append(summary(uniform.name, uniform, F(main_menu)))
    rows += [summary(name, p["max"], F(reps)) for name, (p, reps) in casc.items()]
    rows += [summary(name, pols[s]["max"], F(s.reps)) for name, s in headline.items()]
    rows.append(summary("Oracle (base7+cells; sees the answers)", oracle["max"], F(main_menu)))
    tables["headline"] = pd.DataFrame(rows)
    n_cm = int(B.cross_modal.sum())
    md += ["# Idea 1: learned routing between input representations\n",
           f"{B.n} items, 5 frozen folds: every decision is out of fold, and every choice (L2 strength, lambda, "
           f"cascade threshold, temperature, PCA) is made inside the training folds. Accuracy is the mean over the "
           f"3 option orders; tokens are input tokens relative to the full input, including every cheap pass a "
           f"method runs; ms/item is end to end on the L4 (reading the media, all passes, question embedding, "
           f"router). *vs hull*: accuracy minus random mixing of the method's own fixed options at the same mean "
           f"token cost (paired over items); *oracle gap closed*: that gain as a share of the oracle's (same options, "
           f"same cost). Macro = mean of the 3 audio types; cross-modal = the {n_cm} items the "
           f"full input answers while image only and audio only both fail. 95% CIs: {args.n_boot:,} bootstrap "
           f"resamples over items.{'' if question else ' **Run without question embeddings (--no-question).**'}\n",
           "Cheap passes (always paid by the routers that use them; answering with one costs nothing extra): "
           + "; ".join(f"{g} = {' + '.join(short(p) for p in ps)}: {B.tok[:, [B.col[p] for p in ps]].sum(axis=1).mean():.2f}"
                       f" tokens, {1000 * B.model_s[:, [B.col[p] for p in ps]].sum(axis=1).mean():.0f} ms"
                       for g, ps in PASSES.items()) + ".\n",
           "## Headline (operating point chosen for accuracy)\n", md_table(fmt(tables["headline"])), ""]

    # ---- matched cost
    rows = []
    for t in REPORT_TAUS:
        rows += [summary(name, p[t], F(reps)) for name, (p, reps) in casc.items()]
        rows += [summary(name, pols[s][t], F(s.reps)) for name, s in headline.items()]
        rows.append(summary("Oracle", oracle[t], F(main_menu)))
    tables["matched_cost"] = df = pd.DataFrame(rows)
    md += ["## At a target cost (lambda or threshold chosen in the training folds to reach it)\n",
           "A router or cascade that runs cheap passes cannot cost less than them, so below that floor its rows sit "
           "at the floor (see *tokens*).\n",
           md_table(pd.concat([df.point.map(lambda v: f"{v:g}").rename("target tokens"), fmt(df)], axis=1)), ""]

    # ---- research questions: paired, cost-adjusted against random mixing of all fixed representations
    rows = []
    for fam, a, b, what in cmps:
        for p in ("max", 0.5):
            d = F_all.gain(pols[a][p]) - F_all.gain(pols[b][p])
            m, lo, hi, pv = st.test(d)
            rows.append({"question": fam, "comparison": what, "A": a.label, "B": b.label, "point": point_name(p),
                         "acc_A": pols[a][p].acc.mean(), "acc_B": pols[b][p].acc.mean(),
                         "tokens_A": pols[a][p].tok.mean(), "tokens_B": pols[b][p].tok.mean(),
                         "diff": m, "lo": lo, "hi": hi, "p": pv})
    df = pd.DataFrame(rows)
    df["p_holm"] = df.groupby("point").p.transform(holm)
    tables["comparisons"] = df
    md += ["## Research questions (A - B, cost-adjusted, paired over items)\n",
           "Each side is its accuracy minus random mixing of all 25 fixed representations at that side's mean token "
           "cost, so a cheaper router is not penalized for spending less. p: sign-flip test, Holm-adjusted within "
           "each operating point.\n",
           md_table(pd.DataFrame({
               "": df.question, "comparison": df.comparison, "point": df.point,
               "A: acc / tokens": [f"{pct(x)} / {c:.2f}" for x, c in zip(df.acc_A, df.tokens_A)],
               "B: acc / tokens": [f"{pct(x)} / {c:.2f}" for x, c in zip(df.acc_B, df.tokens_B)],
               "A - B (pts)": [with_ci(*v, signed=True) for v in zip(df["diff"], df.lo, df.hi)],
               "p_holm": df.p_holm.map(lambda v: f"{v:.3f}")})), ""]

    # ---- every router variant
    rows = []
    for s, run in runs.items():
        own = F(s.reps)
        g = st.ci(own.gain(pols[s]["max"]))
        J = [B.col[r] for r in s.reps]
        l2s = pd.Series([f["l2"] for f in run["folds"]]).value_counts()
        row = {"router": s.label, "groups": ", ".join(k for k, v in groups.items() if s in v) or "comparison",
               "model": s.model, "menu": s.menu, "features": s.features, "train_size": s.train_size or "all",
               "repeat": s.repeat, "n_features": len(run["folds"][0]["names"]),
               "acc": pols[s]["max"].acc.mean(), "tokens": pols[s]["max"].tok.mean(),
               "gain_max": g[0], "gain_max_lo": g[1], "gain_max_hi": g[2],
               "oracle_gap_closed": own.gap_closed(pols[s]["max"])}
        for t in REPORT_TAUS:
            row[f"tokens@{t:g}"], row[f"gain@{t:g}"] = pols[s][t].tok.mean(), own.gain(pols[s][t]).mean()
        row["added_area"] = added_area(own.vertices, [(p.tok.mean(), p.acc.mean()) for p in pols[s].values()])
        row["spearman"] = spearman_rows(run["S"], B.logp[:, J])
        row["l2"] = ", ".join(f"{k:g}x{v}" for k, v in l2s.items())
        row["converged"] = all(f["converged"] for f in run["folds"])
        rows.append(row)
    tables["variants"] = var = pd.DataFrame(rows)

    def var_md(df):
        return md_table(pd.DataFrame({
            "router": df.router, "acc (tokens), accuracy-tuned": [f"{pct(a)} ({c:.2f})" for a, c in zip(df.acc, df.tokens)],
            "vs hull, accuracy-tuned": [with_ci(*v, signed=True) for v in zip(df.gain_max, df.gain_max_lo, df.gain_max_hi)],
            **{f"vs hull @ {t:g} (tokens)": [f"{pct(g, signed=True)} ({c:.2f})" for g, c in zip(df[f"gain@{t:g}"], df[f"tokens@{t:g}"])]
               for t in REPORT_TAUS},
            "added area": df.added_area.map(lambda v: pct(v, signed=True)),
            "oracle gap closed": df.oracle_gap_closed.map(lambda v: pct(v, signed=True)),
            "Spearman": df.spearman.map(lambda v: f"{v:.3f}"),
            "L2 (folds)": df.l2}))
    md += ["## Router variants\n",
           "*vs hull @ tau*: at the operating point aimed at mean token cost tau (realized cost in brackets: a router "
           "that runs cheap passes cannot go below their cost). "
           "*Added area*: mean accuracy (pts) the router's operating points add to the fixed-option frontier over "
           "costs 0.2-0.8 (0 = never better than random mixing). *Spearman*: per item, router scores vs log p(gold) "
           "over the menu.\n"]
    for gname, title in [("grid", "Model x features x menu"), ("ablations", "Ablations (new router and its pointwise twin)"),
                         ("sweeps", "Sweeps: L2 strength and pair weighting")]:
        if gname in groups:
            sub = var[var.router.isin([s.label for s in dict.fromkeys(groups[gname])])]
            md += [f"### {title}\n", var_md(sub), ""]

    # ---- learning curve
    if "learning" in groups:
        bases = {"New router": headline["New router"], "New router, no cheap passes": headline["New router, no cheap passes"]}
        rows = []
        for router, base in bases.items():
            for model in ("pairwise", "pointwise"):
                b = replace(base, model=model)
                for s_ in [replace(b, train_size=n, repeat=r) for n in LEARN_SIZES for r in range(LEARN_REPEATS)] + [b]:
                    v = var[var.router == s_.label].iloc[0]
                    rows.append({"router": router, "model": model, "n": int(np.mean([f["n_train"] for f in runs[s_]["folds"]])),
                                 "acc": v.acc, "gain_max": v.gain_max, "added_area": v.added_area})
        agg = pd.DataFrame(rows).groupby(["router", "model", "n"], sort=False).agg(
            runs=("acc", "size"), acc=("acc", "mean"), acc_sd=("acc", "std"), gain_max=("gain_max", "mean"),
            gain_max_sd=("gain_max", "std"), added_area=("added_area", "mean")).reset_index()
        tables["learning_curve"] = agg

        def pm(m, sd, signed=False):
            return f"{pct(m, signed)} ± {'-' if np.isnan(sd) else pct(sd)}"
        md += ["## Learning curve (RQ1.4): training items per fold\n",
               f"Random subsamples of the training folds ({LEARN_REPEATS} per size; ± = SD over them); the largest "
               "size is all of them (one run).\n",
               md_table(pd.DataFrame({
                   "router": agg.router, "model": agg.model, "training items": agg.n, "runs": agg.runs,
                   "accuracy (accuracy-tuned)": [pm(a, s_) for a, s_ in zip(agg.acc, agg.acc_sd)],
                   "vs hull, accuracy-tuned": [pm(a, s_, True) for a, s_ in zip(agg.gain_max, agg.gain_max_sd)],
                   "added area": agg.added_area.map(lambda v: pct(v, signed=True))})), ""]

    # ---- by audio type and the cross-modal subset
    shown = [("I+A (fixed)", fixed_policy(B, "full", idx)), ("I+T (fixed)", fixed_policy(B, "img_trans", idx))]
    shown += [(name, p["max"]) for name, (p, _) in casc.items()] + [(n, pols[s]["max"]) for n, s in headline.items()]
    shown.append(("Oracle", oracle["max"]))
    rows = []
    for name, p in shown:
        row = {"method": name}
        for t in AUDIO_TYPES:
            row[t] = st.ci(p.acc[B.audio_type == t])
        row["macro"], row["cross-modal"] = macro(p, B), st.ci(p.acc[B.cross_modal])
        rows.append(row)
    tables["by_audio_type"] = pd.DataFrame([{k: (v[0] if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows])
    md += ["## By audio type (accuracy-tuned operating point)\n",
           md_table(pd.DataFrame([{k: (with_ci(*v) if isinstance(v, tuple) else (pct(v) if k == "macro" else v))
                                   for k, v in r.items()} for r in rows])),
           "", "items: " + ", ".join(f"{t} {(B.audio_type == t).sum()}" for t in AUDIO_TYPES) + f", cross-modal {n_cm}", ""]

    # ---- latency
    rows = [{"method": "I+A (fixed)", **latency_parts(B, fixed_policy(B, "full", idx))}]
    rows += [{"method": name, **latency_parts(B, p["max"], passes=(reps[0],))} for name, (p, reps) in casc.items()]
    rows += [{"method": n, **latency_parts(B, pols[s]["max"], s, runs[s])} for n, s in headline.items()]
    tables["latency"] = lat = pd.DataFrame(rows)
    md += ["## Where the time goes (ms per item, L4, accuracy-tuned operating point)\n",
           md_table(pd.concat([lat[["method"]], lat.drop(columns="method").map(lambda v: f"{1000 * v:.1f}")], axis=1)),
           "", f"Question embedding: SBERT on CPU, batch size 1 ({1000 * B.embed_s:.1f} ms per question)."
           if question else "No question embedding in this run.", ""]

    # ---- where the new router sends items
    rows = []
    for name in ("New router", "New router, no human text"):
        s = headline[name]
        for p in ("max", 0.5):
            pol = pols[s][p]
            for r in s.reps:
                row = {"router": name, "point": point_name(p), "representation": short(r), "all": (pol.choice == r).mean()}
                for t in AUDIO_TYPES:
                    row[t] = (pol.choice[B.audio_type == t] == r).mean()
                rows.append(row)
    tables["routing_shares"] = sh = pd.DataFrame(rows)
    md += ["## Where the routers send items (share of items)\n"]
    for (name, p), g in sh.groupby(["router", "point"], sort=False):
        g = g[g["all"] > 0]
        md += [f"**{name}, {p}**\n", md_table(pd.concat([g[["representation"]], g[["all"] + AUDIO_TYPES].map(pct)], axis=1)), ""]

    # ---- cheap-pass calibration (plan §6.4a)
    tables["calibration"] = cal = calibration_table(B)
    md += ["## The cheap passes: calibration and confident errors (original option order, one pass)\n",
           "Temperature fitted on the other folds. Confidently wrong = top probability >= 0.8 and wrong.\n",
           md_table(pd.concat([cal[["pass"]], cal[["accuracy", "mean max prob", "ECE", "ECE scaled"]].map(pct),
                               cal[["temperature"]].map(lambda v: f"{v:.2f}"),
                               cal.drop(columns=["pass", "accuracy", "mean max prob", "ECE", "ECE scaled",
                                                 "temperature"]).map(pct)], axis=1)), ""]

    # ---- the midterm test split
    te = np.flatnonzero(B.midterm_test)
    F_mid = Frontier(B, B.reps, te)
    rows, mid = [], {}
    for r in ("full", "img_trans"):
        rows.append(summary(f"{short(r)} (fixed)", fixed_policy(B, r, te), None))
    for name, s in headline.items():
        mid[name] = router_policies(B, mid_runs[s], name)["max"]
        rows.append(summary(name, mid[name], Frontier(B, s.reps, te)))
    o, _ = oracle_policies(B, main_menu, te, ())
    rows.append(summary("Oracle (base7+cells)", o["max"], Frontier(B, main_menu, te)))
    tables["midterm_split"] = df = pd.DataFrame(rows)
    d = F_mid.gain(mid["New router"]) - F_mid.gain(mid["Midterm router"])
    m_, lo, hi, pv = st.test(d)
    md += ["## On the midterm test set (120 items; routers trained on the other 1,021)\n",
           md_table(fmt(df)), "", f"New - midterm router, cost-adjusted: {with_ci(m_, lo, hi, signed=True)} pts "
           f"(sign-flip p = {pv:.3f}).", ""]

    # ---- curves, CSVs, figures
    curve_rows = []
    for name, s in headline.items():
        for p, pol in pols[s].items():
            curve_rows.append({"method": name, "point": p, **dict(zip(("acc", "tokens", "seconds"), pol.means))})
    for name, (p_, _) in casc.items():
        for p, pol in p_.items():
            curve_rows.append({"method": name, "point": p, **dict(zip(("acc", "tokens", "seconds"), pol.means))})
    for r in B.reps:
        curve_rows.append({"method": short(r), "point": "fixed", **dict(zip(("acc", "tokens", "seconds"),
                                                                            fixed_policy(B, r, idx).means))})
    curve_rows += [{"method": "Oracle", "point": "curve", "acc": a, "tokens": c, "seconds": np.nan} for c, a in oracle_curve]
    tables["curves"] = pd.DataFrame(curve_rows)
    choices = pd.DataFrame({"id": B.ids, "audio_type": B.audio_type, "fold": B.fold})
    for name, s in headline.items():
        for p in ("max", 0.5):
            choices[f"{name} | {point_name(p)}"] = pols[s][p].choice
    tables["choices"] = choices
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    figures(B, tables, headline, out)
    md += ["## Files\n", "CSVs: " + ", ".join(f"`{n}.csv`" for n in tables) + ". Figures: `router_curves.pdf/png`"
           + (", `learning_curve.pdf/png`" if "learning_curve" in tables else "") + ".", ""]
    (out / "report.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps({
        "items": B.n, "question_embeddings": question, "n_boot": args.n_boot, "quick": args.quick,
        "headline": tables["headline"].to_dict(orient="records"),
        "specs": {n: asdict(s) for n, s in headline.items()}}, indent=2, default=float))
    return tables, md


# ----------------------------------------------------------------------------- figures
def figures(B, tables, headline, out):
    """Accuracy vs cost (tokens, and seconds on the L4) for the headline routers and a cascade, over the fixed
    options of the base7+cells menu and their hull; and the learning curve. The oracle is left out (it sits near 70%
    and would flatten everything else); it is in the tables."""
    import matplotlib.pyplot as plt

    from src.common import COL_W, INK_2, PAGE_W, PALETTE, save_fig, set_plot_style
    set_plot_style()
    cur = tables["curves"]
    series = [("New router", PALETTE[0], "-"), ("Midterm router", PALETTE[1], "-"),
              ("New router, no human text", PALETTE[2], "-"), ("Cascade v64_a0.5 -> I+A", PALETTE[3], "--"),
              ("New router, no cheap passes", PALETTE[4], "-")]
    fixed = cur[(cur.point == "fixed") & cur.method.isin([short(r) for r in MENUS["base7+cells"]])]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.6), sharey=True)
    for ax, kind, xlabel in [(axes[0], "tokens", "Cost (input tokens / full input, incl. cheap passes)"),
                             (axes[1], "seconds", "Time per item on the L4 (ms)")]:
        scale = 1000 if kind == "seconds" else 1
        hull = upper_hull([(c, a, m) for c, a, m in zip(fixed[kind], fixed.acc, fixed.method)])
        ax.plot([v[0] * scale for v in hull], [100 * v[1] for v in hull], color=INK_2, ls=":", lw=1.0,
                label="Random mixing of fixed options (hull)", zorder=1)
        for name, color, ls in series:
            c = cur[(cur.method == name) & (cur.point != "max")].sort_values(kind).drop_duplicates(["tokens", "acc"])
            ax.plot(c[kind] * scale, 100 * c.acc, color=color, ls=ls, lw=1.4, label=name, zorder=3)
            mx = cur[(cur.method == name) & (cur.point == "max")]
            ax.scatter(mx[kind] * scale, 100 * mx.acc, s=24, color=color, edgecolors="white", linewidths=1.0, zorder=4)
        ax.scatter(fixed[kind] * scale, 100 * fixed.acc, s=16, color="#8a8984", edgecolors="white", linewidths=0.8,
                   zorder=2)
        shown = {v[2] for v in hull} | {"I+A"}  # label the hull's corners and the full input only
        for _, r in fixed[fixed.method.isin(shown)].iterrows():
            ax.annotate(r.method, (r[kind] * scale, 100 * r.acc), xytext=(4, -3), textcoords="offset points",
                        fontsize=5.5, color=INK_2, ha="left", va="top", zorder=5)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=2.2)
    fig.subplots_adjust(wspace=0.06)
    save_fig(fig, out, "router_curves")

    if "learning_curve" in tables:
        lc = tables["learning_curve"]
        routers = list(dict.fromkeys(lc.router))
        fig, axes = plt.subplots(1, len(routers), figsize=(PAGE_W * 0.62, 2.1), sharey=True, squeeze=False)
        for ax, router in zip(axes[0], routers):
            for model, color in (("pairwise", PALETTE[0]), ("pointwise", PALETTE[1])):
                g = lc[(lc.router == router) & (lc.model == model)].sort_values("n")
                ax.errorbar(g.n, 100 * g.gain_max, yerr=100 * g.gain_max_sd.fillna(0), color=color, lw=1.4,
                            marker="o", ms=3.5, capsize=2, elinewidth=0.8, label=model)
            ax.axhline(0, color=INK_2, lw=0.6, ls=":")
            ax.set_title(router)
            ax.set_xlabel("Training items per fold")
        axes[0][0].set_ylabel("Accuracy minus hull (pts)")
        axes[0][-1].legend(loc="lower right")
        fig.subplots_adjust(wspace=0.08)
        save_fig(fig, out, "learning_curve")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--quick", action="store_true", help="headline routers and the model x features x menu grid only")
    ap.add_argument("--no-question", action="store_true", help="no question embeddings (no SBERT needed)")
    ap.add_argument("--embed-only", action="store_true", help="only compute the question embeddings")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="parallel processes (default: $MMML_JOBS, set to 4 on Modal, else all CPUs)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.n_jobs = args.n_jobs or int(os.environ.get("MMML_JOBS", 0)) or -1
    if args.embed_only or (not args.no_question and not repo_path(EMB_CACHE).exists()):
        try:
            compute_embeddings(cfg)
        except ImportError as e:
            raise SystemExit(f"cannot compute the question embeddings here ({e}). Run on Modal "
                             "(modal run scripts/modal_run.py --sections router), or pass --no-question")
        if args.embed_only:
            return
    t0 = time.time()
    B = Bank(cfg, question=not args.no_question)
    print(f"{B.n} items x {len(B.reps)} representations; {B.cross_modal.sum()} cross-modal items", flush=True)
    tables, md = study(B, cfg, args)
    head = tables["headline"]
    print("\nHeadline (operating point chosen for accuracy inside the training folds):\n")
    print(md_table(pd.DataFrame({"method": head.method, "accuracy": head.acc.map(pct), "tokens": head.tokens.round(2),
                                 "vs hull": head.gain_vs_hull.map(lambda v: "-" if np.isnan(v) else pct(v, signed=True))})))
    print(f"\nwritten: {repo_path(args.out_dir)}/report.md and CSVs, figures ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    from src.final.idea1_router import main as _main  # run from the importable module, so worker processes can unpickle
    _main()
