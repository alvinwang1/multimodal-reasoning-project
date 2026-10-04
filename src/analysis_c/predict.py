"""Predict the text-sufficient label from cheap language signals (Analysis C).

Predictors, all scored with stratified k-fold CV (AUROC mean +- std over folds):
  (a) logistic regression on SBERT embeddings (one per text view)
  (b) logistic regression on Empath features (one per text view)
  (c) the model's max option probability under caption+transcript (Analysis A, perm 0; no training)
  (d) majority-class baseline
Plus a learning curve for (a): AUROC vs. training-set fraction, per outer fold.

    python -m src.analysis_c.predict
    python -m src.analysis_c.predict --labels human     # outputs are suffixed _humanlabels
"""
import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis_c.features import VIEW_LABELS, VIEWS
from src.analysis_c.labels import FILE_SUFFIX, LABEL_TAGS, load_labels
from src.common import COL_W, INK_2, PALETTE, load_config, read_jsonl, repo_path, save_fig, set_plot_style

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def emb_model(seed):
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)


def empath_model(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced",
                                                              random_state=seed))


def pick_folds(y, k):
    """Largest k <= requested such that every fold can hold both classes."""
    k_ok = min(k, int(np.bincount(y).min()))
    if k_ok < k:
        warnings.warn(f"minority class has {np.bincount(y).min()} examples; using {k_ok}-fold CV instead of {k}")
    if k_ok < 2:
        raise SystemExit("fewer than 2 examples in a class: AUROC is undefined, nothing to evaluate")
    return k_ok


def cv_scores(make, X, y, folds):
    """Out-of-fold scores per fold -> (per-fold AUROCs, list of (y_true, score) per fold)."""
    aucs, per_fold = [], []
    for tr, te in folds:
        if X is None:  # (c): the score column is the prediction itself
            s = make[te]
        else:
            m = make()
            m.fit(X[tr], y[tr])
            s = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s))
        per_fold.append((y[te], s))
    return aucs, per_fold


def mean_roc(per_fold, grid=np.linspace(0, 1, 101)):
    tprs = []
    for yt, s in per_fold:
        fpr, tpr, _ = roc_curve(yt, s)
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    return grid, np.mean(tprs, 0), np.std(tprs, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", choices=["model", "human"], default="model")
    ap.add_argument("--raw", default=None, help="Analysis A JSONL (for predictor c)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ccfg, seed = cfg["analysis_c"], cfg["seed"]
    set_plot_style()
    tag, suffix = LABEL_TAGS[args.labels], FILE_SUFFIX[args.labels]
    out_dir = repo_path(ccfg["out_dir"])

    F = dict(np.load(repo_path(ccfg["features"])))
    labels = load_labels(args.labels)
    if labels is None:
        raise SystemExit(f"no {args.labels} labels found; run Analysis A (or B) analyze.py first")
    keep = np.array([int(i) in labels for i in F["ids"]])
    ids = F["ids"][keep]
    y = np.array([labels[int(i)] for i in ids])
    print(f"labels: {args.labels} | n={len(y)} | positive (text-sufficient) rate {y.mean():.2f}")

    k = pick_folds(y, ccfg["cv_folds"])
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

    # (c) model confidence under caption+transcript, original option order
    raw = read_jsonl(repo_path(args.raw or cfg["analysis_a"]["out"]))
    conf = {r["id"]: r["max_prob"] for r in raw if r["config"] == "cap_trans" and r["perm"] == 0}
    conf_score = np.array([conf.get(int(i), np.nan) for i in ids])

    results, rocs = [], {}
    for v in VIEWS:
        X = F[f"emb_{v}"][keep]
        aucs, pf = cv_scores(lambda: emb_model(seed), X, y, folds)
        results.append({"predictor": "(a) SBERT + LogReg", "view": v, "aucs": aucs})
        rocs[f"(a) SBERT, {VIEW_LABELS[v]}"] = pf
        X = F[f"empath_{v}"][keep]
        aucs, pf = cv_scores(lambda: empath_model(seed), X, y, folds)
        results.append({"predictor": "(b) Empath + LogReg", "view": v, "aucs": aucs})
        rocs[f"(b) Empath, {VIEW_LABELS[v]}"] = pf
    if not np.isnan(conf_score).any():
        aucs, pf = cv_scores(conf_score, None, y, folds)
        results.append({"predictor": "(c) Omni max prob (cap+trans)", "view": "-", "aucs": aucs})
        rocs["(c) Omni max prob"] = pf
    else:
        print("WARNING: Analysis A cap_trans perm-0 results missing for some ids; skipping (c)")
    aucs, pf = cv_scores(lambda: DummyClassifier(strategy="most_frequent"), np.zeros((len(y), 1)), y, folds)
    results.append({"predictor": "(d) Majority class", "view": "-", "aucs": aucs})
    rocs["(d) Majority"] = pf

    table = pd.DataFrame([{"predictor": r["predictor"], "view": r["view"], "auroc_mean": np.mean(r["aucs"]),
                           "auroc_std": np.std(r["aucs"]), "folds": k} for r in results])
    table.to_csv(out_dir / f"predict_auroc{suffix}.csv", index=False)
    print(f"\nAUROC ({k}-fold stratified CV){tag}")
    print(table.round(3).to_string(index=False))

    # ROC figure: best view for (a) and (b), plus (c) and (d).
    def best(pred):
        sub = table[table.predictor == pred]
        return sub.loc[sub.auroc_mean.idxmax(), "view"]
    shown = [(f"(a) SBERT, {VIEW_LABELS[best('(a) SBERT + LogReg')]}", PALETTE[0]),
             (f"(b) Empath, {VIEW_LABELS[best('(b) Empath + LogReg')]}", PALETTE[1]),
             ("(c) Omni max prob", PALETTE[2]), ("(d) Majority", INK_2)]
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.8))
    for name, color in shown:
        if name not in rocs:
            continue
        g, m, s = mean_roc(rocs[name])
        auc = np.mean([roc_auc_score(a, b) for a, b in rocs[name]])
        ls = "--" if name.startswith("(d)") else "-"
        ax.plot(g, m, color=color, ls=ls, label=f"{name} ({auc:.2f})")
        ax.fill_between(g, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=color, alpha=0.12, lw=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Predicting text-sufficiency ({k}-fold CV){tag}")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=5.5, handlelength=1.5)
    save_fig(fig, out_dir, f"predict_roc{suffix}")

    # Learning curve for (a): subsample each outer training fold, stratified.
    rng = np.random.default_rng(seed)
    curves = {}
    for v in VIEWS:
        X = F[f"emb_{v}"][keep]
        per_frac = {f: [] for f in FRACTIONS}
        for tr, te in folds:
            for f in FRACTIONS:
                sub = []
                for cls in (0, 1):
                    pool = tr[y[tr] == cls]
                    n = max(1, int(round(f * len(pool))))
                    sub += rng.choice(pool, size=n, replace=False).tolist()
                sub = np.array(sub)
                m = emb_model(seed).fit(X[sub], y[sub])
                per_frac[f].append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        curves[v] = per_frac
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.7))
    n_train = len(y) * (k - 1) / k
    for c, v in zip(PALETTE, VIEWS):
        m = np.array([np.mean(curves[v][f]) for f in FRACTIONS])
        s = np.array([np.std(curves[v][f]) for f in FRACTIONS])
        x = np.array(FRACTIONS) * 100
        ax.plot(x, m, color=c, marker="o", ms=3, label=VIEW_LABELS[v])
        ax.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.12, lw=0)
    ax.axhline(0.5, color=INK_2, ls="--", lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Training-set fraction (%) of ~{n_train:.0f} examples")
    ax.set_ylabel("AUROC")
    ax.set_title(f"SBERT + LogReg learning curve{tag}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=5.5, handlelength=1.2,
              columnspacing=0.8)
    save_fig(fig, out_dir, f"predict_learning_curve{suffix}")
    lc = pd.DataFrame([{"view": v, "fraction": f, "auroc_mean": np.mean(curves[v][f]),
                        "auroc_std": np.std(curves[v][f])} for v in VIEWS for f in FRACTIONS])
    lc.to_csv(out_dir / f"predict_learning_curve{suffix}.csv", index=False)
    (out_dir / f"predict_meta{suffix}.json").write_text(json.dumps(
        {"labels": args.labels, "n": int(len(y)), "positive_rate": float(y.mean()), "folds": k, "seed": seed},
        indent=2))
    print(f"wrote figures/tables to {out_dir}")


if __name__ == "__main__":
    main()
