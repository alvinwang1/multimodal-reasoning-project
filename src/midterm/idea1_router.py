"""Idea 1 baseline: a learned representation router, trained on the train split, evaluated on test.

For every item i and input configuration c, the label is the backbone's correctness averaged over the
three option orders (soft label in {0, 1/3, 2/3, 1}); cost is input tokens relative to the full input.
One L2-regularised logistic regression per configuration estimates P(correct_c | x). The routing policy
picks argmax_c P_hat_c(x) - lambda * cost_c; sweeping lambda traces an accuracy-cost curve.

Router variants (features):
  probe       caption+transcript pass (sorted option probs, margin, entropy) + question embedding
              + audio type + surrogate lengths; pays the probe's cost on every item
  free        question embedding + audio type + surrogate lengths (no extra model pass)
  *_noaudio   ablations without the audio-type indicator (unknown at deployment without a classifier)
Baselines: each fixed configuration, the convex hull of fixed configurations (= random mixing at matched
cost), confidence cascades (caption+transcript -> image+transcript / full), and the oracle router.

    python -m src.midterm.idea1_router
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.common import (AUDIO_TYPE_COLORS, COL_W, CONFIG_COLORS, CONFIG_LABELS, CONFIG_ORDER, INK, INK_2, PALETTE,
                        load_config, read_jsonl, repo_path, save_fig, set_plot_style)
from src.data import AUDIO_TYPES, load_omnibench, text_records

PROBE = "cap_trans"
LAMBDAS = np.round(np.linspace(0, 1.5, 301), 4)
VARIANTS = {"probe": dict(probe=True, audio=True), "free": dict(probe=False, audio=True),
            "probe_noaudio": dict(probe=True, audio=False), "free_noaudio": dict(probe=False, audio=False)}
VARIANT_LABELS = {"probe": "Router (probe)", "free": "Router (probe-free)",
                  "probe_noaudio": "Router (probe, no audio type)", "free_noaudio": "Router (probe-free, no audio type)"}


# ----------------------------------------------------------------------------- data
def load_tables(cfg, split):
    df = pd.DataFrame(read_jsonl(repo_path(cfg["analysis_a"]["out"])))
    df = df[df.config.isin(CONFIG_ORDER)].drop_duplicates(["id", "config", "perm"], keep="last")
    counts = df.groupby("id").size()
    complete = set(counts[counts == len(CONFIG_ORDER) * 3].index)
    ids = {s: [i for i in split[s] if i in complete] for s in ("train", "test")}
    df = df[df.id.isin(complete)]
    acc = df.groupby(["id", "config"]).correct.mean().unstack()[CONFIG_ORDER]
    p0 = df[df.perm == 0].set_index(["id", "config"])
    tok = p0.n_total_tok.unstack()[CONFIG_ORDER]
    cost = tok.div(tok["full"], axis=0)
    meta = p0.xs("full", level="config")[["audio_type", "task_type"]]
    probe = p0.xs(PROBE, level="config")
    lengths = pd.DataFrame({"caption_tok": p0.xs("cap_aud", level="config").n_surrogate_tok,
                            "transcript_tok": p0.xs("img_trans", level="config").n_surrogate_tok})
    return ids, acc, cost, meta, probe, lengths


def probe_features(probe):
    p = np.sort(np.stack(probe.probs.values), axis=1)[:, ::-1]  # sorted desc, original order (perm 0)
    ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(1)
    return pd.DataFrame({"p1": p[:, 0], "p2": p[:, 1], "p3": p[:, 2], "margin": p[:, 0] - p[:, 1],
                         "entropy": ent}, index=probe.index)


def question_embeddings(cfg, ids, cache):
    cache = repo_path(cache)
    if cache.exists():
        z = np.load(cache)
        emb = dict(zip(z["ids"].tolist(), z["emb"]))
        if all(i in emb for i in ids):
            return emb
    from sentence_transformers import SentenceTransformer
    records, _ = text_records(load_omnibench(cfg))
    q = {r["index"]: r["question"] for r in records}
    model = SentenceTransformer(cfg["analysis_c"]["sbert_model"], device="cpu")
    E = model.encode([q[i] for i in ids], normalize_embeddings=True, show_progress_bar=False)
    np.savez(cache, ids=np.array(ids), emb=E)
    return dict(zip(ids, E))


def build_X(ids, variant, pf, emb_pca, meta, lengths):
    parts = [pd.DataFrame(emb_pca.loc[ids].values, index=ids).add_prefix("q")]
    parts.append(np.log1p(lengths.loc[ids]))
    if VARIANTS[variant]["audio"]:
        parts.append(pd.get_dummies(meta.loc[ids, "audio_type"]).reindex(columns=AUDIO_TYPES, fill_value=0)
                     .astype(float).add_prefix("aud_"))
    if VARIANTS[variant]["probe"]:
        parts.append(pf.loc[ids])
    return pd.concat(parts, axis=1)


# ----------------------------------------------------------------------------- model
class Router:
    """One logistic regression per configuration on soft (order-averaged) correctness labels."""

    def __init__(self, C=0.5, seed=42):
        self.C, self.seed = C, seed

    def fit(self, X, acc):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.models = {}
        for c in CONFIG_ORDER:
            y = acc[c].values
            # soft labels via weighted duplication: (x, 1) with weight y and (x, 0) with weight 1 - y
            Xd = np.vstack([Xs, Xs])
            yd = np.r_[np.ones(len(y)), np.zeros(len(y))]
            w = np.r_[y, 1 - y]
            if w[yd == 1].sum() == 0 or w[yd == 0].sum() == 0:
                self.models[c] = float(y.mean())
                continue
            self.models[c] = LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed).fit(
                Xd, yd, sample_weight=w)
        self.columns = list(X.columns)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X[self.columns])
        return pd.DataFrame({c: (m.predict_proba(Xs)[:, 1] if not isinstance(m, float) else np.full(len(X), m))
                             for c, m in self.models.items()}, index=X.index)[CONFIG_ORDER]


def effective_cost(cost, probe):
    """Cost actually paid: with a probe pass, the probe is always paid; escalating adds the target's cost."""
    if not probe:
        return cost
    eff = cost.add(cost[PROBE], axis=0)
    eff[PROBE] = cost[PROBE]
    return eff


def route(P, eff_cost, lam):
    choice = (P - lam * eff_cost.values).idxmax(axis=1)
    return choice


def evaluate(choice, acc, eff_cost):
    a = np.array([acc.at[i, c] for i, c in choice.items()])
    k = np.array([eff_cost.at[i, c] for i, c in choice.items()])
    return a.mean(), k.mean()


def hull(points):
    """Upper-left convex hull of (cost, acc) points = best accuracy reachable by mixing fixed configs."""
    pts = sorted(points)
    out = []
    for p in pts:
        while len(out) >= 2:
            (x1, y1), (x2, y2) = out[-2], out[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                out.pop()
            else:
                break
        out.append(p)
    best, mono = -1, []
    for p in out:  # keep the increasing part
        if p[1] > best:
            mono.append(p)
            best = p[1]
    return mono


def hull_acc_at(h, x):
    xs, ys = zip(*h)
    return float(np.interp(x, xs, ys))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--C", type=float, default=0.5, help="inverse L2 strength of the per-config classifiers")
    ap.add_argument("--pca", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_plot_style()
    out = repo_path(cfg["midterm"]["out_dir"]) / "idea1"
    out.mkdir(parents=True, exist_ok=True)

    split = json.loads(repo_path(cfg["midterm"]["split"]).read_text())
    ids, acc, cost, meta, probe, lengths = load_tables(cfg, split)
    tr, te = ids["train"], ids["test"]
    print(f"complete items: train {len(tr)}/{len(split['train'])}, test {len(te)}/{len(split['test'])}")

    pf = probe_features(probe)
    emb = question_embeddings(cfg, tr + te, cfg["midterm"]["out_dir"] + "/question_emb.npz")
    E = pd.DataFrame(np.stack([emb[i] for i in tr + te]), index=tr + te)
    pca = PCA(n_components=min(args.pca, len(tr) - 1), random_state=cfg["seed"]).fit(E.loc[tr])
    emb_pca = pd.DataFrame(pca.transform(E), index=E.index)

    rows, curves = [], {}
    # fixed configurations
    for c in CONFIG_ORDER:
        rows.append({"method": CONFIG_LABELS[c], "kind": "fixed", "acc": acc.loc[te, c].mean(),
                     "cost": cost.loc[te, c].mean()})
    fixed_pts = [(r["cost"], r["acc"]) for r in rows]
    H = hull(fixed_pts)
    best_fixed = max(fixed_pts, key=lambda p: p[1])

    # oracle (per-item best accuracy minus lambda * cost, ties -> cheapest)
    oc = []
    for lam in LAMBDAS:
        ch = (acc.loc[te] - lam * cost.loc[te] - 1e-6 * cost.loc[te]).idxmax(axis=1)
        oc.append((lam, *evaluate(ch, acc, cost)))
    curves["oracle"] = pd.DataFrame(oc, columns=["lambda", "acc", "cost"])
    oracle_acc = curves["oracle"].acc.iloc[0]
    oracle_cost = curves["oracle"].cost.iloc[0]
    rows.append({"method": "Oracle router", "kind": "oracle", "acc": oracle_acc, "cost": oracle_cost})

    # confidence cascades: probe, escalate when max prob < tau (tau tuned on train for max accuracy)
    taus = np.round(np.arange(0.25, 1.0001, 0.01), 3)
    for target in ["img_trans", "full"]:
        res = []
        for t in taus:
            for name, items in (("train", tr), ("test", te)):
                esc = pf.loc[items, "p1"] < t
                a = np.where(esc, acc.loc[items, target], acc.loc[items, PROBE]).mean()
                k = (cost.loc[items, PROBE] + esc * cost.loc[items, target]).mean()
                res.append({"tau": t, "set": name, "acc": a, "cost": k, "esc": esc.mean()})
        r = pd.DataFrame(res)
        curves[f"cascade_{target}"] = r[r.set == "test"].reset_index(drop=True)
        t_star = r[r.set == "train"].sort_values(["acc", "cost"], ascending=[False, True]).iloc[0].tau
        pt = r[(r.set == "test") & (r.tau == t_star)].iloc[0]
        rows.append({"method": f"Cascade -> {CONFIG_LABELS[target]} (tau={t_star:.2f}, tuned on train)",
                     "kind": "cascade", "acc": pt.acc, "cost": pt.cost, "escalation": pt.esc})

    # learned routers
    preds, chosen_lams = {}, {}
    for v, spec in VARIANTS.items():
        Xtr = build_X(tr, v, pf, emb_pca, meta, lengths)
        Xte = build_X(te, v, pf, emb_pca, meta, lengths)
        model = Router(C=args.C, seed=cfg["seed"]).fit(Xtr, acc.loc[tr])
        Ptr, Pte = model.predict(Xtr), model.predict(Xte)
        eff_tr, eff_te = effective_cost(cost.loc[tr], spec["probe"]), effective_cost(cost.loc[te], spec["probe"])
        res = []
        for lam in LAMBDAS:
            a_tr, k_tr = evaluate(route(Ptr, eff_tr, lam), acc, eff_tr)
            a_te, k_te = evaluate(route(Pte, eff_te, lam), acc, eff_te)
            res.append({"lambda": lam, "train_acc": a_tr, "train_cost": k_tr, "acc": a_te, "cost": k_te})
        r = pd.DataFrame(res)
        curves[v] = r
        # operating point chosen on train: max train accuracy (ties -> cheaper)
        lam_star = r.sort_values(["train_acc", "train_cost"], ascending=[False, True]).iloc[0]["lambda"]
        chosen_lams[v] = lam_star
        pt = r[r["lambda"] == lam_star].iloc[0]
        rows.append({"method": f"{VARIANT_LABELS[v]} (lambda={lam_star:.2f}, tuned on train)", "kind": "router",
                     "acc": pt.acc, "cost": pt.cost})
        preds[v] = (Pte, eff_te, model, Xte.columns)

    table = pd.DataFrame(rows)
    table["hull_acc_at_cost"] = [hull_acc_at(H, k) for k in table.cost]
    table["gain_over_hull"] = table.acc - table.hull_acc_at_cost
    denom = oracle_acc - best_fixed[1]
    table["oracle_gap_closed"] = (table.acc - best_fixed[1]) / denom if denom > 0 else np.nan
    table.to_csv(out / "results_test.csv", index=False)
    print("\nTest-set results (accuracy = mean over 3 option orders; cost = tokens / full-input tokens):")
    print(table.round(3).to_string(index=False))
    for k, v in curves.items():
        v.to_csv(out / f"curve_{k}.csv", index=False)

    # per-lambda comparison of router vs hull on test (does the router ever beat mixing fixed configs?)
    for v in VARIANTS:
        r = curves[v]
        gains = r.acc - [hull_acc_at(H, k) for k in r.cost]
        print(f"{v:14s}: max gain over fixed-config hull on test = {gains.max():+.3f} at cost "
              f"{r.cost[gains.idxmax()]:.2f}; share of lambdas with gain > 0: {(gains > 0).mean():.0%}")

    # ---------------- failure analysis for the main router (probe variant) at its train-tuned lambda
    Pte, eff_te, model, cols = preds["probe"]
    ch = route(Pte, eff_te, chosen_lams["probe"])
    best_acc = acc.loc[te].max(axis=1)
    oracle_cfg = (acc.loc[te] - 1e-6 * cost.loc[te]).idxmax(axis=1)
    fa = pd.DataFrame({"chosen": ch, "oracle": oracle_cfg,
                       "acc_chosen": [acc.at[i, c] for i, c in ch.items()], "acc_best": best_acc,
                       "p_hat_chosen": [Pte.at[i, c] for i, c in ch.items()],
                       "probe_p1": pf.loc[te, "p1"], "audio_type": meta.loc[te, "audio_type"],
                       "task_type": meta.loc[te, "task_type"]})
    fa["regret"] = fa.acc_best - fa.acc_chosen
    fa["unsolvable"] = fa.acc_best == 0
    records, _ = text_records(load_omnibench(cfg))
    by_id = {r["index"]: r for r in records}
    fa["question"] = [by_id[i]["question"] for i in fa.index]
    fa.sort_values("regret", ascending=False).to_csv(out / "failure_cases_probe.csv")

    conf = pd.crosstab(fa.chosen, fa.oracle).reindex(index=CONFIG_ORDER, columns=CONFIG_ORDER, fill_value=0)
    conf.to_csv(out / "confusion_chosen_vs_oracle.csv")
    by_aud = fa.groupby("audio_type").agg(n=("regret", "size"), router_acc=("acc_chosen", "mean"),
                                          oracle_acc=("acc_best", "mean"), mean_regret=("regret", "mean"),
                                          unsolvable=("unsolvable", "mean")).reindex(AUDIO_TYPES)
    by_task = fa.groupby("task_type").agg(n=("regret", "size"), router_acc=("acc_chosen", "mean"),
                                          oracle_acc=("acc_best", "mean"), mean_regret=("regret", "mean"))
    by_aud.to_csv(out / "failure_by_audio_type.csv")
    by_task.to_csv(out / "failure_by_task_type.csv")
    solv = fa[~fa.unsolvable]
    calib = pd.DataFrame({"p_hat": fa.p_hat_chosen, "acc": fa.acc_chosen})
    calib["bin"] = pd.cut(calib.p_hat, [0, .3, .45, .6, .75, 1.0])
    calib_tab = calib.groupby("bin", observed=True).agg(n=("acc", "size"), mean_p_hat=("p_hat", "mean"),
                                                        actual_acc=("acc", "mean"))
    calib_tab.to_csv(out / "calibration_probe.csv")
    print(f"\nFailure analysis (probe router, lambda={chosen_lams['probe']:.2f}):")
    print("chosen configs:", fa.chosen.value_counts().to_dict())
    print(f"unsolvable items (no config ever correct): {fa.unsolvable.mean():.0%}; "
          f"router regret on solvable items: {solv.regret.mean():.3f}; "
          f"picked a best-accuracy config on {(solv.regret == 0).mean():.0%} of solvable items")
    print("by audio type:\n" + by_aud.round(3).to_string())
    print("calibration of P_hat for the chosen config:\n" + calib_tab.round(3).to_string())
    coefs = pd.DataFrame({c: m.coef_[0] for c, m in model.models.items() if not isinstance(m, float)},
                         index=cols)
    coefs.to_csv(out / "coefficients_probe.csv")
    print("probe-feature coefficients (standardised):\n" +
          coefs.loc[[c for c in ["p1", "margin", "entropy", "aud_speech", "aud_sound event", "aud_music"]
                     if c in coefs.index]].round(2).to_string())

    # ---------------- figure: accuracy vs cost
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    hx, hy = zip(*H)
    ax.plot(hx, np.array(hy) * 100, color=INK_2, ls=":", lw=0.9, label="Fixed-config hull", zorder=1)
    o = curves["oracle"]
    ax.plot(o.cost, o.acc * 100, color=INK, lw=0.9, ls="--", label="Oracle router", zorder=1)
    for v, col in [("probe", PALETTE[6]), ("free", PALETTE[4])]:
        r = curves[v].drop_duplicates(["acc", "cost"])
        ax.plot(r.cost, r.acc * 100, color=col, lw=1.4, label=VARIANT_LABELS[v], zorder=2)
    cas = curves["cascade_img_trans"]
    ax.plot(cas.cost, cas.acc * 100, color=PALETTE[3], lw=1.0, label="Cascade → Image+Transcript", zorder=2)
    # label each fixed config next to its dot; offsets (points) chosen to avoid the curves
    label_pos = {"full": ((0, -6), "center", "top"), "img_trans": ((0, 5), "center", "bottom"),
                 "cap_aud": ((5, 0), "left", "center"), "cap_trans": ((-4, 4), "right", "bottom"),
                 "img_only": ((5, 0), "left", "center"), "aud_only": ((0, -6), "center", "top"),
                 "q_only": ((5, 0), "left", "center")}
    for c in CONFIG_ORDER:
        x, y = cost.loc[te, c].mean(), acc.loc[te, c].mean() * 100
        ax.scatter(x, y, s=20, color=CONFIG_COLORS[c], edgecolors="white", linewidths=0.6, zorder=3)
        off, ha, va = label_pos[c]
        ax.annotate(CONFIG_LABELS[c], (x, y), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=4.8, color=INK_2, zorder=4)
    ax.set_xlabel("Mean cost (fraction of full-input tokens)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.4)
    save_fig(fig, out, "router_curve")

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    im = ax.imshow(conf.values, cmap="Blues")
    short = {"full": "I+A", "img_trans": "I+T", "cap_aud": "C+A", "cap_trans": "C+T", "img_only": "I",
             "aud_only": "A", "q_only": "Q"}
    ax.set_xticks(range(7), [short[c] for c in CONFIG_ORDER])
    ax.set_yticks(range(7), [short[c] for c in CONFIG_ORDER])
    ax.set_xlabel("Oracle's best config")
    ax.set_ylabel("Router's choice")
    ax.grid(False)
    for (r_, c_), v in np.ndenumerate(conf.values):
        if v:
            ax.text(c_, r_, int(v), ha="center", va="center", fontsize=5.5,
                    color="white" if v > conf.values.max() / 2 else INK)
    save_fig(fig, out, "router_confusion")

    (out / "summary.json").write_text(json.dumps({
        "n_train": len(tr), "n_test": len(te), "lambdas_tuned_on_train": chosen_lams,
        "oracle_acc": oracle_acc, "oracle_cost": oracle_cost, "best_fixed": {"cost": best_fixed[0], "acc": best_fixed[1]},
        "results": table.round(4).to_dict(orient="records")}, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
