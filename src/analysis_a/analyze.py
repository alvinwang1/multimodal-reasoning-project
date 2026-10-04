"""Analysis A: tables and figures from the runner's JSONL.

    python -m src.analysis_a.analyze
    python -m src.analysis_a.analyze --raw results/analysis_a/raw.jsonl --lambda-max 1.5

Outputs (results/analysis_a/, PDF + PNG for every figure):
  acc_by_config_audio      grouped bars, accuracy by config x audio type (Wilson 95% CI)
  acc_vs_tokens            accuracy vs. mean input tokens, Pareto frontier marked
  oracle_routing           distribution of each example's cheapest correct config
  utility_sweep            U = acc - lambda * cost, optimal config as lambda varies
  cascade                  caption+transcript -> full escalation by confidence threshold
  tables/*.csv, summary.json, modality_costs.json
  ../text_sufficient_labels.json
Only examples with every (config, perm) present are used; perm 0 = original option order.
Accuracy is the mean over option permutations (OmniBench's answer positions are skewed, and the model
has a letter preference, so single-order accuracy is biased); perm-0 accuracy is reported alongside.
The oracle counts an example as solved by a config only if it is correct under every permutation.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from src.common import (AUDIO_TYPE_COLORS, COL_W, CONFIG_COLORS, CONFIG_LABELS, CONFIG_ORDER, INK, INK_2,
                        PALETTE, load_config, read_jsonl, repo_path, save_fig, set_plot_style)
from src.data import AUDIO_TYPES

SHORT = {"full": "Img+Aud", "img_trans": "Img+Trans", "cap_aud": "Cap+Aud", "cap_trans": "Cap+Trans",
         "img_only": "Img", "aud_only": "Aud", "q_only": "Q"}


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, method="wilson")
    return lo, hi


def load(raw_path, configs, perm_configs, n_perms):
    df = pd.DataFrame(read_jsonl(raw_path))
    if df.empty:
        raise SystemExit(f"no records in {raw_path}")
    df = df.drop_duplicates(["id", "config", "perm"], keep="last")
    expected = sum(n_perms if c in perm_configs else 1 for c in configs)
    counts = df.groupby("id").size()
    complete = counts[counts == expected].index
    dropped = len(counts) - len(complete)
    df = df[df.id.isin(complete)].copy()
    full_tok = df[(df.config == "full") & (df.perm == 0)].set_index("id")["n_total_tok"]
    df["cost"] = df["n_total_tok"] / df["id"].map(full_tok)
    df["act_mem_mb"] = df["peak_mem_mb"] - df["baseline_mem_mb"]
    return df, dropped


def acc_table(df):
    """Accuracy by config x audio type, averaged over option permutations.

    CI: normal approximation over examples on the per-example mean correctness (examples are the
    independent unit; permutations of the same example are not).
    """
    rows = []
    for c in CONFIG_ORDER:
        sub = df[df.config == c]
        for at in AUDIO_TYPES + ["all"]:
            s = sub if at == "all" else sub[sub.audio_type == at]
            per_ex = s.groupby("id").correct.mean()
            n = len(per_ex)
            acc = per_ex.mean() if n else np.nan
            half = 1.96 * per_ex.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            p0 = s[s.perm == 0]
            rows.append({"config": c, "audio_type": at, "n": n, "acc": acc,
                         "ci_lo": max(acc - half, 0), "ci_hi": min(acc + half, 1),
                         "acc_perm0": p0.correct.mean(), "correct_perm0": int(p0.correct.sum()),
                         "acc_all_perms": per_ex.eq(1).mean()})
    return pd.DataFrame(rows)


def plot_acc_bars(t, out):
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(len(CONFIG_ORDER))
    w = 0.26
    for k, at in enumerate(AUDIO_TYPES):
        s = t[t.audio_type == at].set_index("config").loc[CONFIG_ORDER]
        pos = x + (k - 1) * w
        ax.bar(pos, s.acc * 100, width=w - 0.03, color=AUDIO_TYPE_COLORS[at], label=at)
        ax.errorbar(pos, s.acc * 100, yerr=[(s.acc - s.ci_lo) * 100, (s.ci_hi - s.acc) * 100],
                    fmt="none", ecolor=INK_2, elinewidth=0.5, capsize=0)
    allc = t[t.audio_type == "all"].set_index("config").loc[CONFIG_ORDER]
    ax.scatter(x, allc.acc * 100, marker="_", s=120, color=INK, linewidths=1.4, zorder=4, label="all")
    ax.axhline(25, color=INK_2, ls="--", lw=0.7)
    ax.text(len(x) - 0.45, 25, "chance", ha="left", va="center", fontsize=5.5, color=INK_2, clip_on=False)
    ax.set_xticks(x, [SHORT[c] for c in CONFIG_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1, handlelength=1.2)
    save_fig(fig, out, "acc_by_config_audio")


def pareto(points):
    """points: list of (cost, acc, name); frontier = not dominated (cheaper-or-equal & better-or-equal)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1
    for p in pts:
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


def place_labels(ax, items, fontsize=5.5):
    """Label points without overlaps: try right, left, below, above; keep the first free spot.

    items: (x, y, text, color). Call after the axis limits are final.
    """
    fig = ax.figure
    candidates = [((4, 2), "left", "bottom"), ((-4, 2), "right", "bottom"), ((0, -5), "center", "top"),
                  ((0, 5), "center", "bottom"), ((4, -3), "left", "top"), ((-4, -3), "right", "top")]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(r)
    markers = []
    leg = ax.get_legend()
    if leg is not None:
        lb = leg.get_window_extent(r)
        markers.append((lb.x0, lb.y0, lb.x1, lb.y1))
    for x, y, _, _ in items:
        px, py = ax.transData.transform((x, y))
        markers.append((px - 4, py - 4, px + 4, py + 4))
    labels = []

    def overlaps(a, b, pad=4):
        return not (a[2] + pad < b[0] or a[0] - pad > b[2] or a[3] + pad < b[1] or a[1] - pad > b[3])

    for k, (x, y, text, color) in enumerate(items):
        off_k = len(markers) - len(items)  # legend box (if any) comes first
        others = [m for j, m in enumerate(markers) if j != k + off_k] + labels
        for n, (off, ha, va) in enumerate(candidates):
            t = ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=fontsize,
                            ha=ha, va=va, color=color)
            bb = t.get_window_extent(r)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            inside = bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1
            if (inside and not any(overlaps(box, o) for o in others)) or n == len(candidates) - 1:
                labels.append(box)
                break
            t.remove()


def plot_acc_vs_tokens(summ, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    pts = [(r.mean_tokens, r.acc * 100, c) for c, r in summ.iterrows()]
    front = pareto(pts)
    fx, fy = zip(*[(p[0], p[1]) for p in front])
    ax.step(fx, fy, where="post", color=INK_2, lw=0.8, ls="--", zorder=1, label="Pareto frontier")
    fnames = {p[2] for p in front}
    for x, yv, c in pts:
        ax.scatter(x, yv, s=36 if c in fnames else 22, color=CONFIG_COLORS[c], edgecolors="white",
                   linewidths=0.6, zorder=3)
    ax.axhline(25, color=INK_2, ls=":", lw=0.6)
    ax.set_xlabel("Mean input tokens per example")
    ax.set_ylabel("Accuracy (%)")
    ys = [p[1] for p in pts]
    ax.set_xlim(0, max(p[0] for p in pts) * 1.3)
    ax.set_ylim(min(min(ys), 25) - 3, max(ys) + 3)
    ax.legend(loc="upper left")
    place_labels(ax, [(x, yv, CONFIG_LABELS[c], INK if c in fnames else INK_2) for x, yv, c in pts])
    save_fig(fig, out, "acc_vs_tokens")
    return sorted(fnames, key=lambda c: summ.loc[c, "mean_tokens"])


def oracle(df, robust=True):
    """Cheapest config that is correct per example; robust=True requires correctness under all perms."""
    p0 = df[df.perm == 0]
    C = (df.groupby(["id", "config"]).correct.all() if robust else p0.set_index(["id", "config"]).correct).unstack()
    K = p0.pivot(index="id", columns="config", values="cost")
    rows = []
    for i in C.index:
        ok = [c for c in CONFIG_ORDER if C.loc[i, c]]
        if ok:
            c = min(ok, key=lambda c: K.loc[i, c])
            rows.append({"id": i, "oracle_config": c, "oracle_cost": K.loc[i, c], "oracle_correct": True})
        else:
            # Nothing is correct, so an oracle spends the minimum (the cheapest config).
            rows.append({"id": i, "oracle_config": "none", "oracle_cost": K.loc[i].min(), "oracle_correct": False})
    return pd.DataFrame(rows).set_index("id")


def plot_oracle(orc, audio_type, out):
    order = CONFIG_ORDER + ["none"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for at in AUDIO_TYPES:
        cnt = np.array([((orc.oracle_config == c) & (audio_type == at)).sum() for c in order])
        ax.bar(x, cnt, bottom=bottom, color=AUDIO_TYPE_COLORS[at], width=0.7, label=at,
               edgecolor="white", linewidth=0.5)
        bottom += cnt
    for xi, b in zip(x, bottom):
        ax.text(xi, b + 0.3, f"{int(b)}", ha="center", va="bottom", fontsize=5.5, color=INK_2)
    ax.set_xticks(x, [SHORT.get(c, "None correct") for c in order], rotation=25, ha="right")
    ax.set_ylabel("Examples")
    ax.set_title("Cheapest configuration correct under all option orders")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save_fig(fig, out, "oracle_routing")


def utility_sweep(summ, lam_max, out):
    lams = np.round(np.arange(0, lam_max + 1e-9, 0.01), 3)
    U = {c: summ.loc[c, "acc"] - lams * summ.loc[c, "mean_cost"] for c in CONFIG_ORDER}
    best = [max(CONFIG_ORDER, key=lambda c: U[c][k]) for k in range(len(lams))]
    segs, start = [], 0
    for k in range(1, len(lams) + 1):
        if k == len(lams) or best[k] != best[start]:
            segs.append({"config": best[start], "lambda_from": lams[start], "lambda_to": lams[k - 1]})
            start = k
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for c in CONFIG_ORDER:
        on_top = c in {s["config"] for s in segs}
        ax.plot(lams, U[c] * 100, color=CONFIG_COLORS[c], lw=1.4 if on_top else 0.8,
                alpha=1 if on_top else 0.45, label=CONFIG_LABELS[c])
    ymax = max(U[c].max() for c in CONFIG_ORDER) * 100
    for s in segs:
        ax.axvspan(s["lambda_from"], s["lambda_to"] + 0.01, ymin=0.95, ymax=1.0,
                   color=CONFIG_COLORS[s["config"]], lw=0)
    ax.set_xlabel(r"Cost weight $\lambda$ (cost = tokens / full-input tokens)")
    ax.set_ylabel(r"$U$ = acc $-\ \lambda\cdot$cost (pts)")
    ax.set_xlim(0, lam_max)
    ax.set_ylim(top=ymax + 8)
    ax.legend(fontsize=5.2, ncol=2, loc="lower left", handlelength=1.2, columnspacing=0.8)
    save_fig(fig, out, "utility_sweep")
    return pd.DataFrame(segs)


def cascade(df, summ, orc, acc_full, out):
    """cap_trans first; escalate to full when max prob < tau. Accuracy averaged over option permutations."""
    taus = np.round(np.concatenate([[0.0], np.arange(0.25, 1.0001, 0.005), [1.01]]), 4)
    rows = []
    perms = sorted(df.perm.unique())
    for t in taus:
        accs, costs, escs = [], [], []
        for p in perms:
            cheap = df[(df.config == "cap_trans") & (df.perm == p)].set_index("id")
            full = df[(df.config == "full") & (df.perm == p)].set_index("id").loc[cheap.index]
            esc = cheap.max_prob < t
            accs.append(np.where(esc, full.correct, cheap.correct).mean())
            costs.append((cheap.cost + esc * full.cost).mean())  # the cheap pass is already paid
            escs.append(esc.mean())
        rows.append({"tau": t, "escalation_rate": np.mean(escs), "acc": np.mean(accs), "mean_cost": np.mean(costs)})
    cas = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.plot(cas.mean_cost, cas.acc * 100, color=INK, lw=1.2, label=r"Cascade (sweep $\tau$)", zorder=2)
    for c in CONFIG_ORDER:
        ax.scatter(summ.loc[c, "mean_cost"], summ.loc[c, "acc"] * 100, s=22, color=CONFIG_COLORS[c],
                   edgecolors="white", linewidths=0.6, zorder=3, label=CONFIG_LABELS[c])
    ax.scatter(orc.oracle_cost.mean(), orc.oracle_correct.mean() * 100, marker="*", s=70, color=INK,
               edgecolors="white", linewidths=0.5, zorder=4, label="Oracle router")
    ax.axhline(acc_full * 100, color=INK_2, ls=":", lw=0.6)
    ax.set_xlabel("Mean cost (fraction of full-input tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=5.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), handlelength=1.2,
              columnspacing=0.8)
    save_fig(fig, out, "cascade")
    return cas


def position_bias(df):
    """Gold answer position vs. the model's predicted letter (both in displayed order)."""
    d = df.assign(pred_letter=df.probs_display.map(lambda p: "ABCD"[int(np.argmax(p))]),
                  gold_letter=df.apply(lambda r: "ABCD"[r.perm_order.index(r.answer_idx)], axis=1))
    pred = pd.crosstab(d.config, d.pred_letter, normalize="index").reindex(CONFIG_ORDER)
    g0 = d[(d.config == "full") & (d.perm == 0)]
    gold = pd.crosstab(g0.audio_type, g0.gold_letter, normalize="index").reindex(AUDIO_TYPES)
    gold.loc["all"] = g0.gold_letter.value_counts(normalize=True)
    baselines = {f"always_{L}": float((g0.gold_letter == L).mean()) for L in "ABCD"}
    return pred, gold.fillna(0), baselines


def plot_position_bias(gold, pred, out):
    """Where the correct answer sits (by audio type) vs. which letter the model picks."""
    rows = [("Gold: speech", gold.loc["speech"]), ("Gold: sound event", gold.loc["sound event"]),
            ("Gold: music", gold.loc["music"]), ("Model: Image+Audio", pred.loc["full"]),
            ("Model: Question only", pred.loc["q_only"])]
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    colors = {"A": PALETTE[0], "B": PALETTE[1], "C": PALETTE[2], "D": PALETTE[3]}
    for r, (name, v) in enumerate(rows):
        left = 0
        for L in "ABCD":
            w = float(v.get(L, 0)) * 100
            ax.barh(r, w, left=left, color=colors[L], height=0.65, edgecolor="white", linewidth=0.6,
                    label=L if r == 0 else None)
            if w >= 9:
                ax.text(left + w / 2, r, f"{w:.0f}", ha="center", va="center", fontsize=5.2, color="white")
            left += w
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.invert_yaxis()
    ax.axhline(2.5, color=INK_2, lw=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of items (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1, title=None)
    save_fig(fig, out, "position_bias")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    acfg = cfg["analysis_a"]
    set_plot_style()
    out = repo_path("results/analysis_a")
    (out / "tables").mkdir(parents=True, exist_ok=True)

    df, dropped = load(repo_path(args.raw or acfg["out"]), acfg["configs"], acfg["perm_configs"], acfg["n_perms"])
    p0 = df[df.perm == 0].copy()
    n_ex = p0.id.nunique()
    print(f"{n_ex} complete examples ({dropped} incomplete skipped); "
          f"audio types {p0[p0.config == 'full'].audio_type.value_counts().to_dict()}")

    # 1. accuracy by config x audio type (mean over permutations)
    t = acc_table(df)
    t.to_csv(out / "tables/acc_by_config_audio.csv", index=False)
    cols = AUDIO_TYPES + ["all"]
    wide = t.pivot(index="config", columns="audio_type", values="acc").loc[CONFIG_ORDER, cols]
    wide0 = t.pivot(index="config", columns="audio_type", values="acc_perm0").loc[CONFIG_ORDER, cols]
    disp = (wide * 100).round(1).astype(str) + " [" + (wide0 * 100).round(1).astype(str) + "]"
    disp.to_csv(out / "tables/acc_by_config_audio_wide.csv")
    (wide * 100).round(1).rename(index=CONFIG_LABELS).to_latex(out / "tables/acc_by_config_audio.tex")
    print(f"\nAccuracy %, mean over {df.perm.nunique()} option orders [original order in brackets], n=30 per type:")
    print(disp.rename(index=CONFIG_LABELS).to_string())
    plot_acc_bars(t, out)

    # answer-position bias
    pred_bias, gold_pos, pos_baselines = position_bias(df)
    pred_bias.to_csv(out / "tables/model_letter_preference.csv")
    gold_pos.to_csv(out / "tables/gold_answer_position.csv")
    print("\nGold answer position (original order):\n" + (gold_pos * 100).round(0).to_string())
    print("Model predicted letter share (all orders):\n" + (pred_bias * 100).round(0).rename(index=CONFIG_LABELS).to_string())
    print("Position baselines (original order):", {k: round(v, 3) for k, v in pos_baselines.items()})
    plot_position_bias(gold_pos, pred_bias, out)

    # per-config cost summary; accuracy = mean over permutations, costs identical across permutations
    summ = df.groupby("config").agg(
        acc=("correct", "mean"), mean_tokens=("n_total_tok", "mean"), mean_cost=("cost", "mean"),
        img_tok=("n_image_tok", "mean"), aud_tok=("n_audio_tok", "mean"), text_tok=("n_text_tok", "mean"),
        latency_ms=("latency_ms", "mean"), latency_p90_ms=("latency_ms", lambda x: x.quantile(0.9)),
        act_mem_mb=("act_mem_mb", "mean"), peak_mem_mb=("peak_mem_mb", "max"), mean_max_prob=("max_prob", "mean"),
    ).loc[CONFIG_ORDER]
    summ.insert(1, "acc_perm0", p0.groupby("config").correct.mean().loc[CONFIG_ORDER])
    summ.to_csv(out / "tables/config_costs.csv")
    print("\nPer-config accuracy (mean over orders) and cost:")
    print(summ.rename(index=CONFIG_LABELS).round(3).to_string())

    # 2. accuracy vs tokens + Pareto
    frontier = plot_acc_vs_tokens(summ, out)

    # permutation robustness
    perm_rows = []
    for c in acfg["perm_configs"]:
        s = df[df.config == c]
        pv = s.pivot(index="id", columns="perm", values="pred_idx")
        cv = s.pivot(index="id", columns="perm", values="correct")
        perm_rows.append({"config": c, **{f"acc_perm{p}": cv[p].mean() for p in cv.columns},
                          "acc_all_perms_correct": cv.all(axis=1).mean(),
                          "pred_consistent_across_perms": (pv.nunique(axis=1) == 1).mean()})
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(out / "tables/permutation_robustness.csv", index=False)
    print("\nOption-permutation robustness:")
    print(perm_df.round(3).to_string(index=False))

    # 3. oracle routing
    orc = oracle(df, robust=True)
    orc0 = oracle(df, robust=False)
    at_by_id = p0[p0.config == "full"].set_index("id").audio_type.loc[orc.index]
    orc.join(at_by_id).to_csv(out / "tables/oracle_routing.csv")
    plot_oracle(orc, at_by_id, out)
    acc_full = summ.loc["full", "acc"]
    print(f"\nOracle router (correct under all orders): acc {orc.oracle_correct.mean():.3f} at mean cost "
          f"{orc.oracle_cost.mean():.3f} vs full acc {acc_full:.3f} at cost 1.000")
    print(f"Oracle router (original order only, for comparison): acc {orc0.oracle_correct.mean():.3f} "
          f"at mean cost {orc0.oracle_cost.mean():.3f}")
    print("cheapest-correct config counts:", orc.oracle_config.value_counts().to_dict())

    # 4. text-sufficient labels
    ct = df[df.config == "cap_trans"].pivot(index="id", columns="perm", values="correct")
    lab = ct.all(axis=1).astype(int)
    lab_path = repo_path("results/text_sufficient_labels.json")
    lab_path.write_text(json.dumps({
        "definition": "caption+transcript config correct under all option permutations",
        "n_perms": int(ct.shape[1]), "n": int(len(lab)), "positive_rate": float(lab.mean()),
        "by_audio_type": {at: float(lab[at_by_id.loc[lab.index] == at].mean()) for at in AUDIO_TYPES},
        "labels": {str(i): int(v) for i, v in lab.items()},
    }, indent=2))
    print(f"\ntext-sufficient: {lab.sum()}/{len(lab)} ({lab.mean():.2f}) -> {lab_path}")

    # 5. utility sweep
    segs = utility_sweep(summ, args.lambda_max, out)
    segs.to_csv(out / "tables/utility_sweep_optimal.csv", index=False)
    print("\nOptimal config by lambda:")
    print(segs.to_string(index=False))

    # 6. cascade
    cas = cascade(df, summ, orc, acc_full, out)
    cas.to_csv(out / "tables/cascade.csv", index=False)
    ok = cas[cas.acc >= acc_full]
    best_match = ok.loc[ok.mean_cost.idxmin()] if not ok.empty else None
    best_acc = cas.loc[cas.acc.idxmax()]
    print(f"\nCascade: best acc {best_acc.acc:.3f} at cost {best_acc.mean_cost:.3f} (tau={best_acc.tau:.3f})")
    if best_match is not None:
        print(f"Cascade matching full accuracy: cost {best_match.mean_cost:.3f}, tau={best_match.tau:.3f}, "
              f"escalates {best_match.escalation_rate:.0%}")

    # modality costs for Analysis B (means and per example)
    by = p0.set_index(["id", "config"])
    per_ex = {}
    for i in orc.index:
        per_ex[str(i)] = {
            "base_text": int(by.loc[(i, "q_only"), "n_total_tok"]),
            "caption": int(by.loc[(i, "cap_aud"), "n_surrogate_tok"]),
            "transcript": int(by.loc[(i, "img_trans"), "n_surrogate_tok"]),
            "image": int(by.loc[(i, "img_only"), "n_image_tok"]),
            "audio": int(by.loc[(i, "aud_only"), "n_audio_tok"]),
        }
    means = {k: float(np.mean([v[k] for v in per_ex.values()])) for k in next(iter(per_ex.values()))}
    (out / "modality_costs.json").write_text(json.dumps({"mean": means, "per_example": per_ex}, indent=2))
    print("\nmean modality token costs:", {k: round(v, 1) for k, v in means.items()})

    summary = {
        "n_examples": int(n_ex), "device": df.device.iloc[0], "dtype": df.dtype.iloc[0],
        "acc_mean_over_perms": summ.acc.round(4).to_dict(), "acc_perm0": summ.acc_perm0.round(4).to_dict(),
        "position_baselines": pos_baselines, "mean_tokens": summ.mean_tokens.round(1).to_dict(),
        "mean_cost": summ.mean_cost.round(4).to_dict(), "latency_ms": summ.latency_ms.round(1).to_dict(),
        "pareto_frontier": frontier,
        "oracle_robust": {"acc": float(orc.oracle_correct.mean()), "mean_cost": float(orc.oracle_cost.mean()),
                          "counts": orc.oracle_config.value_counts().to_dict()},
        "oracle_perm0": {"acc": float(orc0.oracle_correct.mean()), "mean_cost": float(orc0.oracle_cost.mean())},
        "text_sufficient_rate": float(lab.mean()),
        "permutation": perm_df.round(4).to_dict(orient="records"),
        "cascade_match_full": None if best_match is None else best_match.round(4).to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote figures, tables and summary.json to {out}")


if __name__ == "__main__":
    main()
