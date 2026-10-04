"""Print a readable per-row table and summary of an Analysis A JSONL file.

    python -m src.analysis_a.inspect_results results/analysis_a/smoke.jsonl
"""
import argparse

import pandas as pd

from src.common import read_jsonl, repo_path
from src.data import LETTERS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--rows", action="store_true", help="print every row")
    args = ap.parse_args()

    df = pd.DataFrame(read_jsonl(repo_path(args.path)))
    if df.empty:
        print("no records")
        return
    df["gold"] = df["answer_idx"].map(lambda i: LETTERS[i])
    df["pred"] = df["pred_idx"].map(lambda i: LETTERS[i])
    df["P(A,B,C,D)"] = df["probs"].map(lambda p: " ".join(f"{x:.2f}" for x in p))
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)

    if args.rows:
        cols = ["id", "audio_type", "config", "perm", "gold", "pred", "correct", "P(A,B,C,D)",
                "n_text_tok", "n_image_tok", "n_audio_tok", "n_total_tok", "latency_ms", "peak_mem_mb"]
        if "generated" in df:
            cols.append("generated")
        print(df[cols].round(0).to_string(index=False))

    print("\nPer config (all perms pooled):")
    summ = df.groupby("config").agg(
        n=("correct", "size"), acc=("correct", "mean"), max_prob=("max_prob", "mean"),
        tok=("n_total_tok", "mean"), img_tok=("n_image_tok", "mean"), aud_tok=("n_audio_tok", "mean"),
        lat_ms=("latency_ms", "mean"), lat_p90=("latency_ms", lambda x: x.quantile(0.9)),
        peak_mb=("peak_mem_mb", "max"), preproc_ms=("preproc_ms", "mean"),
    )
    print(summ.round(2).to_string())

    print(f"\ntop vocab token is a letter A-D: {df['top_vocab_is_letter'].mean():.1%} of rows")
    if "generated" in df:
        gen_letter = df["generated"].str.strip().str[:1]
        disp_pred = df.apply(lambda r: LETTERS[max(range(4), key=lambda j: r["probs_display"][j])], axis=1)
        print(f"greedy generation starts with argmax letter: {(gen_letter == disp_pred).mean():.1%} of rows")
    print(f"total forward time: {df['latency_ms'].sum() / 1000:.1f}s over {len(df)} passes "
          f"({df['latency_ms'].mean() / 1000:.2f}s/pass, +{df['preproc_ms'].mean() / 1000:.2f}s preprocessing)")


if __name__ == "__main__":
    main()
