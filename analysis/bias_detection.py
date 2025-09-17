#!/usr/bin/env python3
"""
Bias detection analysis script

Run:
  python analysis/bias_detection.py --input data/sim_rater_dataset.csv
"""

import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def krippendorff_alpha_nominal_items_by_raters(A: np.ndarray) -> float:
    """
    Nominal-scale Krippendorff's alpha for items x raters matrix with NaN for missing.
    Builds a coincidence matrix per item (correct approach). Returns np.nan if undefined.
    """
    vals = A[~np.isnan(A)]
    if vals.size == 0:
        return np.nan

    cats = np.unique(vals)
    idx = {c: i for i, c in enumerate(cats)}
    C = np.zeros((len(cats), len(cats)), dtype=float)

    # build coincidence counts per item (row)
    for row in A:
        row_vals = row[~np.isnan(row)]
        m = len(row_vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                C[idx[row_vals[i]], idx[row_vals[j]]] += 1
                C[idx[row_vals[j]], idx[row_vals[i]]] += 1

    Do = C.sum() - np.trace(C)
    marg = C.sum(axis=0)
    De = C.sum() ** 2 - (marg ** 2).sum()
    if De <= 0:
        return np.nan
    return 1.0 - (Do / De)


def fleiss_kappa_from_counts(counts: pd.DataFrame) -> float:
    """
    Fleiss' kappa from counts table with index=item_id, columns=labels (counts per item).
    Handles variable numbers of ratings by scaling rows to modal n.
    """
    N = counts.shape[0]
    if N == 0:
        return np.nan
    n_per_item = counts.sum(axis=1).values
    n = int(pd.Series(n_per_item).mode().iloc[0])  # modal n

    scaled = counts.div(counts.sum(axis=1), axis=0).mul(n)
    p_j = scaled.sum(axis=0).values / (N * n)
    P_i = ((scaled**2).sum(axis=1) - n) / (n * (n - 1) + 1e-12)
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    return (P_bar - P_e) / (1 - P_e + 1e-12)


def main(path: str):
    df = pd.read_csv(path)
    print(f"Rows: {len(df)} | Items: {df['item_id'].nunique()} | Raters: {df['annotator_id'].nunique()}")
    print(df.head().to_string(index=False))

    # pivot items x raters for agreement metrics
    M = df.pivot_table(index="item_id", columns="annotator_id", values="label", aggfunc="first")
    raters = list(M.columns)

    # pairwise Cohen's kappa across all rater pairs
    pair_rows = []
    for a, b in itertools.combinations(raters, 2):
        sub = M[[a, b]].dropna()
        k = np.nan if sub.empty else cohen_kappa_score(sub[a], sub[b])
        pair_rows.append({"rater_a": a, "rater_b": b, "kappa": float(k) if pd.notna(k) else np.nan, "n": len(sub)})
    pairs_df = pd.DataFrame(pair_rows)
    print("\nPairwise Cohen's kappa (sorted lowest->highest):")
    print(pairs_df.sort_values("kappa").to_string(index=False))
    print(f"\nMean pairwise kappa = {pairs_df['kappa'].mean():.3f}\n")

    # Fleiss' kappa (multi-rater)
    counts = df.groupby(["item_id", "label"]).size().unstack(fill_value=0)
    fleiss = fleiss_kappa_from_counts(counts)
    print(f"Fleiss' kappa = {fleiss:.3f}")

    # Krippendorff's alpha (nominal, handles missing)
    alpha = krippendorff_alpha_nominal_items_by_raters(M.values)
    print(f"Krippendorff's alpha = {alpha:.3f}\n")

    # per-rater strictness/leniency
    rater_rates = df.groupby("annotator_id")["label"].mean().sort_values()
    print("Per-rater label=1 rates (leniency):")
    print(rater_rates.to_string())

    # category × rater
    cat_rater = df.pivot_table(index="category", columns="annotator_id", values="label", aggfunc="mean")
    print("\nCategory × Rater label rates:")
    print(cat_rater.round(3).to_string())

    # slice label rates + chi-square
    slice_rates = df.groupby("category")["label"].apply(lambda s: (s == 1).mean()).sort_values(ascending=False)
    print("\nLabel rate by category:")
    print(slice_rates.to_string())

    ctab = pd.crosstab(df["category"], df["label"])
    if ctab.shape[0] > 1 and ctab.shape[1] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(ctab)
        print(f"\nChi-square: chi2={chi2:.2f}, dof={dof}, p={p:.4g}")
    else:
        print("Chi-square: insufficient table")

    # throughput (labels/hour per rater)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].notna().any():
        span_hours = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600.0
        throughput = df.groupby("annotator_id").size() / max(span_hours, 1e-9)
        print("\nThroughput (labels/hour per rater):")
        print(throughput.round(2).to_string())

    # short ops summary
    print("\n=== OPS SUMMARY ===")
    safe_mean = pairs_df["kappa"].mean()
    def lvl(x): return "LOW" if (pd.notna(x) and x < 0.60) else "OK"
    print(f"Agreement → mean κ={safe_mean:.3f} ({lvl(safe_mean)}), Fleiss={fleiss:.3f} ({lvl(fleiss)}), α={alpha:.3f} ({lvl(alpha)})")
    print("Actions → clarify guidelines for lowest slice; add targeted golds; coach weak-pair raters; watch throughput vs quality.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sim_rater_dataset.csv", help="Path to CSV")
    args = parser.parse_args()
    main(args.input)
