import streamlit as st
import pandas as pd, numpy as np, itertools
from sklearn.metrics import cohen_kappa_score
from pathlib import Path

st.set_page_config(layout="wide", page_title="Rater Analytics Demo")

# --- Robust paths ---
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_DATA = BASE_DIR / "data" / "sim_rater_dataset.csv"

@st.cache_data
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    # defensive typing
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    req = {"item_id", "annotator_id", "label", "category", "timestamp"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing required columns: {miss}")
    return df

def krippendorff_alpha_nominal(A: np.ndarray) -> float:
    vals = A[~np.isnan(A)]
    if vals.size == 0:
        return np.nan
    cats = np.unique(vals)
    idx = {c: i for i, c in enumerate(cats)}
    C = np.zeros((len(cats), len(cats)), dtype=float)
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
    De = C.sum()**2 - (marg**2).sum()
    if De <= 0:
        return np.nan
    return 1.0 - (Do / De)

def fleiss_kappa_from_counts(counts: pd.DataFrame) -> float:
    N = counts.shape[0]
    if N == 0:
        return np.nan
    n = int(pd.Series(counts.sum(axis=1)).mode().iloc[0])  # modal n
    scaled = counts.div(counts.sum(axis=1), axis=0).mul(n)
    p_j = scaled.sum(axis=0).values / (N * n)
    P_i = ((scaled**2).sum(axis=1) - n) / (n * (n - 1) + 1e-12)
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    return (P_bar - P_e) / (1 - P_e + 1e-12)

# --- Load data or allow upload fallback ---
try:
    df = load_df(DEFAULT_DATA)
except Exception as e:
    st.error(f"Could not load default data: {e}")
    st.info("Upload a CSV with columns: item_id, annotator_id, label, category, timestamp")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        st.stop()

# --- Top KPIs ---
st.title("Rater Analytics — Demo")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Items", int(df["item_id"].nunique()))
    st.metric("Ratings", int(len(df)))
with c2:
    st.metric("Raters", int(df["annotator_id"].nunique()))
    st.metric("Categories", int(df["category"].nunique()))
with c3:
    start, end = df["timestamp"].min(), df["timestamp"].max()
    st.metric("Start", str(start.date()) if pd.notna(start) else "N/A")
    st.metric("End", str(end.date()) if pd.notna(end) else "N/A")

# --- Agreement summary ---
st.header("Agreement summary")
M = df.pivot_table(index="item_id", columns="annotator_id", values="label", aggfunc="first")
raters = list(M.columns)

pair_rows = []
for a, b in itertools.combinations(raters, 2):
    sub = M[[a, b]].dropna()
    k = np.nan if sub.empty else cohen_kappa_score(sub[a], sub[b])
    pair_rows.append({"a": a, "b": b, "k": k, "n": len(sub)})
pairs_df = pd.DataFrame(pair_rows)

mean_pairwise = float(pairs_df["k"].mean()) if not pairs_df.empty else float("nan")
st.write("Mean pairwise Cohen's κ:", round(mean_pairwise, 3) if pd.notna(mean_pairwise) else "N/A")

counts = df.groupby(["item_id", "label"]).size().unstack(fill_value=0)
fleiss = fleiss_kappa_from_counts(counts) if not counts.empty else np.nan
st.write("Fleiss' κ:", round(float(fleiss), 3) if pd.notna(fleiss) else "N/A")

alpha = krippendorff_alpha_nominal(M.values) if M.size else np.nan
st.write("Krippendorff's α:", round(float(alpha), 3) if pd.notna(alpha) else "N/A")

st.subheader("Pairwise κ (lowest 10)")
if not pairs_df.empty:
    st.dataframe(pairs_df.sort_values("k").head(10))
else:
    st.info("Not enough overlapping labels to compute pairwise kappas.")

# --- Bias slices ---
st.header("Bias slices")
slice_rates = df.groupby("category")["label"].apply(lambda s: (s == 1).mean()).sort_values(ascending=False)
if not slice_rates.empty:
    st.bar_chart(slice_rates)
else:
    st.info("No categories found.")

st.subheader("Category × Rater rates")
cat_rater = df.pivot_table(index="category", columns="annotator_id", values="label", aggfunc="mean")
if not cat_rater.empty:
    st.dataframe(cat_rater.round(3))
else:
    st.info("Not enough data to show category × rater rates.")

# --- Throughput ---
st.header("Throughput (labels/hour)")
if df["timestamp"].notna().any():
    span_hours = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600.0
    throughput = df.groupby("annotator_id").size() / max(span_hours, 1e-9)
    st.bar_chart(throughput)
else:
    st.info("Timestamps missing — throughput disabled.")

# --- Simple guardrail alert ---
guard = st.number_input("Agreement guardrail (κ/α)", value=0.60, step=0.05)
if pd.notna(mean_pairwise) and mean_pairwise < guard:
    st.warning("Agreement below guardrail — investigate slices/guidelines & retrain.")
