#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

OUT = Path(__file__).resolve().parents[1] / "data" / "sim_rater_dataset.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

n_items = 600
raters = [f"r{i}" for i in range(1, 7)]
cats = ["sports", "finance", "health"]

items = [f"A{i}" for i in range(1, n_items + 1)]
item_cat = rng.choice(cats, size=n_items, p=[0.33, 0.33, 0.34])

rows = []
t0 = datetime(2025, 8, 25, 9, 0, 0)

for i, iid in enumerate(items):
    cat = item_cat[i]
    base = {"sports": 0.70, "health": 0.58, "finance": 0.42}[cat]
    rater_offset = {r: rng.normal(0, 0.05) for r in raters}

    for r in raters:
        p = np.clip(base + rater_offset[r], 0.02, 0.98)
        label = rng.random() < p
        # FIX: cast NumPy int to Python int
        ts = t0 + timedelta(minutes=i // 3, seconds=int(rng.integers(0, 50)))
        rows.append((iid, r, int(label), cat, ts.isoformat()))

df = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label", "category", "timestamp"])
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(df)} rows.")
