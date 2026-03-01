# prep_stan_media_data.py
#
# Reads media_monthly_complete.xlsx (to Aug-2025),
# builds Stan "Better" measurement-model inputs:
#   - t_scaled in [-1,1]
#   - observed intensity indices/values
#   - observed count indices/values (optional but recommended)
# Writes: data/stan_media_data.json
#
# Usage:
#   python prep_stan_media_data.py --xlsx media_monthly_complete.xlsx --out data/stan_media_data.json

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd


REQ_COLS = [
    "year_month",
    "t_scaled",
    "media_damage_intensity",
    "media_event_count",
    "intensity_observed",
    "event_count_observed",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/media_monthly_complete.xlsx",
                    help="Path to media_monthly_complete.xlsx (default: data/media_monthly_complete.xlsx)")
    ap.add_argument("--sheet", default=None, help="Optional sheet name (default: first sheet)")
    ap.add_argument("--out", default="data/stan_media_data.json",
                    help="Output JSON path (default: data/stan_media_data.json)")
    ap.add_argument("--include_counts", action="store_true", help="Include event count thinning inputs")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet if args.sheet is not None else 0, engine="openpyxl")

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XLSX: {missing}\nFound: {list(df.columns)}")

    # Sort by time (year_month assumed sortable as YYYY-MM)
    df = df.copy()
    df["year_month"] = df["year_month"].astype(str).str.strip()
    df = df.sort_values("year_month").reset_index(drop=True)

    T = len(df)
    if T < 12:
        raise ValueError(f"Unexpectedly small T={T}; check input file")

    # t_scaled: if present, use it; otherwise construct
    t_scaled = pd.to_numeric(df["t_scaled"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(t_scaled)):
        # construct from row index if needed
        t = np.arange(1, T + 1)
        t_scaled = -1.0 + 2.0 * (t - 1) / (T - 1)

    # Observed intensity
    intensity_mask = pd.to_numeric(df["intensity_observed"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    y_all = pd.to_numeric(df["media_damage_intensity"], errors="coerce").to_numpy(dtype=float)

    y_idx_0 = np.where(intensity_mask)[0]
    y_obs = y_all[intensity_mask]

    if len(y_idx_0) == 0:
        raise ValueError("No observed media_damage_intensity rows found (intensity_observed==1)")

    if np.any(y_obs < 0):
        bad = y_obs[y_obs < 0]
        raise ValueError(f"Negative media_damage_intensity values found: {bad.tolist()}")

    # Stan uses 1-based indexing
    y_idx = (y_idx_0 + 1).astype(int).tolist()

    data = {
        "T": int(T),
        "eps": 1e-6,
        "t_scaled": t_scaled.astype(float).tolist(),
        "N_y": int(len(y_idx)),
        "y_idx": y_idx,
        "y_obs": y_obs.astype(float).tolist(),
    }

    # Optional counts
    if args.include_counts:
        count_mask = pd.to_numeric(df["event_count_observed"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        n_all = pd.to_numeric(df["media_event_count"], errors="coerce").to_numpy(dtype=float)

        n_idx_0 = np.where(count_mask)[0]
        n_obs = n_all[count_mask]

        if len(n_idx_0) == 0:
            raise ValueError("include_counts set but no observed media_event_count rows found (event_count_observed==1)")

        # Coerce to non-negative ints safely
        n_obs_int = np.maximum(0, np.round(n_obs)).astype(int)

        data.update({
            "N_n": int(len(n_idx_0)),
            "n_idx": (n_idx_0 + 1).astype(int).tolist(),
            "n_obs": n_obs_int.tolist(),
        })

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote: {args.out}")
    print("\nSanity:")
    print(f"  T={data['T']}")
    print(f"  N_y={data['N_y']}")
    if args.include_counts:
        print(f"  N_n={data['N_n']}")
    print(f"  First month: {df['year_month'].iloc[0]}")
    print(f"  Last month:  {df['year_month'].iloc[-1]}")
    print(f"  t_scaled range: [{min(data['t_scaled']):.3f}, {max(data['t_scaled']):.3f}]")


if __name__ == "__main__":
    main()