# build_media_monthly_complete.py
#
# Purpose:
#   Take your sparse monthly media sheet (monthly_media_variables.xlsx),
#   build a complete monthly grid for a specified window (e.g., 2020-10 to 2023-09),
#   left-join the observed media variables, and output a "complete" table with:
#     - explicit missingness (NaN where unobserved)
#     - observation masks (media_observed, intensity_observed, etc.)
#     - a scaled time index for modelling coverage (t_scaled in [-1, 1])
#
# Usage (Windows):
#   python build_media_monthly_complete.py ^
#       --xlsx monthly_media_variables.xlsx ^
#       --start 2020-10 ^
#       --end 2023-09 ^
#       --out data/media_monthly_complete.csv
#
# Notes:
#   - If a month is not present in the XLSX, we treat it as "unobserved" (NaN), not 0.
#   - If a month is present but a particular field is blank, that field's mask will be 0.
#
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Configuration: expected columns
# ----------------------------

# You said your XLSX has these columns:
EXPECTED_COLS = [
    "year",
    "month",
    "year_month",
    "media_event_count",
    "media_damage_intensity",
    "media_sentiment_mean",
    "media_response_quality",
    "media_concentration_index",
]

MEDIA_NUMERIC_COLS = [
    "media_event_count",
    "media_damage_intensity",
    "media_sentiment_mean",
    "media_response_quality",
    "media_concentration_index",
]


def _parse_yyyymm(s: str) -> pd.Period:
    """
    Parse 'YYYY-MM' into a pandas Period('M').
    """
    try:
        return pd.Period(s, freq="M")
    except Exception as e:
        raise ValueError(f"Invalid YYYY-MM: {s}") from e


def _month_range(start: pd.Period, end: pd.Period) -> List[pd.Period]:
    if end < start:
        raise ValueError("end must be >= start")
    return list(pd.period_range(start=start, end=end, freq="M"))


def load_media_xlsx(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"XLSX not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, engine="openpyxl")

    # Basic column check
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"XLSX missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Normalise year_month to Period('M')
    df = df.copy()
    df["year_month"] = df["year_month"].astype(str).str.strip()

    # Some files may have 'YYYY-MM' or 'YYYY-M' or 'YYYY/MM' - normalise
    df["year_month"] = (
        df["year_month"]
        .str.replace("/", "-", regex=False)
        .str.replace(" ", "", regex=False)
    )

    # If year_month is not reliable, rebuild from year+month
    # (we'll prefer year_month if it parses, otherwise fallback)
    def to_period(row) -> pd.Period:
        ym = row["year_month"]
        try:
            return pd.Period(ym, freq="M")
        except Exception:
            y = int(row["year"])
            m = int(row["month"])
            return pd.Period(f"{y:04d}-{m:02d}", freq="M")

    df["period"] = df.apply(to_period, axis=1)

    # Numeric coercion
    for c in MEDIA_NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Deduplicate by month if needed (keep the last row; you can change policy)
    df = df.sort_values("period").drop_duplicates(subset=["period"], keep="last")

    # Keep relevant columns
    keep = ["period"] + MEDIA_NUMERIC_COLS
    df = df[keep].reset_index(drop=True)

    return df


def build_complete_grid(start: pd.Period, end: pd.Period) -> pd.DataFrame:
    periods = _month_range(start, end)
    grid = pd.DataFrame({"period": periods})
    grid["year_month"] = grid["period"].astype(str)
    grid["year"] = grid["period"].dt.year.astype(int)
    grid["month"] = grid["period"].dt.month.astype(int)
    grid["t"] = np.arange(1, len(grid) + 1)

    # Scaled time index in [-1, 1] (useful for coverage trend)
    if len(grid) == 1:
        grid["t_scaled"] = 0.0
    else:
        grid["t_scaled"] = -1.0 + 2.0 * (grid["t"] - 1) / (len(grid) - 1)

    return grid


def merge_media(grid: pd.DataFrame, media: pd.DataFrame) -> pd.DataFrame:
    df = grid.merge(media, on="period", how="left", suffixes=("", "_obs"))

    # This mask means: "we have a row for that month in the XLSX"
    # We approximate this by whether at least one of the media cols is non-null.
    df["media_observed"] = df[MEDIA_NUMERIC_COLS].notna().any(axis=1).astype(int)

    # Field-level masks (useful when some fields are missing even in observed months)
    df["event_count_observed"] = df["media_event_count"].notna().astype(int)
    df["intensity_observed"] = df["media_damage_intensity"].notna().astype(int)
    df["sentiment_observed"] = df["media_sentiment_mean"].notna().astype(int)
    df["response_observed"] = df["media_response_quality"].notna().astype(int)
    df["concentration_observed"] = df["media_concentration_index"].notna().astype(int)

    # Optional: if event_count is missing but media_observed is 1, set it to 0
    # (Interpretation: observed month with explicitly no events recorded)
    # If you'd rather leave NaN, comment this out.
    df.loc[(df["media_observed"] == 1) & (df["media_event_count"].isna()), "media_event_count"] = 0.0
    df["event_count_observed"] = df["media_event_count"].notna().astype(int)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/monthly_media_variables.xlsx",
                    help="Path to monthly_media_variables.xlsx (default: data/monthly_media_variables.xlsx)")
    ap.add_argument("--sheet", default=None, help="Optional sheet name (default: first sheet)")
    ap.add_argument("--start", default="2020-10",
                    help="Start month YYYY-MM inclusive (default: 2020-10)")
    ap.add_argument("--end", default=None,
                    help="End month YYYY-MM inclusive (default: latest month in xlsx)")
    ap.add_argument("--out", default="data/media_monthly_complete.xlsx",
                    help="Output XLSX path (default: data/media_monthly_complete.xlsx)")
    ap.add_argument("--out_parquet", default=None, help="Optional output parquet path")
    args = ap.parse_args()

    media = load_media_xlsx(args.xlsx, sheet=args.sheet)

    start = _parse_yyyymm(args.start)
    if args.end:
        end = _parse_yyyymm(args.end)
    else:
        end = media["period"].max()
        print(f"Auto-detected --end from xlsx: {end}")

    grid = build_complete_grid(start, end)
    merged = merge_media(grid, media)

    # Sort and final column ordering
    ordered_cols = [
        "t",
        "t_scaled",
        "year_month",
        "year",
        "month",
        "media_observed",
        "event_count_observed",
        "intensity_observed",
        "sentiment_observed",
        "response_observed",
        "concentration_observed",
        "media_event_count",
        "media_damage_intensity",
        "media_sentiment_mean",
        "media_response_quality",
        "media_concentration_index",
    ]
    # Keep any extra columns if present
    extra = [c for c in merged.columns if c not in ordered_cols and c != "period"]
    merged = merged[ordered_cols + extra].copy()

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    merged.to_excel(args.out, index=False, engine="openpyxl")
    print(f"Wrote: {args.out} ({len(merged)} rows)")

    if args.out_parquet:
        outdir2 = os.path.dirname(args.out_parquet)
        if outdir2:
            os.makedirs(outdir2, exist_ok=True)
        merged.to_parquet(args.out_parquet, index=False)
        print(f"Wrote: {args.out_parquet} ({len(merged)} rows)")

    # Quick summary
    n_obs = int(merged["media_observed"].sum())
    n_missing = len(merged) - n_obs
    print("\nQuick summary:")
    print(f"  Months in grid:        {len(merged)}")
    print(f"  Observed months:       {n_obs}")
    print(f"  Unobserved months:     {n_missing}")
    print(f"  Intensity observed:    {int(merged['intensity_observed'].sum())}")
    print(f"  Event count observed:  {int(merged['event_count_observed'].sum())}")


if __name__ == "__main__":
    main()