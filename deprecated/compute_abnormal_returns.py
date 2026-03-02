# compute_abnormal_returns.py
# Usage:
#   python compute_abnormal_returns.py
#     (defaults: data/qantas_share_price_data.json, 2020-10-01 to 2023-09-30, outdir=data)
#   python compute_abnormal_returns.py --start 2020-10-01 --end 2023-09-30
#   python compute_abnormal_returns.py --json path/to/data.json --outdir output
#
# Outputs:
#   data/abret_daily.csv
#   data/abret_monthly.csv
#
# Assumptions about JSON structure (matches your file):
#   top-level keys: "metadata", "statistics", "data"
#   each item in "data" has: date, close, index_close (ASX200), volume (optional)

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def _to_datetime(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=False).tz_localize(None)


def load_qantas_json(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "data" not in obj or not isinstance(obj["data"], list):
        raise ValueError("JSON missing top-level list key 'data'")

    df = pd.DataFrame(obj["data"])
    required = ["date", "close", "index_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"JSON data items missing required fields: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["index_close"] = pd.to_numeric(df["index_close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Drop rows with missing closes
    df = df.dropna(subset=["close", "index_close"]).reset_index(drop=True)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r_qan"] = np.log(out["close"]).diff()
    out["r_mkt"] = np.log(out["index_close"]).diff()
    out = out.dropna(subset=["r_qan", "r_mkt"]).reset_index(drop=True)
    return out


@dataclass
class MarketModelFit:
    alpha: float
    beta: float
    resid_sd: float
    n: int


def fit_market_model_ols(r_qan: np.ndarray, r_mkt: np.ndarray) -> MarketModelFit:
    """
    OLS: r_qan = alpha + beta * r_mkt + eps
    """
    if r_qan.ndim != 1 or r_mkt.ndim != 1:
        raise ValueError("r_qan and r_mkt must be 1D arrays")
    if len(r_qan) != len(r_mkt):
        raise ValueError("r_qan and r_mkt must have same length")
    if len(r_qan) < 30:
        raise ValueError("Need at least 30 observations to fit market model")

    x = r_mkt
    y = r_qan
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    x_var = float(np.var(x, ddof=1))
    if x_var <= 0:
        raise ValueError("Market return variance is zero; cannot fit beta")

    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    beta = cov_xy / x_var
    alpha = y_mean - beta * x_mean

    resid = y - (alpha + beta * x)
    resid_sd = float(np.std(resid, ddof=1))

    return MarketModelFit(alpha=alpha, beta=beta, resid_sd=resid_sd, n=len(y))


def compute_abnormal_returns(df_ret: pd.DataFrame, fit: MarketModelFit) -> pd.DataFrame:
    out = df_ret.copy()
    out["abret"] = out["r_qan"] - (fit.alpha + fit.beta * out["r_mkt"])
    return out


def aggregate_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly aggregation suitable for a monthly belief-state model.
    """
    out = df_daily.copy()
    out["month"] = out["date"].dt.to_period("M").astype(str)

    # Tail frequency threshold: 2 * daily residual sd (model-based)
    # If you prefer an empirical threshold, replace with out["abret"].std()
    ab_sd = float(out["abret"].std(ddof=1))
    thr = -2.0 * ab_sd if ab_sd > 0 else -0.0

    grp = out.groupby("month", as_index=False)
    monthly = grp.agg(
        abret_sum=("abret", "sum"),
        abret_mean=("abret", "mean"),
        abret_vol=("abret", "std"),
        qan_ret_sum=("r_qan", "sum"),
        mkt_ret_sum=("r_mkt", "sum"),
        n_days=("abret", "size"),
    )

    # tail frequency = share of days below threshold
    tail = out.groupby("month")["abret"].apply(lambda x: (x < thr).mean())
    tail = tail.rename("neg_tail_freq").reset_index()
    monthly = monthly.merge(tail, on="month", how="left")

    # Optional: add month start/end dates
    monthly["month_start"] = pd.to_datetime(monthly["month"] + "-01")
    monthly["month_end"] = (monthly["month_start"] + pd.offsets.MonthEnd(0)).dt.date
    monthly["month_start"] = monthly["month_start"].dt.date

    return monthly.sort_values("month").reset_index(drop=True)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.join(script_dir, "data", "qantas_share_price_data.json")

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=default_json,
                    help=f"Path to qantas_share_price_data.json (default: {default_json})")
    ap.add_argument("--start", default="2020-10-01",
                    help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default="2023-12-31",
                    help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--beta_window_days", type=int, default=0,
                    help="0 = static beta over full window; else rolling window length (e.g., 252)")
    args = ap.parse_args()

    df = load_qantas_json(args.json)
    start = _to_datetime(args.start)
    end = _to_datetime(args.end)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows after filtering by start/end dates")

    df_ret = compute_log_returns(df)

    os.makedirs(args.outdir, exist_ok=True)

    if args.beta_window_days and args.beta_window_days > 0:
        # Rolling beta/alpha (optional). This is useful if beta shifts materially over time.
        w = args.beta_window_days
        out = df_ret.copy()
        out["alpha_hat"] = np.nan
        out["beta_hat"] = np.nan
        out["abret"] = np.nan

        rq = out["r_qan"].to_numpy()
        rm = out["r_mkt"].to_numpy()

        for i in range(w - 1, len(out)):
            sl = slice(i - w + 1, i + 1)
            fit = fit_market_model_ols(rq[sl], rm[sl])
            out.loc[out.index[i], "alpha_hat"] = fit.alpha
            out.loc[out.index[i], "beta_hat"] = fit.beta
            out.loc[out.index[i], "abret"] = rq[i] - (fit.alpha + fit.beta * rm[i])

        # Drop initial rows without rolling estimates
        out = out.dropna(subset=["abret"]).reset_index(drop=True)
        daily_out = out[["date", "close", "index_close", "r_qan", "r_mkt", "alpha_hat", "beta_hat", "abret"]].copy()
        fit_summary = None
    else:
        # Static beta/alpha fit over the whole filtered window
        fit = fit_market_model_ols(df_ret["r_qan"].to_numpy(), df_ret["r_mkt"].to_numpy())
        out = compute_abnormal_returns(df_ret, fit)
        daily_out = out[["date", "close", "index_close", "r_qan", "r_mkt", "abret"]].copy()
        fit_summary = fit

    daily_path = os.path.join(args.outdir, "abret_daily.csv")
    daily_out.to_csv(daily_path, index=False)

    monthly = aggregate_monthly(daily_out.rename(columns={"date": "date"}))
    monthly_path = os.path.join(args.outdir, "abret_monthly.csv")
    monthly.to_csv(monthly_path, index=False)

    print(f"Wrote: {daily_path}")
    print(f"Wrote: {monthly_path}")

    if fit_summary is not None:
        print("\nMarket model (static OLS):")
        print(f"  n        = {fit_summary.n}")
        print(f"  alpha    = {fit_summary.alpha:.8f} (daily log-return intercept)")
        print(f"  beta     = {fit_summary.beta:.4f} (QAN vs ASX200)")
        print(f"  resid_sd = {fit_summary.resid_sd:.6f} (daily)")

        # quick sanity: abret should have ~0 mean if model includes alpha
        print(f"\nAbret daily mean (should be ~0): {daily_out['abret'].mean():.8f}")


if __name__ == "__main__":
    main()