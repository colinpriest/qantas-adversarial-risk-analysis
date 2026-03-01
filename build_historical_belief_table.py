# build_historical_belief_table.py
#
# Merges:
#   data/abret_monthly.csv
#   data/media_shock_draws.npz
#   data/agm-votes.csv
#   data/voting-recommendations.csv   (NEW: market-wide + Qantas-specific features)
# into:
#   data/historical_belief_table.csv
#
# Usage:
#   python build_historical_belief_table.py --start 2020-10 --end 2023-12
#
# Notes:
# - This script NEVER filters out months from the requested window.
# - Voting recommendations are aggregated to month-level and left-joined,
#   so your Oct–Dec 2023 rows remain present.

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def month_range(start_ym: str, end_ym: str) -> list[str]:
    start = pd.Period(start_ym, freq="M")
    end = pd.Period(end_ym, freq="M")
    if end < start:
        raise ValueError("end must be >= start")
    return [str(p) for p in pd.period_range(start=start, end=end, freq="M")]


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _safe_exp(x: pd.Series) -> pd.Series:
    # log_mkt_cap in your file is log(market cap). Using exp(log_mkt_cap) gives a size weight.
    # Guard against silly values / missing.
    x = pd.to_numeric(x, errors="coerce")
    return np.exp(x.clip(lower=-50, upper=50))


def build_vote_features(
    vote_csv: str,
    months: list[str],
    target_asx_code: str = "QAN",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (market_monthly_df, target_company_monthly_df)

    Both have a 'year_month' column. All months not present in the recommendations file
    are simply absent (caller left-joins to keep the master monthly grid intact).
    """
    votes = pd.read_csv(vote_csv)
    _require_cols(votes, ["year", "month", "asx_code"], "voting-recommendations.csv")

    # Normalise / build month key
    votes["year"] = pd.to_numeric(votes["year"], errors="raise").astype(int)
    votes["month"] = pd.to_numeric(votes["month"], errors="raise").astype(int)
    votes["year_month"] = votes["year"].astype(str) + "-" + votes["month"].map(lambda m: f"{m:02d}")
    votes["asx_code"] = votes["asx_code"].astype(str).str.upper().str.strip()

    # Coerce expected numeric/binary columns if present
    bin_cols = ["asa_against", "asa_proxy_against", "proxy_adv_against", "multi_target", "first_strike", "headline_incident"]
    for c in bin_cols:
        if c in votes.columns:
            votes[c] = pd.to_numeric(votes[c], errors="coerce")

    cont_cols = ["rem_against_pct", "prior_year_pct", "log_mkt_cap"]
    for c in cont_cols:
        if c in votes.columns:
            votes[c] = pd.to_numeric(votes[c], errors="coerce")

    # ------- Market-wide monthly aggregates -------
    # Counts/any for each binary signal; plus size-weighted versions when log_mkt_cap exists.
    market_group = votes.groupby("year_month", dropna=False)

    market = pd.DataFrame({"year_month": sorted(votes["year_month"].unique())})
    market = market.sort_values("year_month").reset_index(drop=True)

    def add_market_bin_features(col: str) -> None:
        if col not in votes.columns:
            return
        # count of "1"s (treat NaN as 0 for counting)
        cnt = market_group[col].sum(min_count=1).rename(f"mkt_{col}_count")
        any_ = market_group[col].max().fillna(0).rename(f"mkt_{col}_any")
        out = pd.concat([cnt, any_], axis=1).reset_index()
        nonlocal market
        market = market.merge(out, on="year_month", how="left")

        # size-weighted sum of indicator (exp(log_mkt_cap) * indicator)
        if "log_mkt_cap" in votes.columns:
            w = _safe_exp(votes["log_mkt_cap"])
            wcol = f"_w_{col}"
            votes[wcol] = w * votes[col].fillna(0)
            wsum = votes.groupby("year_month")[wcol].sum(min_count=1).rename(f"mkt_{col}_w_sum").reset_index()
            market = market.merge(wsum, on="year_month", how="left")
            votes.drop(columns=[wcol], inplace=True)

    for bc in ["asa_against", "asa_proxy_against", "proxy_adv_against", "multi_target", "headline_incident", "first_strike"]:
        add_market_bin_features(bc)

    # Helpful continuous aggregates for priors/controls
    if "rem_against_pct" in votes.columns:
        # unconditional mean vs conditional on ASA against
        mean_all = market_group["rem_against_pct"].mean().rename("mkt_rem_against_pct_mean").reset_index()
        market = market.merge(mean_all, on="year_month", how="left")

        if "asa_against" in votes.columns:
            cond = votes.loc[votes["asa_against"] == 1].groupby("year_month")["rem_against_pct"].mean()
            cond = cond.rename("mkt_rem_against_pct_mean_given_asa_against").reset_index()
            market = market.merge(cond, on="year_month", how="left")

    # ------- Target company (Qantas) month-level features -------
    target = votes.loc[votes["asx_code"] == target_asx_code].copy()

    if target.empty:
        # Return empty shell with year_month so merge works cleanly.
        target_monthly = pd.DataFrame({"year_month": []})
    else:
        # If multiple rows in the same month (unlikely), take max for binaries and mean for continuous.
        agg = {}
        for c in ["asa_against", "asa_proxy_against", "proxy_adv_against", "multi_target", "headline_incident", "first_strike"]:
            if c in target.columns:
                agg[c] = "max"
        for c in ["rem_against_pct", "prior_year_pct", "log_mkt_cap"]:
            if c in target.columns:
                agg[c] = "mean"
        # Also keep a label column if present
        if "company" in target.columns:
            # just take first (should be "Qantas")
            agg["company"] = "first"
        if "gics" in target.columns:
            agg["gics"] = "first"

        target_monthly = target.groupby("year_month").agg(agg).reset_index()

        # Prefix to avoid collisions
        rename = {c: f"{target_asx_code.lower()}_{c}" for c in target_monthly.columns if c != "year_month"}
        target_monthly = target_monthly.rename(columns=rename)

    # Keep only months that are in the master window (optional cleanliness)
    month_set = set(months)
    market = market[market["year_month"].isin(month_set)].copy()
    target_monthly = target_monthly[target_monthly["year_month"].isin(month_set)].copy()

    return market, target_monthly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-10", help="Start month YYYY-MM inclusive (default: 2020-10)")
    ap.add_argument("--end", default="2023-12", help="End month YYYY-MM inclusive (default: 2023-12)")
    ap.add_argument("--abret_csv", default="data/abret_monthly.csv")
    ap.add_argument("--shock_npz", default="data/media_shock_draws.npz")
    ap.add_argument("--agm_csv", default="data/agm-votes.csv")
    ap.add_argument("--vote_csv", default="data/voting-recommendations.csv", help="Voting recommendations panel CSV")
    ap.add_argument("--target_asx_code", default="QAN", help="ASX code for company-specific flags (default: QAN)")
    ap.add_argument("--out", default="data/historical_belief_table.csv")
    args = ap.parse_args()

    months = month_range(args.start, args.end)
    T = len(months)

    # ---- Load abnormal returns (monthly) ----
    ab = pd.read_csv(args.abret_csv)
    if "month" not in ab.columns:
        raise ValueError("abret_monthly.csv must contain a 'month' column in YYYY-MM format")
    ab["month"] = ab["month"].astype(str)
    keep_cols = ["month", "abret_sum"]
    if "abret_vol" in ab.columns:
        keep_cols.append("abret_vol")
    ab = ab[keep_cols].copy()

    # ---- Load shock draws and summarise ----
    d = np.load(args.shock_npz, allow_pickle=True)
    shock_draws = d["shock_draws"]          # (S, T_full)
    ym_all = d["year_month"].astype(str)    # (T_full,)

    # Map months to indices in the shock file
    idx = []
    for m in months:
        matches = np.where(ym_all == m)[0]
        if len(matches) != 1:
            raise ValueError(f"Month {m} not found uniquely in media_shock_draws.npz")
        idx.append(int(matches[0]))
    idx = np.array(idx, dtype=int)

    shocks = shock_draws[:, idx]  # (S, T)

    shock_mean = shocks.mean(axis=0)
    shock_p05 = np.quantile(shocks, 0.05, axis=0)
    shock_p95 = np.quantile(shocks, 0.95, axis=0)

    # ---- Build grid ----
    df = pd.DataFrame({
        "t": np.arange(1, T + 1),
        "year_month": months,
        "media_shock_mean": shock_mean,
        "media_shock_p05": shock_p05,
        "media_shock_p95": shock_p95,
    })

    # Merge abnormal returns
    df = df.merge(ab.rename(columns={"month": "year_month"}), on="year_month", how="left")

    # Sanity check: should have abret for all months in window
    if df["abret_sum"].isna().any():
        miss = df.loc[df["abret_sum"].isna(), "year_month"].tolist()
        raise ValueError(f"Missing abret_sum for months: {miss}. Check abret_monthly.csv coverage/window.")

    # Merge AGM vote data
    if os.path.exists(args.agm_csv):
        votes = pd.read_csv(args.agm_csv)
        votes["year_month"] = votes["year_month"].astype(str).str.strip()
        df = df.merge(votes, on="year_month", how="left")
        matched = votes["year_month"].isin(df["year_month"]).sum()
        print(f"AGM votes merged: {matched}/{len(votes)} months matched")
    else:
        df["vote_against_rem_pct"] = np.nan
        df["vote_against_chair_pct"] = np.nan
        print(f"Warning: {args.agm_csv} not found; vote columns left as NaN")

    # ASA engagement indicators (your crisis-specific manual flags stay as-is)
    df["asa_engagement_private"] = 0
    df["asa_public_mobilisation"] = 0
    df.loc[df["year_month"] == "2023-09", "asa_engagement_private"] = 1
    df.loc[df["year_month"] == "2023-10", "asa_engagement_private"] = 1
    df.loc[df["year_month"] == "2023-10", "asa_public_mobilisation"] = 1
    df.loc[df["year_month"] == "2023-11", "asa_engagement_private"] = 1
    df.loc[df["year_month"] == "2023-11", "asa_public_mobilisation"] = 1

    # ---- NEW: merge voting-recommendations features (market-wide + Qantas-specific) ----
    if os.path.exists(args.vote_csv):
        mkt, target = build_vote_features(
            vote_csv=args.vote_csv,
            months=months,
            target_asx_code=args.target_asx_code,
        )
        df = df.merge(mkt, on="year_month", how="left")
        df = df.merge(target, on="year_month", how="left")

        # Fill market-wide columns with 0 where month has no recs in the panel
        mkt_cols = [c for c in df.columns if c.startswith("mkt_")]
        df[mkt_cols] = df[mkt_cols].fillna(0.0)

        # Fill company-specific binary flags with 0; keep continuous fields as NaN (unknown)
        tgt_prefix = f"{args.target_asx_code.lower()}_"
        tgt_bin_suffixes = [
            "asa_against",
            "asa_proxy_against",
            "proxy_adv_against",
            "multi_target",
            "headline_incident",
            "first_strike",
        ]
        for sfx in tgt_bin_suffixes:
            col = tgt_prefix + sfx
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        print(f"Voting recs merged: market months={len(mkt)}, {args.target_asx_code} months={len(target)}")
    else:
        print(f"Warning: {args.vote_csv} not found; voting recommendation columns not added")

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out} ({len(df)} rows)")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()