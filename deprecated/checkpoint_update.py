#!/usr/bin/env python3
"""
checkpoint_update.py (drop-in replacement)

Build belief distributions at multiple checkpoints for Qantas 2023, incorporating:
- Private ASA engagement (known to Qantas before public mobilisation)
- Public ASA mobilisation (becomes visible to market mid-Oct)
- Review announcement shock (10 Oct)
- AGM measurement update using remuneration vote only (03 Nov)

Inputs:
  - data/belief_state_draws.npz  (from Stan fit)
      keys: year_month (T,), B_draws (S,T), alpha_rem (S,), sigma_rem (S,), ...
  - data/historical_belief_table.csv (or XLSX)  (single source of truth)
      must include columns:
        year_month (YYYY-MM)
        asa_engagement_private (0/1)
        asa_public_mobilisation (0/1)

NEW (optional):
  - data/priors/shock_priors_C2_2023-10-18.json (default)
      created by fit_shock_priors.py
      expects keys: gamma_A_vote_logit, gamma_A_strike_prob, gamma_A_combined
      each with fields: dist, mu, sigma, ...

Outputs:
  - data/checkpoints/belief_C0_2023-10-01.npz ... belief_C3_2023-11-03.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


def logit(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return math.log(p / (1.0 - p))


def normal_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Elementwise log N(x | mu, sigma). x can be scalar; arrays must broadcast."""
    sigma = np.maximum(sigma, 1e-12)
    z = (x - mu) / sigma
    return -0.5 * (z * z) - np.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def softmax_logw(logw: np.ndarray) -> np.ndarray:
    m = np.max(logw)
    w = np.exp(logw - m)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def weighted_resample_idx(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return rng.choice(len(weights), size=n, replace=True, p=weights)


def read_monthly_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "year_month" not in df.columns:
        raise ValueError(f"Monthly table missing required column 'year_month': {path}")

    df["year_month"] = df["year_month"].astype(str).str.slice(0, 7)

    for col in ["asa_engagement_private", "asa_public_mobilisation"]:
        if col not in df.columns:
            raise ValueError(f"Monthly table missing required column '{col}': {path}")
        df[col] = df[col].fillna(0).astype(int)
        bad = df.loc[~df[col].isin([0, 1]), col].unique()
        if len(bad) > 0:
            raise ValueError(f"Column {col} must be binary 0/1. Found: {bad}")

    return df


def idx_of_month(months: np.ndarray, target: str) -> int:
    target = str(target)[:7]
    hits = np.where(months == target)[0]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one match for month={target}, got {len(hits)}")
    return int(hits[0])


def summarize_draws(x: np.ndarray, name: str) -> str:
    return (
        f"{name}: mean={float(np.mean(x)):.4f}, "
        f"p05={float(np.quantile(x, 0.05)):.4f}, "
        f"p50={float(np.quantile(x, 0.50)):.4f}, "
        f"p95={float(np.quantile(x, 0.95)):.4f}"
    )


def load_gamma_A_prior(prior_json: Path, key: str) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns (mu, sigma, raw_dict_for_key).
    Auto-detects key if the requested one is missing:
      gamma_A_vote_logit > gamma_A_combined > first available.
    """
    if not prior_json.exists():
        raise FileNotFoundError(f"Prior JSON not found: {prior_json}")

    with prior_json.open("r", encoding="utf-8") as f:
        d = json.load(f)

    # Auto-detect: prefer requested key, fall back gracefully
    if key not in d:
        fallback_order = ["gamma_A_vote_logit", "gamma_A_combined"]
        resolved = None
        for fb in fallback_order:
            if fb in d:
                resolved = fb
                break
        if resolved is None:
            raise KeyError(
                f"Key '{key}' not found in {prior_json}. Available keys: {list(d.keys())}"
            )
        print(f"[prior] Requested key '{key}' not in {prior_json}; falling back to '{resolved}'")
        key = resolved

    entry = d[key]
    if entry.get("dist") != "normal":
        raise ValueError(f"Expected dist='normal' for {key} in {prior_json}, got {entry.get('dist')}")

    mu = float(entry["mu"])
    sigma = float(entry["sigma"])
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"Bad mu/sigma in {prior_json} for {key}: mu={mu}, sigma={sigma}")

    return mu, sigma, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/belief_state_draws.npz")
    ap.add_argument("--table", default="data/historical_belief_table.csv")
    ap.add_argument("--outdir", default="data/checkpoints")
    ap.add_argument("--seed", type=int, default=123)

    # NEW: data-driven priors for gamma_A (public mobilisation)
    ap.add_argument(
        "--gamma_A_prior_json",
        default="data/priors/shock_priors_C2_2023-10-18.json",
        help="JSON produced by fit_shock_priors.py; default is C2 because mobilisation is applied at C2.",
    )
    ap.add_argument(
        "--gamma_A_prior_key",
        default="gamma_A_vote_logit",
        choices=["gamma_A_vote_logit", "gamma_A_belief_from_vote", "gamma_A_combined", "gamma_A_strike_logodds"],
        help="Which key from shock_priors_*.json to use for gamma_A. Default aligns with anchored belief scale.",
    )

    # Back-compat CLI overrides (used if --no_use_gamma_A_prior_json)
    ap.add_argument("--gamma_A_mean", type=float, default=0.8, help="(fallback) Public ASA mobilisation shock mean")
    ap.add_argument("--gamma_A_sd", type=float, default=0.4, help="(fallback) Public ASA mobilisation shock sd")
    ap.add_argument(
        "--no_use_gamma_A_prior_json",
        action="store_true",
        help="Ignore prior JSON and use --gamma_A_mean/--gamma_A_sd instead.",
    )

    # Private engagement is fraction of public impact
    ap.add_argument("--kappa_a", type=float, default=2.0, help="Beta(a,b) for kappa where gamma_E = kappa * gamma_A")
    ap.add_argument("--kappa_b", type=float, default=2.0)

    # Review shock (kept as simple normal prior here)
    ap.add_argument("--gamma_review_mean", type=float, default=-0.2, help="Review announcement shock mean")
    ap.add_argument("--gamma_review_sd", type=float, default=0.2, help="Review announcement shock sd")

    # AGM observed vote
    ap.add_argument("--agm_rem_against", type=float, default=0.829, help="Observed rem 'against' proportion at AGM")

    # Optional experimentation: redraw shocks independently per checkpoint
    ap.add_argument(
        "--redraw_shocks_each_checkpoint",
        action="store_true",
        help="If set, draws (gamma_A, gamma_E, gamma_review) separately for each checkpoint and saves them per file. "
             "Default is to draw once and apply consistently across checkpoints (recommended).",
    )

    args = ap.parse_args()

    npz_path = Path(args.npz)
    table_path = Path(args.table)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load posterior draws from Stan
    d = np.load(npz_path, allow_pickle=True)
    year_month = d["year_month"].astype(str)
    B_draws = d["B_draws"]          # (S, T)
    alpha_rem = d["alpha_rem"]      # (S,)
    assert np.allclose(d.get("lambda_rem", np.ones_like(alpha_rem)), 1.0, atol=1e-6), \
        "Anchored model expects lambda_rem == 1"
    sigma_rem = d["sigma_rem"]      # (S,)

    S, T = B_draws.shape
    if year_month.shape[0] != T:
        raise ValueError(f"Mismatch: year_month has {year_month.shape[0]} months but B_draws has T={T}")

    # Load monthly table (source of truth for engagement/mobilisation flags)
    df = read_monthly_table(table_path)

    # Ensure table covers months we reference
    needed = ["2023-09", "2023-10", "2023-11"]
    for m in needed:
        if m not in set(df["year_month"].values):
            raise ValueError(f"Monthly table does not contain required month {m}")

    # Index in the Stan timeline (training ends at 2023-09 typically)
    i_sep = idx_of_month(year_month, "2023-09")
    B_sep = B_draws[:, i_sep]  # (S,)

    # --- Prior selection for gamma_A ---
    prior_json_path = Path(args.gamma_A_prior_json)
    if args.no_use_gamma_A_prior_json:
        gamma_A_mu = float(args.gamma_A_mean)
        gamma_A_sd = float(args.gamma_A_sd)
        prior_meta = {"source": "cli_fallback", "mu": gamma_A_mu, "sigma": gamma_A_sd}
    else:
        gamma_A_mu, gamma_A_sd, prior_entry = load_gamma_A_prior(prior_json_path, args.gamma_A_prior_key)
        prior_meta = {
            "source": str(prior_json_path),
            "key": args.gamma_A_prior_key,
            "mu": gamma_A_mu,
            "sigma": gamma_A_sd,
            "n": prior_entry.get("n", None),
            "max_included_month": prior_entry.get("max_included_month", None),
            "model": prior_entry.get("model", None),
        }

    def draw_shocks(rng_local: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Draw uncertain shock parameters per sample
        gamma_A = rng_local.normal(gamma_A_mu, gamma_A_sd, size=S)
        gamma_A = np.maximum(gamma_A, 0.0)  # mobilisation should not reduce pressure

        kappa = rng_local.beta(args.kappa_a, args.kappa_b, size=S)
        gamma_E = kappa * gamma_A  # private engagement is fraction of public impact

        gamma_review = rng_local.normal(args.gamma_review_mean, args.gamma_review_sd, size=S)
        return gamma_A, gamma_E, gamma_review

    # Default: draw once and keep consistent across checkpoints (recommended)
    gamma_A_global, gamma_E_global, gamma_review_global = draw_shocks(rng)

    # Checkpoint construction:
    # C0 (01-Oct): Market does NOT see mobilisation; management sees engagement from Sep
    # - Market belief = B_sep
    # - Management belief = B_sep + gamma_E
    def compute_checkpoints(gamma_A: np.ndarray, gamma_E: np.ndarray, gamma_review: np.ndarray):
        B_mkt_C0 = B_sep.copy()
        B_mgmt_C0 = B_sep + gamma_E

        B_mkt_C1 = B_mkt_C0 + gamma_review
        B_mgmt_C1 = B_mgmt_C0 + gamma_review

        B_mkt_C2 = B_mkt_C1 + gamma_A
        B_mgmt_C2 = B_mgmt_C1 + gamma_A

        # C3 (AGM): rem-only measurement update on market belief
        y_obs = float(args.agm_rem_against)
        y_logit = logit(y_obs)

        mu = alpha_rem + B_mkt_C2
        logw = normal_logpdf(y_logit, mu, sigma_rem)
        w = softmax_logw(logw)
        idx = weighted_resample_idx(w, n=S, rng=rng)

        B_mkt_C3 = B_mkt_C2[idx]
        B_mgmt_C3 = B_mgmt_C2[idx]  # aligned via same resample

        return (B_mkt_C0, B_mgmt_C0,
                B_mkt_C1, B_mgmt_C1,
                B_mkt_C2, B_mgmt_C2,
                B_mkt_C3, B_mgmt_C3)

    if args.redraw_shocks_each_checkpoint:
        # For experiments: redraw shocks independently per checkpoint file.
        # Note: this breaks the "single realised shock" interpretation; use for sensitivity sweeps only.
        y_obs = float(args.agm_rem_against)

        def save_checkpoint(name: str, B_mkt: np.ndarray, B_mgmt: np.ndarray,
                            gamma_A: np.ndarray, gamma_E: np.ndarray, gamma_review: np.ndarray):
            out = outdir / f"belief_{name}.npz"
            np.savez_compressed(
                out,
                checkpoint=name,
                seed=args.seed,
                B_mkt=B_mkt,
                B_mgmt=B_mgmt,
                gamma_A=gamma_A,
                gamma_E=gamma_E,
                gamma_review=gamma_review,
                agm_rem_against=y_obs,
                gamma_A_prior=prior_meta,
            )
            print(f"Wrote: {out}")

        # C0
        gA0, gE0, gR0 = draw_shocks(np.random.default_rng(args.seed + 1000))
        (B0m, B0g, *_rest) = compute_checkpoints(gA0, gE0, gR0)
        save_checkpoint("C0_2023-10-01", B0m, B0g, gA0, gE0, gR0)

        # C1
        gA1, gE1, gR1 = draw_shocks(np.random.default_rng(args.seed + 1001))
        (_, _, B1m, B1g, *_rest) = compute_checkpoints(gA1, gE1, gR1)
        save_checkpoint("C1_2023-10-10", B1m, B1g, gA1, gE1, gR1)

        # C2
        gA2, gE2, gR2 = draw_shocks(np.random.default_rng(args.seed + 1002))
        (_, _, _, _, B2m, B2g, *_rest) = compute_checkpoints(gA2, gE2, gR2)
        save_checkpoint("C2_2023-10-18", B2m, B2g, gA2, gE2, gR2)

        # C3
        gA3, gE3, gR3 = draw_shocks(np.random.default_rng(args.seed + 1003))
        (_, _, _, _, _, _, B3m, B3g) = compute_checkpoints(gA3, gE3, gR3)
        save_checkpoint("C3_2023-11-03", B3m, B3g, gA3, gE3, gR3)

        # Summaries printed for the last run (C3)
        print("\nSummaries (C3 run):")
        print(summarize_draws(B3m, "B_mkt_C3 (post-AGM)"))
        print("\nShock priors actually drawn (C3 run):")
        print(summarize_draws(gA3, "gamma_A"))
        print(summarize_draws(gE3, "gamma_E"))
        print(summarize_draws(gR3, "gamma_review"))

    else:
        # Recommended: draw once; keep consistent across checkpoints.
        (B_mkt_C0, B_mgmt_C0,
         B_mkt_C1, B_mgmt_C1,
         B_mkt_C2, B_mgmt_C2,
         B_mkt_C3, B_mgmt_C3) = compute_checkpoints(gamma_A_global, gamma_E_global, gamma_review_global)

        y_obs = float(args.agm_rem_against)

        def save_checkpoint(name: str, B_mkt: np.ndarray, B_mgmt: np.ndarray):
            out = outdir / f"belief_{name}.npz"
            np.savez_compressed(
                out,
                checkpoint=name,
                seed=args.seed,
                B_mkt=B_mkt,
                B_mgmt=B_mgmt,
                gamma_A=gamma_A_global,
                gamma_E=gamma_E_global,
                gamma_review=gamma_review_global,
                agm_rem_against=y_obs,
                gamma_A_prior=prior_meta,
            )
            print(f"Wrote: {out}")

        save_checkpoint("C0_2023-10-01", B_mkt_C0, B_mgmt_C0)
        save_checkpoint("C1_2023-10-10", B_mkt_C1, B_mgmt_C1)
        save_checkpoint("C2_2023-10-18", B_mkt_C2, B_mgmt_C2)
        save_checkpoint("C3_2023-11-03", B_mkt_C3, B_mgmt_C3)

        print("\nSummaries:")
        print(summarize_draws(B_mkt_C0, "B_mkt_C0"))
        print(summarize_draws(B_mgmt_C0, "B_mgmt_C0"))
        print(summarize_draws(B_mkt_C1, "B_mkt_C1"))
        print(summarize_draws(B_mkt_C2, "B_mkt_C2"))
        print(summarize_draws(B_mkt_C3, "B_mkt_C3 (post-AGM)"))
        print("\nShock priors actually drawn:")
        print(summarize_draws(gamma_A_global, "gamma_A"))
        print(summarize_draws(gamma_E_global, "gamma_E"))
        print(summarize_draws(gamma_review_global, "gamma_review"))
        print("\nGamma_A prior source:")
        print(prior_meta)


if __name__ == "__main__":
    main()