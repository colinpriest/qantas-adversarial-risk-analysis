# fit_shock_priors.py
#
# Fits data-driven priors for gamma_A (ASA public mobilisation shock magnitude)
# from the cross-company voting-recommendations panel.
#
# NEW: Strike channel uses Bayesian logistic regression (MAP + Laplace approx)
# to avoid quasi-separation dominating estimates.
#
# Usage:
#   python fit_shock_priors.py --asof 2023-10-01
#   python fit_shock_priors.py --write_all_checkpoints
#
# Outputs:
#   data/priors/shock_priors_*.json
#
# Notes:
# - "As-of" filtering happens ONLY here. build_historical_belief_table.py remains unfiltered.

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------

def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _to_year_month(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df["year"], errors="raise").astype(int)
    m = pd.to_numeric(df["month"], errors="raise").astype(int)
    return y.astype(str) + "-" + m.map(lambda x: f"{x:02d}")


def _asof_max_month(asof_date: str) -> str:
    p = pd.Period(asof_date[:7], freq="M")
    max_p = p - 1
    return str(max_p)


def _clip01(x: pd.Series, eps: float = 1e-4) -> pd.Series:
    return x.clip(lower=eps, upper=1 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def _vote_logit_to_belief_prior(
    mu_logit: float, sd_logit: float, lambda_draws: np.ndarray, n_mc: int = 200_000
) -> Tuple[float, float]:
    """
    Map a vote-logit-scale effect N(mu_logit, sd_logit) onto belief scale
    by dividing by lambda_rem draws from the Stan posterior (Monte Carlo propagation).
    Returns (mu_belief, sd_belief).
    """
    rng = np.random.default_rng(123)
    lam = np.asarray(lambda_draws, dtype=float)
    lam = lam[np.isfinite(lam) & (lam > 1e-8)]
    if lam.size < 50:
        raise ValueError("lambda_rem draws too few/invalid to map to belief scale")

    delta = rng.normal(mu_logit, sd_logit, size=n_mc)
    lam_s = rng.choice(lam, size=n_mc, replace=True)
    gamma_b = delta / lam_s
    return float(np.mean(gamma_b)), float(np.std(gamma_b, ddof=1))


def _ols_beta_se(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        ridge = 1e-6
        XtX_inv = np.linalg.inv(XtX + ridge * np.eye(k))

    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = (resid @ resid) / dof
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))
    sigma_hat = float(np.sqrt(sigma2))
    return beta, se, sigma_hat


def _bayes_logit_map_laplace(
    X: np.ndarray,
    y: np.ndarray,
    prior_sd: float = 2.5,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian logistic regression with independent Normal prior:
      beta ~ N(0, prior_sd^2 I)

    Returns:
      beta_map (k,)
      se_laplace (k,)   where Var(beta) ≈ H^{-1} at MAP (Laplace approx)

    Newton steps on the log-posterior (equivalently, penalised log-likelihood).
    """
    n, k = X.shape
    beta = np.zeros(k, dtype=float)
    lam = 1.0 / (prior_sd ** 2)

    for _ in range(max_iter):
        eta = X @ beta
        p = _sigmoid(eta)

        # Gradient of log-likelihood: X^T (y - p)
        g_ll = X.T @ (y - p)

        # Add gradient of log-prior: -lam * beta
        g = g_ll - lam * beta

        # Hessian of negative log-likelihood: X^T W X, where W = p(1-p)
        w = p * (1.0 - p)
        # Build Hessian of negative log-posterior: H = X^T W X + lam I
        # (so Newton step solves H * step = g)
        Xw = X * w[:, None]
        H = X.T @ Xw + lam * np.eye(k)

        # Solve for step; damp if needed
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # tiny ridge for stability
            H = H + 1e-6 * np.eye(k)
            step = np.linalg.solve(H, g)

        beta_new = beta + step

        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Laplace covariance approx: cov = H^{-1} at MAP
    eta = X @ beta
    p = _sigmoid(eta)
    w = p * (1.0 - p)
    Xw = X * w[:, None]
    H = X.T @ Xw + lam * np.eye(k)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.inv(H + 1e-6 * np.eye(k))

    se = np.sqrt(np.diag(cov))
    return beta, se


def _make_design(df: pd.DataFrame, outcome: str) -> Tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Design:
      intercept
      asa_against (treatment)
      prior_year_pct (control)
      log_mkt_cap (control)
      industry FE (gics) if present
    Returns:
      X, y, names, mask_used
    """
    cols_needed = ["asa_against"]
    if "prior_year_pct" in df.columns:
        cols_needed.append("prior_year_pct")
    if "log_mkt_cap" in df.columns:
        cols_needed.append("log_mkt_cap")
    for c in cols_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X_parts: List[np.ndarray] = []
    names: List[str] = []

    intercept = np.ones((len(df), 1), dtype=float)
    X_parts.append(intercept)
    names.append("intercept")

    t = df["asa_against"].fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
    X_parts.append(t)
    names.append("asa_against")

    if "prior_year_pct" in df.columns:
        v = df["prior_year_pct"]
        X_parts.append(v.fillna(v.mean()).to_numpy(dtype=float).reshape(-1, 1))
        names.append("prior_year_pct")

    if "log_mkt_cap" in df.columns:
        v = df["log_mkt_cap"]
        X_parts.append(v.fillna(v.mean()).to_numpy(dtype=float).reshape(-1, 1))
        names.append("log_mkt_cap")

    if "gics" in df.columns:
        g = df["gics"].astype(str).fillna("UNKNOWN")
        dummies = pd.get_dummies(g, prefix="gics", drop_first=True)
        if dummies.shape[1] > 0:
            X_parts.append(dummies.to_numpy(dtype=float))
            names.extend(list(dummies.columns))

    X = np.concatenate(X_parts, axis=1)
    y = df[outcome].to_numpy(dtype=float)

    mask = np.isfinite(y)
    X = X[mask, :]
    y = y[mask]
    return X, y, names, mask


@dataclass
class NormalPrior:
    dist: str
    mu: float
    sigma: float
    n: int
    model: str
    asof: str
    max_included_month: str
    notes: str


def _prior_from_beta(beta_hat: float, se_hat: float, extra_sigma: float,
                     *, asof: str, max_month: str, n: int, model: str, notes: str) -> NormalPrior:
    sigma = float(math.sqrt(se_hat**2 + extra_sigma**2))
    return NormalPrior(
        dist="normal",
        mu=float(beta_hat),
        sigma=sigma,
        n=int(n),
        model=model,
        asof=asof,
        max_included_month=max_month,
        notes=notes,
    )


# -------------------------
# Main fitting
# -------------------------

def fit_priors(vote_csv: str, asof_date: str, strike_prior_sd: float = 2.5) -> Dict[str, NormalPrior]:
    votes = pd.read_csv(vote_csv)
    _require_cols(votes, ["year", "month", "asa_against"], "voting-recommendations.csv")

    votes["year_month"] = _to_year_month(votes)
    max_month = _asof_max_month(asof_date)

    # Enforce "as-of": keep <= max_month (lexicographic safe for YYYY-MM)
    df = votes.loc[votes["year_month"] <= max_month].copy()

    # Coerce numeric columns if present
    for c in ["rem_against_pct", "prior_year_pct", "log_mkt_cap", "first_strike"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["asa_against"] = pd.to_numeric(df["asa_against"], errors="coerce").fillna(0).astype(int)

    # --- DIAGNOSTICS: ASA treatment balance + separation checks (keep short) ---
    n_all = len(df)
    n_t = int((df["asa_against"] == 1).sum())
    n_c = n_all - n_t
    print(f"[diag asof={asof_date} max_month={max_month}] n={n_all} treated={n_t} control={n_c}")

    if "rem_against_pct" in df.columns:
        m_t = df.loc[df["asa_against"] == 1, "rem_against_pct"].mean()
        m_c = df.loc[df["asa_against"] != 1, "rem_against_pct"].mean()
        print(f"[diag] rem_against_pct mean: treated={m_t:.3f} control={m_c:.3f}")

    if "first_strike" in df.columns:
        ct = pd.crosstab(df["asa_against"], df["first_strike"].fillna(0).astype(int), dropna=False)
        print("[diag] crosstab asa_against x first_strike:\n", ct)
        print("[diag] row rates:\n", (ct.div(ct.sum(axis=1), axis=0).round(3)))
        if (ct == 0).any().any():
            print("[diag][WARN] zero cell(s) in crosstab -> quasi-separation risk (Bayesian logit will stabilise)")

    # ---------- Outcome channel 1: vote opposition (continuous, logit scale) ----------
    _require_cols(df, ["rem_against_pct"], "voting-recommendations.csv (for vote channel)")
    df["y_vote_logit"] = _logit(_clip01(df["rem_against_pct"]).to_numpy(dtype=float))

    X1, y1, names1, _mask1 = _make_design(df, "y_vote_logit")
    b1, se1, sigma1 = _ols_beta_se(X1, y1)

    idx_t = names1.index("asa_against")
    beta_vote = float(b1[idx_t])
    se_vote = float(se1[idx_t])

    extra_vote = float(0.35 * sigma1)

    prior_vote = _prior_from_beta(
        beta_hat=beta_vote,
        se_hat=se_vote,
        extra_sigma=extra_vote,
        asof=asof_date,
        max_month=max_month,
        n=len(y1),
        model="OLS: logit(rem_against_pct) ~ asa_against + prior_year_pct + log_mkt_cap + gics_FE",
        notes="Vote-channel prior on additive logit scale. sigma = sqrt(SE^2 + (0.35*resid_sd)^2).",
    )

    priors: Dict[str, NormalPrior] = {"gamma_A_vote_logit": prior_vote}

    # ---------- Outcome channel 2: first strike (binary, Bayesian logistic; log-odds scale) ----------
    if "first_strike" in df.columns:
        # Ensure 0/1
        df["y_strike"] = df["first_strike"].fillna(0).astype(int)

        X2, y2, names2, _mask2 = _make_design(df, "y_strike")
        idx_t2 = names2.index("asa_against")

        beta2, se2 = _bayes_logit_map_laplace(X2, y2, prior_sd=strike_prior_sd)
        beta_strike = float(beta2[idx_t2])
        se_strike = float(se2[idx_t2])

        # Extra dispersion term: small, because Laplace SE already captures much of the uncertainty
        extra_strike = 0.10  # keep conservative; you can tune (0.05-0.20)

        prior_strike = _prior_from_beta(
            beta_hat=beta_strike,
            se_hat=se_strike,
            extra_sigma=extra_strike,
            asof=asof_date,
            max_month=max_month,
            n=len(y2),
            model=f"BayesLogit(MAP+Laplace): first_strike ~ asa_against + prior_year_pct + log_mkt_cap + gics_FE; "
                  f"beta~N(0,{strike_prior_sd}^2)",
            notes="Strike-channel prior on log-odds scale from Bayesian logistic regression; Laplace approx for SE.",
        )

        priors["gamma_A_strike_logodds"] = prior_strike

        # ---------- Combined prior (now logit/log-odds scales, more coherent) ----------
        v1 = prior_vote.sigma**2
        v2 = prior_strike.sigma**2
        w1 = 1.0 / v1 if v1 > 0 else 0.0
        w2 = 1.0 / v2 if v2 > 0 else 0.0
        mu_c = (w1 * prior_vote.mu + w2 * prior_strike.mu) / max(w1 + w2, 1e-12)
        sig_c = math.sqrt(1.0 / max(w1 + w2, 1e-12))

        priors["gamma_A_combined"] = NormalPrior(
            dist="normal",
            mu=float(mu_c),
            sigma=float(sig_c),
            n=int(min(prior_vote.n, prior_strike.n)),
            model="Inverse-variance blend of vote-logit and strike-logodds priors",
            asof=asof_date,
            max_included_month=max_month,
            notes="Combined prior blends two logit-scale effects (vote opposition and strike propensity).",
        )

    return priors


def write_priors_json(priors: Dict[str, NormalPrior], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {k: asdict(v) for k, v in priors.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote: {out_path}")
    for k, v in priors.items():
        print(f"{k}: mu={v.mu:.4f}, sigma={v.sigma:.4f}, n={v.n}, max_month={v.max_included_month}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vote_csv", default="data/voting-recommendations.csv")
    ap.add_argument("--asof", default="2023-10-01", help="As-of date YYYY-MM-DD (strictly excludes as-of month)")
    ap.add_argument("--outdir", default="data/priors")
    ap.add_argument("--strike_prior_sd", type=float, default=2.5, help="Prior SD for Bayes logit coefficients")
    ap.add_argument("--belief_npz", default=None,
                    help="Optional: data/belief_state_draws.npz to map vote-logit effect onto belief scale")
    ap.add_argument("--write_all_checkpoints", action="store_true", help="Write priors for C0..C3 dates")
    args = ap.parse_args()

    # Load lambda_rem draws once if belief-scale mapping requested
    lambda_rem = None
    if args.belief_npz:
        bd = np.load(args.belief_npz, allow_pickle=True)
        lambda_rem = bd["lambda_rem"]
        print(f"[belief-scale] Loaded lambda_rem: n={lambda_rem.shape[0]}, "
              f"mean={float(np.mean(lambda_rem)):.4f}, sd={float(np.std(lambda_rem)):.4f}")

    def _add_belief_prior(priors: Dict[str, NormalPrior]) -> None:
        if lambda_rem is None:
            return
        pv = priors.get("gamma_A_vote_logit")
        if pv is None:
            return
        mu_b, sd_b = _vote_logit_to_belief_prior(pv.mu, pv.sigma, lambda_rem)
        priors["gamma_A_belief_from_vote"] = NormalPrior(
            dist="normal",
            mu=mu_b,
            sigma=sd_b,
            n=pv.n,
            model="Mapped: gamma_A_vote_logit / lambda_rem (Monte Carlo over Stan lambda posterior)",
            asof=pv.asof,
            max_included_month=pv.max_included_month,
            notes="Belief-scale prior compatible with checkpoint_update.py additive B-shock.",
        )

    if args.write_all_checkpoints:
        checkpoints = [
            ("C0", "2023-10-01"),
            ("C1", "2023-10-10"),
            ("C2", "2023-10-18"),
            ("C3", "2023-11-03"),
        ]
        for ck, date in checkpoints:
            priors = fit_priors(args.vote_csv, date, strike_prior_sd=args.strike_prior_sd)
            _add_belief_prior(priors)
            out = os.path.join(args.outdir, f"shock_priors_{ck}_{date}.json")
            write_priors_json(priors, out)
    else:
        priors = fit_priors(args.vote_csv, args.asof, strike_prior_sd=args.strike_prior_sd)
        _add_belief_prior(priors)
        out = os.path.join(args.outdir, f"shock_priors_asof_{args.asof}.json")
        write_priors_json(priors, out)


if __name__ == "__main__":
    main()