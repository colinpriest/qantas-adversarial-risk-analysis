# fit_belief_model_stan.py
#
# Fits the belief state model and exports posterior draws for B_t.
#
# Drop-in replacement for anchored-scale belief_model.stan where lambda_rem is FIXED to 1.
#
# Usage:
#   python fit_belief_model_stan.py --data data/stan_belief_data.json --stan models/belief_model.stan --outdir data

from __future__ import annotations

import argparse
import json
import os
import platform

import numpy as np
import pandas as pd

# Ensure the C++ toolchain is discoverable by CmdStanPy on Windows.
if platform.system() == "Windows" and "MAKE" not in os.environ:
    _rtools_make = r"C:\rtools40\usr\bin\make.exe"
    _rtools_gpp = r"C:\rtools40\ucrt64\bin"
    if os.path.isfile(_rtools_make):
        os.environ["MAKE"] = _rtools_make
        os.environ["PATH"] = (
            _rtools_gpp + os.pathsep
            + os.path.dirname(_rtools_make) + os.pathsep
            + os.environ.get("PATH", "")
        )

from cmdstanpy import CmdStanModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stan_belief_data.json",
                    help="Path to stan_belief_data.json (default: data/stan_belief_data.json)")
    ap.add_argument("--stan", default="models/belief_model.stan",
                    help="Path to belief_model.stan (default: models/belief_model.stan)")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--iter_warmup", type=int, default=2000)
    ap.add_argument("--iter_sampling", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--adapt_delta", type=float, default=0.9999)
    ap.add_argument("--max_treedepth", type=int, default=20)
    ap.add_argument("--start_month", default="2020-10",
                    help="Start month for labeling B_t draws (YYYY-MM). Default matches project window.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = CmdStanModel(stan_file=args.stan)
    fit = model.sample(
        data=data,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        seed=args.seed,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        show_progress=True,
    )

    # Extract B draws
    B_draws = fit.stan_variable("B")  # shape (S, T)

    # Build month labels from known window
    T = int(data["T"])
    start = pd.Period(args.start_month, freq="M")
    months = [str(start + i) for i in range(T)]

    # Summaries
    B_mean = B_draws.mean(axis=0)
    B_p05 = np.quantile(B_draws, 0.05, axis=0)
    B_p95 = np.quantile(B_draws, 0.95, axis=0)

    summary = pd.DataFrame({
        "t": np.arange(1, T + 1),
        "year_month": months,
        "B_mean": B_mean,
        "B_p05": B_p05,
        "B_p95": B_p95,
    })

    summary_path = os.path.join(args.outdir, "belief_state_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Extract additional posterior draws for checkpoint updating
    alpha_rem = fit.stan_variable("alpha_rem")      # (draws,)
    sigma_rem = fit.stan_variable("sigma_rem")      # (draws,)
    beta = fit.stan_variable("beta")                # (draws,)
    sigma_B = fit.stan_variable("sigma_B")          # (draws,)
    sigma_B0 = fit.stan_variable("sigma_B0")        # (draws,)
    rho = fit.stan_variable("rho")                  # (draws,) — transformed param

    # Anchored model: lambda_rem is FIXED at 1.0 (not a Stan parameter).
    # Keep it in the NPZ for backward compatibility with downstream code.
    lambda_rem = np.ones_like(alpha_rem, dtype=float)

    draws_path = os.path.join(args.outdir, "belief_state_draws.npz")
    np.savez_compressed(
        draws_path,
        year_month=np.array(months, dtype=object),
        B_draws=B_draws,
        alpha_rem=alpha_rem,
        lambda_rem=lambda_rem,   # constant 1.0
        sigma_rem=sigma_rem,
        beta=beta,
        sigma_B=sigma_B,
        sigma_B0=sigma_B0,
        rho=rho,
    )

    diag_path = os.path.join(args.outdir, "belief_state_diagnostics.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(str(fit.diagnose()))

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {draws_path}")
    print("  Saved keys: B_draws, alpha_rem, lambda_rem(=1), sigma_rem, beta, sigma_B, sigma_B0, rho, year_month")
    print(f"  B_draws shape: {B_draws.shape}")
    print(f"Wrote: {diag_path}")
    print("\nDiagnostics:")
    print(fit.diagnose())


if __name__ == "__main__":
    main()