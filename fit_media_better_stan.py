# fit_media_better_stan.py
#
# Fits models/media_better.stan using cmdstanpy and your data/stan_media_data.json
# Outputs posterior draws for C[t] and M[t] for downstream belief modelling.
#
# Usage:
#   python fit_media_better_stan.py --data data/stan_media_data.json --stan models/media_better.stan --outdir data

from __future__ import annotations

import argparse
import json
import os
import platform

import numpy as np
import pandas as pd

# Ensure the C++ toolchain is discoverable by CmdStanPy on Windows.
# RTools4 at C:\rtools40 is preferred (space-free path required by CmdStan make).
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
    ap.add_argument("--data", default="data/stan_media_data.json",
                    help="Path to stan_media_data.json (default: data/stan_media_data.json)")
    ap.add_argument("--stan", default="models/media_better.stan",
                    help="Path to media_better.stan (default: models/media_better.stan)")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--iter_warmup", type=int, default=2000)
    ap.add_argument("--iter_sampling", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
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
        adapt_delta=0.99,
        max_treedepth=15,
        show_progress=True,
    )

    # Extract draws
    # Each returns (draws, T)
    C_draws = fit.stan_variable("C")
    logM_draws = fit.stan_variable("logM")
    M_draws = np.exp(logM_draws)

    # Summaries
    def summarize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = arr.mean(axis=0)
        p05 = np.quantile(arr, 0.05, axis=0)
        p95 = np.quantile(arr, 0.95, axis=0)
        return mean, p05, p95

    C_mean, C_p05, C_p95 = summarize(C_draws)
    M_mean, M_p05, M_p95 = summarize(M_draws)

    T = data["T"]
    # Reconstruct year_month from the complete media xlsx (authoritative source for date labels)
    media_xlsx = os.path.join(os.path.dirname(args.data), "media_monthly_complete.xlsx")
    if os.path.exists(media_xlsx):
        month_df = pd.read_excel(media_xlsx, sheet_name=0, engine="openpyxl", usecols=["year_month"])
        months = month_df["year_month"].astype(str).str.strip().tolist()[:T]
    else:
        # Fallback: regenerate from default start
        start = pd.Period("2020-10", freq="M")
        months = [str(start + i) for i in range(T)]

    summary = pd.DataFrame({
        "t": np.arange(1, T + 1),
        "year_month": months,
        "C_mean": C_mean,
        "C_p05": C_p05,
        "C_p95": C_p95,
        "M_mean": M_mean,
        "M_p05": M_p05,
        "M_p95": M_p95,
    })

    summary_path = os.path.join(args.outdir, "media_better_summary.csv")
    summary.to_csv(summary_path, index=False)

    draws_path = os.path.join(args.outdir, "media_better_draws.npz")
    np.savez_compressed(
        draws_path,
        year_month=np.array(months, dtype=object),
        C_draws=C_draws,   # (S, T)
        M_draws=M_draws,   # (S, T)
    )

    diag_path = os.path.join(args.outdir, "media_better_diagnostics.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(str(fit.diagnose()))

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {draws_path}")
    print(f"Wrote: {diag_path}")
    print("\nKey diagnostics (first pass):")
    print(fit.diagnose())


if __name__ == "__main__":
    main()