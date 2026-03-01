# toy_ara_qantas_governance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# ----------------------------
# Decision set D (your 4 packages)
# ----------------------------

@dataclass(frozen=True)
class GovernancePackage:
    name: str
    indep_review_public: int
    timeline_specificity: int
    remuneration_clawback: int
    transparency_high: int
    ceo_change: int

    def feature_vector(self) -> Dict[str, int]:
        return {
            "indep_review": self.indep_review_public,
            "timeline": self.timeline_specificity,
            "clawback": self.remuneration_clawback,
            "transparency": self.transparency_high,
            "ceo_change": self.ceo_change,
        }


def default_packages() -> List[GovernancePackage]:
    # As specified by you
    return [
        GovernancePackage("D0 Minimal", 0, 0, 0, 0, 0),
        GovernancePackage("D1 Review-first", 1, 1, 0, 0, 0),
        GovernancePackage("D2 Accountability-lite", 0, 0, 1, 1, 0),
        GovernancePackage("D3 CEO transition", 1, 0, 0, 1, 1),
    ]


# ----------------------------
# Toy parameters (plausible defaults)
# ----------------------------

@dataclass
class ToyParams:
    # Latent belief dynamics
    rho: float = 0.85
    sigma_B: float = 0.35

    # Governance feature impacts on initial belief (negative reduces distrust)
    delta: Dict[str, float] = None

    # Exogenous monthly shock index X_t and its coefficient
    X: np.ndarray = None
    beta_X: float = 0.0  # zeroed: checkpoint posteriors already reflect info to date

    # Shareholder response (AGM opposition) mapping from B_agm
    alpha_rem: float = -0.35
    kappa_rem: float = 1.0
    alpha_dir: float = -0.90
    kappa_dir: float = 1.10

    # Exit pressure (negative = sell pressure) mapping from B_agm
    mu_exit: float = 0.00
    kappa_exit: float = -0.60
    sigma_exit: float = 0.40

    # Market CAR around month-1 governance announcement
    eta0: float = 0.002
    etaB: float = -0.012
    etaCEO: float = 0.010
    sigma_car: float = 0.012

    # Implementation costs (loss units)
    c_impl: Dict[str, float] = None

    # Loss weights and thresholds
    w_price: float = 40.0          # weight on downside CAR only
    w_strike: float = 4.0
    w_revolt: float = 2.5
    w_exit: float = 0.20           # weight on negative exit pressure only
    theta_strike: float = 0.25
    theta_revolt: float = 0.30

    # Initial belief baseline distribution (controversy intensity)
    B0_mean: float = 0.90
    B0_sd: float = 0.25

    def __post_init__(self):
        if self.delta is None:
            self.delta = {
                "indep_review": -0.50,
                "timeline": -0.35,
                "clawback": -0.40,
                "transparency": -0.30,
                "ceo_change": -0.80,
            }
        if self.X is None:
            # Mostly mild, with two spikes (toy)
            self.X = np.array([0.10, 0.05, 0.60, 0.10, 0.05, 0.00,
                               0.10, 0.55, 0.10, 0.05, 0.00, 0.05])
        if self.c_impl is None:
            self.c_impl = {
                "D0 Minimal": 0.10,
                "D1 Review-first": 0.25,
                "D2 Accountability-lite": 0.35,
                "D3 CEO transition": 0.70,
            }


# ----------------------------
# Core simulation engine
# ----------------------------

@dataclass
class SimulationOutputs:
    package_name: str
    loss: np.ndarray
    strike_event: np.ndarray
    revolt_event: np.ndarray
    car_m1: np.ndarray
    exit_m12: np.ndarray
    B_path: np.ndarray  # shape (n_sims, months)


class ARASimulator:
    """
    ARA simulator for Qantas governance packages.

    Designed so you can later replace ToyParams with posterior draws
    (rho, deltas, kappas, etc.) from an MCMC model with minimal changes.
    """

    def __init__(
        self,
        params: ToyParams,
        B0_draws: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.p = params
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.B0_draws = B0_draws

    def _governance_shift(self, pkg: GovernancePackage) -> float:
        f = pkg.feature_vector()
        return sum(self.p.delta[k] * v for k, v in f.items())

    def simulate_package(
        self,
        pkg: GovernancePackage,
        n_sims: int = 20000,
        months: int = 12,
        agm_month: int = 12,
        observed_y_rem: Optional[float] = None,
        observed_y_dir: Optional[float] = None,
    ) -> SimulationOutputs:
        """
        Simulate belief dynamics, shareholder responses, and loss for one package.

        agm_month is 1-indexed (12 means month 12).
        """
        if agm_month > months:
            raise ValueError(f"agm_month ({agm_month}) cannot exceed months ({months}).")

        # 1) Initial baseline belief (controversy intensity)
        if self.B0_draws is None:
            # fallback to toy prior
            B0 = self.rng.normal(self.p.B0_mean, self.p.B0_sd, size=n_sims)
        else:
            idx = self.rng.integers(0, len(self.B0_draws), size=n_sims)
            B0 = self.B0_draws[idx]

        # 2) Belief path
        B = np.zeros((n_sims, months))
        shift = self._governance_shift(pkg)

        # Month 1
        B[:, 0] = (
            self.p.rho * B0
            + self.p.beta_X * self.p.X[0]
            + shift
            + self.rng.normal(0.0, self.p.sigma_B, size=n_sims)
        )

        # Months 2..12
        for t in range(1, months):
            B[:, t] = (
                self.p.rho * B[:, t - 1]
                + shift
                + self.p.beta_X * self.p.X[t]
                + self.rng.normal(0.0, self.p.sigma_B, size=n_sims)
            )

        B1 = B[:, 0]
        B_agm = B[:, agm_month - 1]

        # 3) Shareholder opposition propensities at AGM
        p_rem = logistic(self.p.alpha_rem + self.p.kappa_rem * B_agm)
        p_dir = logistic(self.p.alpha_dir + self.p.kappa_dir * B_agm)

        # 4) Observed AGM "against %" (toy measurement noise)
        #    You can swap this to Beta regression later.
        def noisy_against(p: np.ndarray) -> np.ndarray:
            # more variance near 0.5 (toy)
            sd = 0.06 + 0.06 * (0.5 - np.abs(p - 0.5))
            y = self.rng.normal(p, sd, size=p.shape[0])
            return clip01(y)

        if observed_y_rem is None:
            y_rem = noisy_against(p_rem)
        else:
            y_rem = np.full(n_sims, float(observed_y_rem))

        if observed_y_dir is None:
            y_dir = noisy_against(p_dir)
        else:
            y_dir = np.full(n_sims, float(observed_y_dir))

        strike = (y_rem > self.p.theta_strike).astype(float)
        revolt = (y_dir > self.p.theta_revolt).astype(float)

        # 5) Market reaction around month-1 governance announcement
        ceo_change = pkg.ceo_change
        car = self.rng.normal(
            self.p.eta0 + self.p.etaB * B1 + self.p.etaCEO * ceo_change,
            self.p.sigma_car,
            size=n_sims,
        )

        # 6) Exit pressure in month 12 (negative = selling pressure)
        exit_m12 = self.rng.normal(
            self.p.mu_exit + self.p.kappa_exit * B_agm,
            self.p.sigma_exit,
            size=n_sims,
        )

        # 7) Loss function (penalise downside CAR and negative exit pressure)
        impl_cost = self.p.c_impl[pkg.name]
        loss = (
            impl_cost
            + self.p.w_price * np.maximum(-car, 0.0)
            + self.p.w_strike * strike
            + self.p.w_revolt * revolt
            + self.p.w_exit * np.maximum(-exit_m12, 0.0)
        )

        return SimulationOutputs(
            package_name=pkg.name,
            loss=loss,
            strike_event=strike,
            revolt_event=revolt,
            car_m1=car,
            exit_m12=exit_m12,
            B_path=B,
        )

    def simulate_all(
        self,
        packages: List[GovernancePackage],
        n_sims: int = 20000,
        months: int = 12,
        agm_month: int = 12,
        observed_y_rem: Optional[float] = None,
        observed_y_dir: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, SimulationOutputs]]:
        """
        Simulate all packages, return summary table and raw outputs per package.
        """
        outputs: Dict[str, SimulationOutputs] = {}
        rows = []

        for pkg in packages:
            out = self.simulate_package(
                pkg, n_sims=n_sims, months=months, agm_month=agm_month,
                observed_y_rem=observed_y_rem, observed_y_dir=observed_y_dir,
            )
            outputs[pkg.name] = out

            rows.append(
                {
                    "Package": pkg.name,
                    "Expected loss": float(out.loss.mean()),
                    "Loss 5%": float(np.quantile(out.loss, 0.05)),
                    "Loss 95%": float(np.quantile(out.loss, 0.95)),
                    "P(strike >25%)": float(out.strike_event.mean()),
                    "P(chair revolt >30%)": float(out.revolt_event.mean()),
                    "Mean CAR (m1)": float(out.car_m1.mean()),
                    "CAR 5%": float(np.quantile(out.car_m1, 0.05)),
                    "Mean exit pressure (m12)": float(out.exit_m12.mean()),
                }
            )

        df = pd.DataFrame(rows).sort_values("Expected loss").reset_index(drop=True)
        df["Rank"] = np.arange(1, len(df) + 1)
        return df, outputs


# ----------------------------
# Example usage
# ----------------------------

def main() -> None:
    packages = default_packages()
    params = ToyParams()

    # Checkpoint calendar: months-to-AGM for 2023 AGM on 03-Nov-2023
    #   C0 (01-Oct): ~1 month to AGM
    #   C1 (10-Oct): ~1 month to AGM
    #   C2 (18-Oct): ~1 month to AGM (half-month, round to 1)
    #   C3 (03-Nov): post-AGM — observed outcomes, no stochastic strike/revolt
    checkpoints = {
        "C0": {"path": "data/checkpoints/belief_C0_2023-10-01.npz", "months": 2, "agm_month": 2},
        "C1": {"path": "data/checkpoints/belief_C1_2023-10-10.npz", "months": 2, "agm_month": 2},
        "C2": {"path": "data/checkpoints/belief_C2_2023-10-18.npz", "months": 1, "agm_month": 1},
        "C3": {"path": "data/checkpoints/belief_C3_2023-11-03.npz", "months": 1, "agm_month": 1},
    }

    all_rows = []
    for ck, cfg in checkpoints.items():
        cp = np.load(cfg["path"])
        B0_draws = cp["B_mkt"]

        # sanity print: proves draws differ by checkpoint
        q = np.quantile(B0_draws, [0.05, 0.50, 0.95])
        print(f"\n=== {ck} | B0 mean={B0_draws.mean():.4f}  p05={q[0]:.4f}  p50={q[1]:.4f}  p95={q[2]:.4f} ===")

        if ck == "C3":
            # Post-AGM: strike/revolt already realised, not part of forward objective
            params_ck = ToyParams()
            params_ck.w_strike = 0.0
            params_ck.w_revolt = 0.0
            sim = ARASimulator(params_ck, B0_draws=B0_draws, rng=np.random.default_rng(42))
        else:
            sim = ARASimulator(params, B0_draws=B0_draws, rng=np.random.default_rng(42))

        summary, outputs = sim.simulate_all(
            packages, n_sims=20000,
            months=cfg["months"], agm_month=cfg["agm_month"],
            observed_y_rem=cfg.get("observed_y_rem"),
            observed_y_dir=cfg.get("observed_y_dir"),
        )

        # diagnostic: B_agm distribution per package
        for pkg_name, out in outputs.items():
            B_agm = out.B_path[:, cfg["agm_month"] - 1]
            bq = np.quantile(B_agm, [0.05, 0.50, 0.95])
            print(f"  {pkg_name:30s}  B_agm  mean={B_agm.mean():.3f}  p05={bq[0]:.3f}  p50={bq[1]:.3f}  p95={bq[2]:.3f}")

        # C3 post-AGM: strike/revolt are observed, not predicted — mark as NaN
        if ck == "C3":
            summary["P(strike >25%)"] = np.nan
            summary["P(chair revolt >30%)"] = np.nan

        print(summary.to_string(index=False))

        tmp = summary.copy()
        tmp.insert(0, "Checkpoint", ck)
        all_rows.append(tmp)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv("data/sim_summary_by_checkpoint.csv", index=False)
    print("\nWrote: data/sim_summary_by_checkpoint.csv")


if __name__ == "__main__":
    main()