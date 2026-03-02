# ara_qantas_governance.py
# Sequential stateful ARA simulator with multi-actor utilities,
# second-strike modelling, switching costs, and ASA response dynamics.
from __future__ import annotations

from dataclasses import dataclass, field
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


def _noisy_against(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate noisy vote-against proportions from latent probabilities."""
    sd = 0.06 + 0.06 * (0.5 - np.abs(p - 0.5))
    y = rng.normal(p, sd, size=p.shape[0])
    return clip01(y)


# ----------------------------
# Decision set D (4 packages)
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
    return [
        GovernancePackage("D0 Minimal", 0, 0, 0, 0, 0),
        GovernancePackage("D1 Review-first", 1, 1, 0, 0, 0),
        GovernancePackage("D2 Accountability-lite", 0, 0, 1, 1, 0),
        GovernancePackage("D3 CEO transition", 1, 0, 0, 1, 1),
    ]


# Ordinal strength for switching / escalation logic
PACKAGE_STRENGTH: Dict[str, int] = {
    "D0 Minimal": 0,
    "D1 Review-first": 1,
    "D2 Accountability-lite": 2,
    "D3 CEO transition": 3,
}


# ----------------------------
# ASA adversary action set
# ----------------------------

@dataclass(frozen=True)
class ASAAction:
    """A discrete action available to the ASA adversary."""
    name: str
    vote_logit_shift: float  # additive shift to rem-against vote logit
    cost: float              # cost to ASA of taking this action


def asa_action_set(params: "ToyParams", state: "DecisionState") -> List[ASAAction]:
    """Return the set of ASA actions available at this checkpoint."""
    if state.strike_count >= 1:  # C3: campaign intensity
        return [
            ASAAction("Low", params.asa_shift_low, params.asa_cost_low),
            ASAAction("Medium", params.asa_shift_medium, params.asa_cost_medium),
            ASAAction("High", params.asa_shift_high, params.asa_cost_high),
        ]
    else:  # C0/C1/C2: binary choice
        return [
            ASAAction("DoNothing", 0.0, params.asa_cost_do_nothing),
            ASAAction("RecommendStrike", params.asa_shift_recommend_strike,
                      params.asa_cost_recommend_strike),
        ]


# ----------------------------
# Sequential decision state
# ----------------------------

@dataclass
class DecisionState:
    """Carries sequential state across checkpoints."""
    checkpoint_id: str
    horizon_months: int
    agm_month: int
    strike_count: int = 0
    prev_action: Optional[str] = None
    V: float = 0.30
    B0_mkt_draws: Optional[np.ndarray] = None
    B0_mgmt_draws: Optional[np.ndarray] = None
    gamma_A_draws: Optional[np.ndarray] = None
    observed_y_rem: Optional[float] = None
    observed_y_dir: Optional[float] = None
    imminent_agm: bool = False


# ----------------------------
# Feasibility and ASA functions
# ----------------------------

def feasible_packages(
    packages: List[GovernancePackage],
    state: DecisionState,
) -> List[GovernancePackage]:
    """Return packages feasible given state. Near-AGM: no de-escalation."""
    if state.prev_action is None:
        return list(packages)
    prev_strength = PACKAGE_STRENGTH[state.prev_action]
    if state.imminent_agm:
        return [p for p in packages if PACKAGE_STRENGTH[p.name] >= prev_strength]
    return list(packages)


def compute_gamma_asa_eff(
    params: "ToyParams",
    state: DecisionState,
    pkg: GovernancePackage,
    rng: np.random.Generator,
) -> float:
    """Compute effective ASA mobilisation multiplier for gamma_A draws."""
    eff = params.asa_base_gamma

    if PACKAGE_STRENGTH[pkg.name] == 0:
        eff += params.asa_weak_action_boost

    if state.prev_action is not None:
        if PACKAGE_STRENGTH[pkg.name] < PACKAGE_STRENGTH[state.prev_action]:
            eff += params.asa_backtrack_boost

    if state.strike_count >= 1:
        eff += params.asa_strike1_boost

    if state.imminent_agm:
        eff += params.asa_imminent_agm_boost

    eff += rng.normal(0.0, params.asa_noise_sigma)
    return max(eff, 0.0)


# ----------------------------
# Parameters
# ----------------------------

@dataclass
class ToyParams:
    # --- Latent belief dynamics (legacy single-parameter defaults) ---
    rho: float = 0.85
    sigma_B: float = 0.35

    # Governance feature impacts on belief (negative reduces distrust)
    delta: Dict[str, float] = None

    # Exogenous monthly shock index X_t and its coefficient
    X: np.ndarray = None
    beta_X: float = 0.0

    # Shareholder response (AGM opposition) mapping from B_agm
    alpha_rem: float = -0.35
    kappa_rem: float = 1.0
    alpha_dir: float = -0.90
    kappa_dir: float = 1.10

    # Exit pressure mapping from B_agm
    mu_exit: float = 0.00
    kappa_exit: float = -0.60
    sigma_exit: float = 0.40

    # Market CAR (base, non-CEO terms)
    eta0: float = 0.002
    etaB: float = -0.012
    etaCEO: float = 0.010  # legacy
    sigma_car: float = 0.012

    # Implementation costs (loss units)
    c_impl: Dict[str, float] = None

    # Market loss weights and thresholds
    w_price: float = 40.0
    w_strike: float = 4.0
    w_revolt: float = 2.5
    w_exit: float = 0.20
    theta_strike: float = 0.25
    theta_revolt: float = 0.30

    # Initial belief baseline (fallback)
    B0_mean: float = 0.90
    B0_sd: float = 0.25

    # --- Split belief dynamics ---
    rho_mkt: Optional[float] = None
    sigma_B_mkt: Optional[float] = None
    rho_mgmt: Optional[float] = None
    sigma_B_mgmt: Optional[float] = None

    # --- Management overconfidence ---
    mgmt_bias: float = 0.90
    mgmt_bias_sigma: float = 0.12

    # --- Split governance deltas ---
    delta_mkt: Optional[Dict[str, float]] = None
    delta_mgmt: Optional[Dict[str, float]] = None

    # --- Board deference ---
    V: float = 0.30

    # --- CEO transition probability ---
    ceo_transition_intercept: float = 1.8
    ceo_transition_slope_Bmgmt: float = 0.6
    ceo_transition_slope_V: float = 1.5

    # --- CEO mode selection: resign(1) vs sacked(2) ---
    ceo_mode_intercept: float = -2.4
    ceo_mode_slope_Bmgmt: float = 0.6
    ceo_mode_slope_V: float = 1.8

    # --- Market perception by CEO mode ---
    eta_ceo_resign: float = 0.012
    eta_ceo_sacked: float = -0.006

    # --- Belief shifts by CEO mode ---
    delta_ceo_resign_mkt: float = -1.0
    delta_ceo_sacked_mkt: float = -0.6

    # --- Management agency costs ---
    w_ceo_resign_mgmt: float = 0.8
    w_ceo_sacked_mgmt: float = 2.4

    # --- Second-strike / spill model ---
    theta_strike2: float = 0.25
    theta_spill: float = 0.50
    w_strike2: float = 8.0
    w_spill: float = 15.0

    # --- Board utility weights ---
    w_price_board: float = 30.0
    w_strike_board: float = 6.0
    w_revolt_board: float = 3.0
    w_exit_board: float = 0.15
    # Board loyalty: direct penalty for choosing D3 (CEO removal)
    w_ceo_removal_board: float = 3.0
    # Board overconfidence: negative shift to perceived distrust for strike evaluation
    board_optimism_shift: float = -0.8

    # --- CEO utility weights ---
    w_price_ceo: float = 10.0
    w_impl_ceo_multiplier: float = 0.5

    # --- Switching cost matrix ---
    c_switch: Optional[Dict[str, Dict[str, float]]] = None

    # --- ASA response modifiers ---
    asa_base_gamma: float = 1.0
    asa_weak_action_boost: float = 0.5
    asa_backtrack_boost: float = 1.0
    asa_strike1_boost: float = 0.6
    asa_imminent_agm_boost: float = 0.3
    asa_noise_sigma: float = 0.15

    # --- ASA adversarial actions (Stackelberg layer) ---
    # C0-C2: binary action vote logit shifts
    asa_shift_recommend_strike: float = 0.5
    # C3: campaign intensity vote logit shifts
    asa_shift_low: float = 0.2
    asa_shift_medium: float = 0.5
    asa_shift_high: float = 0.9
    # ASA action costs
    asa_cost_do_nothing: float = 0.0
    asa_cost_recommend_strike: float = 0.3
    asa_cost_low: float = 0.1
    asa_cost_medium: float = 0.4
    asa_cost_high: float = 0.8
    # ASA utility weights (benefit from outcomes)
    u_asa_strike1: float = 2.0
    u_asa_strike2: float = 3.0
    u_asa_spill: float = 5.0
    # Probabilistic spill model (replaces threshold in adversarial mode)
    p_spill_given_strike2: float = 0.35

    # --- De-escalation distrust penalties ---
    backtrack_distrust_shift: float = 0.4
    backtrack_mgmt_shift: float = 0.1

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
            self.X = np.array([0.10, 0.05, 0.60, 0.10, 0.05, 0.00,
                               0.10, 0.55, 0.10, 0.05, 0.00, 0.05])
        if self.c_impl is None:
            self.c_impl = {
                "D0 Minimal": 0.10,
                "D1 Review-first": 0.25,
                "D2 Accountability-lite": 0.35,
                "D3 CEO transition": 0.70,
            }

        # Split belief dynamics
        if self.rho_mkt is None:
            self.rho_mkt = self.rho
        if self.sigma_B_mkt is None:
            self.sigma_B_mkt = self.sigma_B
        if self.rho_mgmt is None:
            self.rho_mgmt = self.rho
        if self.sigma_B_mgmt is None:
            self.sigma_B_mgmt = self.sigma_B

        # Split governance deltas
        if self.delta_mkt is None:
            self.delta_mkt = dict(self.delta)
        if self.delta_mgmt is None:
            self.delta_mgmt = {
                "indep_review": 1.2 * self.delta["indep_review"],
                "timeline":     1.2 * self.delta["timeline"],
                "clawback":     0.7 * self.delta["clawback"],
                "transparency": 1.1 * self.delta["transparency"],
                "ceo_change":   self.delta["ceo_change"],
            }

        # Switching cost matrix
        if self.c_switch is None:
            self.c_switch = {
                "D0 Minimal": {
                    "D0 Minimal": 0.0,
                    "D1 Review-first": 0.15,
                    "D2 Accountability-lite": 0.20,
                    "D3 CEO transition": 0.30,
                },
                "D1 Review-first": {
                    "D0 Minimal": 3.0,
                    "D1 Review-first": 0.0,
                    "D2 Accountability-lite": 0.80,
                    "D3 CEO transition": 0.60,
                },
                "D2 Accountability-lite": {
                    "D0 Minimal": 3.5,
                    "D1 Review-first": 2.0,
                    "D2 Accountability-lite": 0.0,
                    "D3 CEO transition": 0.50,
                },
                "D3 CEO transition": {
                    "D0 Minimal": 5.0,
                    "D1 Review-first": 4.0,
                    "D2 Accountability-lite": 3.0,
                    "D3 CEO transition": 0.0,
                },
            }


# ----------------------------
# Simulation outputs
# ----------------------------

@dataclass
class SimulationOutputs:
    package_name: str
    # --- Core arrays ---
    loss: np.ndarray               # L_market (n_sims,)
    strike_event: np.ndarray       # first strike indicator (n_sims,)
    revolt_event: np.ndarray       # (n_sims,)
    car_m1: np.ndarray             # (n_sims,)
    exit_m12: np.ndarray           # (n_sims,)
    B_mkt_path: np.ndarray         # (n_sims, months)
    B_mgmt_path: np.ndarray        # (n_sims, months)
    ceo_mode: np.ndarray           # (n_sims,) 0=stay, 1=resign, 2=sacked
    loss_mgmt: np.ndarray          # (n_sims,) legacy management loss

    # --- Multi-actor losses ---
    loss_board: Optional[np.ndarray] = None   # L_board (n_sims,)
    loss_ceo: Optional[np.ndarray] = None     # L_ceo (n_sims,)

    # --- Second-strike / spill ---
    strike2_event: Optional[np.ndarray] = None  # (n_sims,)
    spill_event: Optional[np.ndarray] = None    # (n_sims,)

    # --- Loss decomposition ---
    impl_cost_component: Optional[np.ndarray] = None
    car_penalty_component: Optional[np.ndarray] = None
    strike1_penalty_component: Optional[np.ndarray] = None
    revolt_penalty_component: Optional[np.ndarray] = None
    exit_penalty_component: Optional[np.ndarray] = None
    strike2_penalty_component: Optional[np.ndarray] = None
    spill_penalty_component: Optional[np.ndarray] = None
    switch_cost_component: Optional[np.ndarray] = None
    ceo_agency_component: Optional[np.ndarray] = None

    # --- ASA effective gamma ---
    gamma_asa_eff: Optional[float] = None

    # --- Adversarial layer ---
    asa_utility: Optional[np.ndarray] = None   # U_ASA (n_sims,)
    asa_action_name: Optional[str] = None      # name of ASA action used
    ceo_action: Optional[str] = None           # CEO strategic action used


# ----------------------------
# Core simulation engine
# ----------------------------

class ARASimulator:
    """
    ARA simulator for Qantas governance packages.

    Supports split market/management beliefs, CEO transition mode,
    board deference, sequential state, and multi-actor utilities.
    """

    def __init__(
        self,
        params: ToyParams,
        B0_draws: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        B0_mgmt_draws: Optional[np.ndarray] = None,
    ):
        self.p = params
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.B0_mkt_draws = B0_draws
        self.B0_mgmt_draws = B0_mgmt_draws

    def _draw_B0_pair(
        self, n_sims: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Draw initial market and management beliefs. Returns (B0_mkt, B0_mgmt, idx)."""
        if self.B0_mkt_draws is None:
            B0_mkt = self.rng.normal(self.p.B0_mean, self.p.B0_sd, size=n_sims)
            bias = self.rng.normal(self.p.mgmt_bias, self.p.mgmt_bias_sigma, size=n_sims)
            B0_mgmt = B0_mkt - bias
            return B0_mkt, B0_mgmt, None
        else:
            idx = self.rng.integers(0, len(self.B0_mkt_draws), size=n_sims)
            B0_mkt = self.B0_mkt_draws[idx]
            if self.B0_mgmt_draws is not None:
                B0_mgmt = self.B0_mgmt_draws[idx]
            else:
                bias = self.rng.normal(self.p.mgmt_bias, self.p.mgmt_bias_sigma, size=n_sims)
                B0_mgmt = B0_mkt - bias
            return B0_mkt, B0_mgmt, idx

    def _governance_shift_pair(self, pkg: GovernancePackage) -> Tuple[float, float]:
        """Governance-induced belief shifts. CEO change excluded (handled by mode system)."""
        f = pkg.feature_vector()
        shift_mkt = 0.0
        shift_mgmt = 0.0
        for k, v in f.items():
            if k == "ceo_change":
                continue
            shift_mkt += self.p.delta_mkt[k] * v
            shift_mgmt += self.p.delta_mgmt[k] * v
        return shift_mkt, shift_mgmt

    def _simulate_belief_paths(
        self,
        B0_mkt: np.ndarray,
        B0_mgmt: np.ndarray,
        shift_mkt: float,
        shift_mgmt: float,
        months: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve two belief paths with separate AR(1) dynamics."""
        n = B0_mkt.size
        Bm = np.zeros((n, months))
        Bg = np.zeros((n, months))

        Bm[:, 0] = (
            self.p.rho_mkt * B0_mkt
            + self.p.beta_X * self.p.X[0]
            + shift_mkt
            + self.rng.normal(0.0, self.p.sigma_B_mkt, size=n)
        )
        Bg[:, 0] = (
            self.p.rho_mgmt * B0_mgmt
            + self.p.beta_X * self.p.X[0]
            + shift_mgmt
            + self.rng.normal(0.0, self.p.sigma_B_mgmt, size=n)
        )

        for t in range(1, months):
            Bm[:, t] = (
                self.p.rho_mkt * Bm[:, t - 1]
                + shift_mkt
                + self.p.beta_X * self.p.X[t]
                + self.rng.normal(0.0, self.p.sigma_B_mkt, size=n)
            )
            Bg[:, t] = (
                self.p.rho_mgmt * Bg[:, t - 1]
                + shift_mgmt
                + self.p.beta_X * self.p.X[t]
                + self.rng.normal(0.0, self.p.sigma_B_mgmt, size=n)
            )

        return Bm, Bg

    def _simulate_ceo_mode(
        self, pkg: GovernancePackage, B_mgmt_agm: np.ndarray
    ) -> np.ndarray:
        """Sample CEO transition mode. Returns 0=stay, 1=resign, 2=sacked."""
        n = B_mgmt_agm.size
        mode = np.zeros(n, dtype=int)

        if not pkg.ceo_change:
            return mode

        logit_trans = (
            self.p.ceo_transition_intercept
            + self.p.ceo_transition_slope_Bmgmt * B_mgmt_agm
            + self.p.ceo_transition_slope_V * self.p.V
        )
        p_trans = logistic(logit_trans)
        trans = self.rng.uniform(0, 1, size=n) < p_trans

        logit_sacked = (
            self.p.ceo_mode_intercept
            + self.p.ceo_mode_slope_Bmgmt * B_mgmt_agm
            + self.p.ceo_mode_slope_V * self.p.V
        )
        p_sacked = logistic(logit_sacked)
        sacked = self.rng.uniform(0, 1, size=n) < p_sacked

        mode[trans & ~sacked] = 1
        mode[trans & sacked] = 2

        return mode

    # ------------------------------------------------------------------
    # Legacy interface (retained for backward compatibility)
    # ------------------------------------------------------------------

    def simulate_package(
        self,
        pkg: GovernancePackage,
        n_sims: int = 20000,
        months: int = 12,
        agm_month: int = 12,
        observed_y_rem: Optional[float] = None,
        observed_y_dir: Optional[float] = None,
    ) -> SimulationOutputs:
        """Legacy single-shot simulation (no state, no multi-actor losses)."""
        if agm_month > months:
            raise ValueError(f"agm_month ({agm_month}) cannot exceed months ({months}).")

        B0_mkt, B0_mgmt, _idx = self._draw_B0_pair(n_sims)
        shift_mkt, shift_mgmt = self._governance_shift_pair(pkg)
        Bm, Bg = self._simulate_belief_paths(B0_mkt, B0_mgmt, shift_mkt, shift_mgmt, months)

        B_mkt_1 = Bm[:, 0]
        B_mkt_agm = Bm[:, agm_month - 1].copy()
        B_mgmt_agm = Bg[:, agm_month - 1]

        ceo_mode = self._simulate_ceo_mode(pkg, B_mgmt_agm)
        B_mkt_agm[ceo_mode == 1] += self.p.delta_ceo_resign_mkt
        B_mkt_agm[ceo_mode == 2] += self.p.delta_ceo_sacked_mkt

        p_rem = logistic(self.p.alpha_rem + self.p.kappa_rem * B_mkt_agm)
        p_dir = logistic(self.p.alpha_dir + self.p.kappa_dir * B_mkt_agm)

        y_rem = _noisy_against(p_rem, self.rng) if observed_y_rem is None \
            else np.full(n_sims, float(observed_y_rem))
        y_dir = _noisy_against(p_dir, self.rng) if observed_y_dir is None \
            else np.full(n_sims, float(observed_y_dir))

        strike = (y_rem > self.p.theta_strike).astype(float)
        revolt = (y_dir > self.p.theta_revolt).astype(float)

        mu_car = np.asarray(self.p.eta0 + self.p.etaB * B_mkt_1, dtype=float).copy()
        mu_car[ceo_mode == 1] += self.p.eta_ceo_resign
        mu_car[ceo_mode == 2] += self.p.eta_ceo_sacked
        car = self.rng.normal(mu_car, self.p.sigma_car, size=n_sims)

        exit_m12 = self.rng.normal(
            self.p.mu_exit + self.p.kappa_exit * B_mkt_agm,
            self.p.sigma_exit, size=n_sims,
        )

        impl_cost = self.p.c_impl[pkg.name]
        loss_mkt = (
            impl_cost
            + self.p.w_price * np.maximum(-car, 0.0)
            + self.p.w_strike * strike
            + self.p.w_revolt * revolt
            + self.p.w_exit * np.maximum(-exit_m12, 0.0)
        )

        loss_mgmt = loss_mkt.copy()
        loss_mgmt[ceo_mode == 1] += self.p.w_ceo_resign_mgmt
        loss_mgmt[ceo_mode == 2] += self.p.w_ceo_sacked_mgmt

        return SimulationOutputs(
            package_name=pkg.name,
            loss=loss_mkt,
            strike_event=strike,
            revolt_event=revolt,
            car_m1=car,
            exit_m12=exit_m12,
            B_mkt_path=Bm,
            B_mgmt_path=Bg,
            ceo_mode=ceo_mode,
            loss_mgmt=loss_mgmt,
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
        """Legacy: simulate all packages independently."""
        outputs: Dict[str, SimulationOutputs] = {}
        rows = []
        for pkg in packages:
            out = self.simulate_package(
                pkg, n_sims=n_sims, months=months, agm_month=agm_month,
                observed_y_rem=observed_y_rem, observed_y_dir=observed_y_dir,
            )
            outputs[pkg.name] = out
            agm_idx = agm_month - 1
            rows.append({
                "Package": pkg.name,
                "Expected loss": float(out.loss.mean()),
                "Loss 5%": float(np.quantile(out.loss, 0.05)),
                "Loss 95%": float(np.quantile(out.loss, 0.95)),
                "P(strike >25%)": float(out.strike_event.mean()),
                "P(chair revolt >30%)": float(out.revolt_event.mean()),
                "Mean CAR (m1)": float(out.car_m1.mean()),
                "CAR 5%": float(np.quantile(out.car_m1, 0.05)),
                "Mean exit pressure (m12)": float(out.exit_m12.mean()),
                "Expected loss (mgmt)": float(out.loss_mgmt.mean()),
                "Agency gap": float(out.loss.mean() - out.loss_mgmt.mean()),
                "P(CEO resign)": float((out.ceo_mode == 1).mean()),
                "P(CEO sacked)": float((out.ceo_mode == 2).mean()),
                "Mean B_mkt_agm": float(out.B_mkt_path[:, agm_idx].mean()),
                "Mean B_mgmt_agm": float(out.B_mgmt_path[:, agm_idx].mean()),
                "Belief gap": float(
                    out.B_mgmt_path[:, agm_idx].mean()
                    - out.B_mkt_path[:, agm_idx].mean()
                ),
            })
        df = pd.DataFrame(rows).sort_values("Expected loss").reset_index(drop=True)
        df["Rank"] = np.arange(1, len(df) + 1)
        return df, outputs

    # ------------------------------------------------------------------
    # Stateful interface (sequential decision problem)
    # ------------------------------------------------------------------

    def simulate_package_stateful(
        self,
        pkg: GovernancePackage,
        state: DecisionState,
        n_sims: int = 20000,
    ) -> SimulationOutputs:
        """
        Simulate one package within a sequential decision context.

        Uses state for: horizon, strike_count, prev_action, ASA effectiveness,
        switching costs, second-strike/spill outcomes, multi-actor losses.
        """
        months = state.horizon_months
        agm_month = state.agm_month

        if agm_month > months:
            raise ValueError(f"agm_month ({agm_month}) > months ({months})")

        # 1) Draw initial beliefs + resampling index
        B0_mkt, B0_mgmt, resample_idx = self._draw_B0_pair(n_sims)

        # 2) Backtrack distrust penalty
        is_backtrack = (
            state.prev_action is not None
            and PACKAGE_STRENGTH[pkg.name] < PACKAGE_STRENGTH[state.prev_action]
        )
        if is_backtrack:
            B0_mkt = B0_mkt + self.p.backtrack_distrust_shift
            B0_mgmt = B0_mgmt + self.p.backtrack_mgmt_shift

        # 3) Governance shifts (excludes CEO change)
        shift_mkt, shift_mgmt = self._governance_shift_pair(pkg)

        # 4) Belief paths
        Bm, Bg = self._simulate_belief_paths(B0_mkt, B0_mgmt, shift_mkt, shift_mgmt, months)

        B_mkt_1 = Bm[:, 0]
        B_mkt_agm = Bm[:, agm_month - 1].copy()
        B_mgmt_agm = Bg[:, agm_month - 1]

        # 5) CEO mode
        ceo_mode = self._simulate_ceo_mode(pkg, B_mgmt_agm)

        # 6) CEO mode belief shift (market)
        B_mkt_agm[ceo_mode == 1] += self.p.delta_ceo_resign_mkt
        B_mkt_agm[ceo_mode == 2] += self.p.delta_ceo_sacked_mkt

        # 7) ASA effective gamma
        gamma_asa_eff = compute_gamma_asa_eff(self.p, state, pkg, self.rng)

        # 8) Resample gamma_A draws to match sim draws
        if state.gamma_A_draws is not None and resample_idx is not None:
            gamma_A_sim = state.gamma_A_draws[resample_idx % len(state.gamma_A_draws)]
        elif state.gamma_A_draws is not None:
            gamma_A_sim = self.rng.choice(state.gamma_A_draws, size=n_sims, replace=True)
        else:
            gamma_A_sim = np.zeros(n_sims)

        # 9) Vote probabilities with ASA effect
        asa_effect = gamma_asa_eff * gamma_A_sim
        p_rem = logistic(self.p.alpha_rem + self.p.kappa_rem * B_mkt_agm + asa_effect)
        p_dir = logistic(self.p.alpha_dir + self.p.kappa_dir * B_mkt_agm)

        # 10) Vote draws
        y_rem = _noisy_against(p_rem, self.rng) if state.observed_y_rem is None \
            else np.full(n_sims, float(state.observed_y_rem))
        y_dir = _noisy_against(p_dir, self.rng) if state.observed_y_dir is None \
            else np.full(n_sims, float(state.observed_y_dir))

        # 11) Strike / revolt events
        strike = (y_rem > self.p.theta_strike).astype(float)
        revolt = (y_dir > self.p.theta_revolt).astype(float)

        # 12) Second-strike and spill (when strike_count >= 1)
        if state.strike_count >= 1:
            strike2 = strike.copy()  # this strike IS strike 2
            spill = (y_rem > self.p.theta_spill).astype(float)
        else:
            strike2 = np.zeros(n_sims)
            spill = np.zeros(n_sims)

        # 13) Market CAR
        mu_car = np.asarray(self.p.eta0 + self.p.etaB * B_mkt_1, dtype=float).copy()
        mu_car[ceo_mode == 1] += self.p.eta_ceo_resign
        mu_car[ceo_mode == 2] += self.p.eta_ceo_sacked
        car = self.rng.normal(mu_car, self.p.sigma_car, size=n_sims)

        # 14) Exit pressure
        exit_m = self.rng.normal(
            self.p.mu_exit + self.p.kappa_exit * B_mkt_agm,
            self.p.sigma_exit, size=n_sims,
        )

        # 15) Switching cost
        if state.prev_action is not None:
            switch_cost = self.p.c_switch[state.prev_action][pkg.name]
        else:
            switch_cost = 0.0

        # ===== LOSS DECOMPOSITION =====
        impl_cost = self.p.c_impl[pkg.name]
        car_penalty = self.p.w_price * np.maximum(-car, 0.0)
        exit_penalty = self.p.w_exit * np.maximum(-exit_m, 0.0)

        # --- L_market ---
        if state.strike_count == 0:
            strike1_penalty_mkt = self.p.w_strike * strike
            revolt_penalty_mkt = self.p.w_revolt * revolt
            strike2_penalty_mkt = np.zeros(n_sims)
            spill_penalty_mkt = np.zeros(n_sims)
        else:
            strike1_penalty_mkt = np.zeros(n_sims)
            revolt_penalty_mkt = np.zeros(n_sims)
            strike2_penalty_mkt = self.p.w_strike2 * strike2
            spill_penalty_mkt = self.p.w_spill * spill

        loss_mkt = (
            impl_cost + car_penalty + strike1_penalty_mkt + revolt_penalty_mkt
            + exit_penalty + strike2_penalty_mkt + spill_penalty_mkt
        )

        # --- L_board ---
        car_penalty_board = self.p.w_price_board * np.maximum(-car, 0.0)
        exit_penalty_board = self.p.w_exit_board * np.maximum(-exit_m, 0.0)

        if state.strike_count == 0:
            strike_penalty_board = self.p.w_strike_board * strike
            revolt_penalty_board = self.p.w_revolt_board * revolt
            strike2_penalty_board = np.zeros(n_sims)
            spill_penalty_board = np.zeros(n_sims)
        else:
            strike_penalty_board = np.zeros(n_sims)
            revolt_penalty_board = np.zeros(n_sims)
            strike2_penalty_board = self.p.w_strike2 * strike2
            spill_penalty_board = self.p.w_spill * spill

        loss_board = (
            impl_cost + car_penalty_board + strike_penalty_board + revolt_penalty_board
            + exit_penalty_board + strike2_penalty_board + spill_penalty_board
            + switch_cost
        )

        # --- L_ceo ---
        car_penalty_ceo = self.p.w_price_ceo * np.maximum(-car, 0.0)
        loss_ceo = self.p.w_impl_ceo_multiplier * impl_cost + car_penalty_ceo
        ceo_agency = np.zeros(n_sims)
        ceo_agency[ceo_mode == 1] = self.p.w_ceo_resign_mgmt
        ceo_agency[ceo_mode == 2] = self.p.w_ceo_sacked_mgmt
        loss_ceo = loss_ceo + ceo_agency

        # Legacy management loss
        loss_mgmt = loss_mkt.copy()
        loss_mgmt[ceo_mode == 1] += self.p.w_ceo_resign_mgmt
        loss_mgmt[ceo_mode == 2] += self.p.w_ceo_sacked_mgmt

        return SimulationOutputs(
            package_name=pkg.name,
            loss=loss_mkt,
            strike_event=strike,
            revolt_event=revolt,
            car_m1=car,
            exit_m12=exit_m,
            B_mkt_path=Bm,
            B_mgmt_path=Bg,
            ceo_mode=ceo_mode,
            loss_mgmt=loss_mgmt,
            loss_board=loss_board,
            loss_ceo=loss_ceo,
            strike2_event=strike2,
            spill_event=spill,
            impl_cost_component=np.full(n_sims, impl_cost),
            car_penalty_component=car_penalty,
            strike1_penalty_component=strike1_penalty_mkt,
            revolt_penalty_component=revolt_penalty_mkt,
            exit_penalty_component=exit_penalty,
            strike2_penalty_component=strike2_penalty_mkt,
            spill_penalty_component=spill_penalty_mkt,
            switch_cost_component=np.full(n_sims, switch_cost),
            ceo_agency_component=ceo_agency,
            gamma_asa_eff=gamma_asa_eff,
        )

    def simulate_all_stateful(
        self,
        packages: List[GovernancePackage],
        state: DecisionState,
        n_sims: int = 20000,
    ) -> Tuple[pd.DataFrame, Dict[str, SimulationOutputs]]:
        """Simulate all feasible packages given state. Returns summary and raw outputs."""
        feasible = feasible_packages(packages, state)
        outputs: Dict[str, SimulationOutputs] = {}
        rows = []

        for pkg in feasible:
            out = self.simulate_package_stateful(pkg, state, n_sims=n_sims)
            outputs[pkg.name] = out

            agm_idx = state.agm_month - 1
            rows.append({
                "Checkpoint": state.checkpoint_id,
                "Package": pkg.name,
                "Feasible": True,

                # Multi-actor expected losses
                "E[L_market]": float(out.loss.mean()),
                "E[L_board]": float(out.loss_board.mean()),
                "E[L_ceo]": float(out.loss_ceo.mean()),

                # Loss quantiles (market)
                "L_market 5%": float(np.quantile(out.loss, 0.05)),
                "L_market 95%": float(np.quantile(out.loss, 0.95)),

                # Decomposition
                "E[impl_cost]": float(out.impl_cost_component.mean()),
                "E[car_penalty]": float(out.car_penalty_component.mean()),
                "E[strike1_penalty]": float(out.strike1_penalty_component.mean()),
                "E[revolt_penalty]": float(out.revolt_penalty_component.mean()),
                "E[exit_penalty]": float(out.exit_penalty_component.mean()),
                "E[strike2_penalty]": float(out.strike2_penalty_component.mean()),
                "E[spill_penalty]": float(out.spill_penalty_component.mean()),
                "E[switch_cost]": float(out.switch_cost_component.mean()),
                "E[ceo_agency]": float(out.ceo_agency_component.mean()),

                # Probabilities
                "P(strike1)": float(out.strike_event.mean()),
                "P(revolt)": float(out.revolt_event.mean()),
                "P(strike2)": float(out.strike2_event.mean()),
                "P(spill)": float(out.spill_event.mean()),

                # Market observables
                "Mean CAR (m1)": float(out.car_m1.mean()),
                "CAR 5%": float(np.quantile(out.car_m1, 0.05)),
                "Mean exit pressure": float(out.exit_m12.mean()),

                # CEO mode
                "P(CEO stay)": float((out.ceo_mode == 0).mean()),
                "P(CEO resign)": float((out.ceo_mode == 1).mean()),
                "P(CEO sacked)": float((out.ceo_mode == 2).mean()),

                # Beliefs
                "Mean B_mkt_agm": float(out.B_mkt_path[:, agm_idx].mean()),
                "Mean B_mgmt_agm": float(out.B_mgmt_path[:, agm_idx].mean()),
                "Belief gap": float(
                    out.B_mgmt_path[:, agm_idx].mean()
                    - out.B_mkt_path[:, agm_idx].mean()
                ),

                # ASA and context
                "gamma_ASA_eff": out.gamma_asa_eff,
                "strike_count": state.strike_count,
                "prev_action": state.prev_action or "",
            })

        df = pd.DataFrame(rows)

        if len(df) > 0:
            df["market_optimal"] = df["E[L_market]"] == df["E[L_market]"].min()
            df["board_optimal"] = df["E[L_board]"] == df["E[L_board]"].min()
            df["ceo_preferred"] = df["E[L_ceo]"] == df["E[L_ceo]"].min()

        return df, outputs

    # ------------------------------------------------------------------
    # Adversarial Stackelberg layer
    # ------------------------------------------------------------------

    def simulate_DA(
        self,
        pkg: GovernancePackage,
        asa_action: ASAAction,
        state: DecisionState,
        n_sims: int = 20000,
        ceo_action: str = "Stay",
    ) -> SimulationOutputs:
        """
        Simulate one (D, A, CEO_action) triple in the Stackelberg game.

        Key differences from simulate_package_stateful:
        - ASA action provides a deterministic vote-logit shift (replaces gamma_asa_eff)
        - Spill model is probabilistic: Bernoulli(p_spill_given_strike2) * strike2
        - Board overconfidence: Board evaluates strike/revolt through shifted beliefs
        - Board loyalty: CEO removal penalty added to L_board when pkg.ceo_change=1
        - CEO strategic action: "Stay" uses stochastic mode, "Resign" forces mode=1
        - Computes and stores ASA utility
        """
        months = state.horizon_months
        agm_month = state.agm_month

        if agm_month > months:
            raise ValueError(f"agm_month ({agm_month}) > months ({months})")

        # 1) Draw initial beliefs + resampling index
        B0_mkt, B0_mgmt, resample_idx = self._draw_B0_pair(n_sims)

        # 2) Backtrack distrust penalty
        is_backtrack = (
            state.prev_action is not None
            and PACKAGE_STRENGTH[pkg.name] < PACKAGE_STRENGTH[state.prev_action]
        )
        if is_backtrack:
            B0_mkt = B0_mkt + self.p.backtrack_distrust_shift
            B0_mgmt = B0_mgmt + self.p.backtrack_mgmt_shift

        # 3) Governance shifts (excludes CEO change)
        shift_mkt, shift_mgmt = self._governance_shift_pair(pkg)

        # 4) Belief paths
        Bm, Bg = self._simulate_belief_paths(B0_mkt, B0_mgmt, shift_mkt, shift_mgmt, months)

        B_mkt_1 = Bm[:, 0]
        B_mkt_agm = Bm[:, agm_month - 1].copy()
        B_mgmt_agm = Bg[:, agm_month - 1]

        # 5) CEO mode: strategic action overrides stochastic model
        if ceo_action == "Resign":
            ceo_mode = np.ones(n_sims, dtype=int)  # forced resignation
        else:
            ceo_mode = self._simulate_ceo_mode(pkg, B_mgmt_agm)

        # 6) CEO mode belief shift (market)
        B_mkt_agm[ceo_mode == 1] += self.p.delta_ceo_resign_mkt
        B_mkt_agm[ceo_mode == 2] += self.p.delta_ceo_sacked_mkt

        # 7) Actual vote probabilities (market truth): ASA logit shift
        p_rem = logistic(
            self.p.alpha_rem + self.p.kappa_rem * B_mkt_agm
            + asa_action.vote_logit_shift
        )
        p_dir = logistic(self.p.alpha_dir + self.p.kappa_dir * B_mkt_agm)

        # 8) Vote draws (actual outcomes)
        y_rem = _noisy_against(p_rem, self.rng) if state.observed_y_rem is None \
            else np.full(n_sims, float(state.observed_y_rem))
        y_dir = _noisy_against(p_dir, self.rng) if state.observed_y_dir is None \
            else np.full(n_sims, float(state.observed_y_dir))

        # 9) Strike / revolt events (actual)
        strike = (y_rem > self.p.theta_strike).astype(float)
        revolt = (y_dir > self.p.theta_revolt).astype(float)

        # 9b) Board-PERCEIVED strike/revolt (overconfident: shifted beliefs)
        B_board_agm = B_mkt_agm + self.p.board_optimism_shift
        p_rem_board = logistic(
            self.p.alpha_rem + self.p.kappa_rem * B_board_agm
            + asa_action.vote_logit_shift
        )
        p_dir_board = logistic(self.p.alpha_dir + self.p.kappa_dir * B_board_agm)
        y_rem_board = _noisy_against(p_rem_board, self.rng)
        y_dir_board = _noisy_against(p_dir_board, self.rng)
        strike_board = (y_rem_board > self.p.theta_strike).astype(float)
        revolt_board = (y_dir_board > self.p.theta_revolt).astype(float)

        # 10) Second-strike and spill (probabilistic spill model)
        if state.strike_count >= 1:
            strike2 = strike.copy()  # actual strike2
            spill_draw = self.rng.uniform(0, 1, size=n_sims) < self.p.p_spill_given_strike2
            spill = (strike2 * spill_draw.astype(float))
            # Board-perceived
            strike2_board = strike_board.copy()
            spill_board = (strike2_board * spill_draw.astype(float))
        else:
            strike2 = np.zeros(n_sims)
            spill = np.zeros(n_sims)
            strike2_board = np.zeros(n_sims)
            spill_board = np.zeros(n_sims)

        # 11) Market CAR
        mu_car = np.asarray(self.p.eta0 + self.p.etaB * B_mkt_1, dtype=float).copy()
        mu_car[ceo_mode == 1] += self.p.eta_ceo_resign
        mu_car[ceo_mode == 2] += self.p.eta_ceo_sacked
        car = self.rng.normal(mu_car, self.p.sigma_car, size=n_sims)

        # 12) Exit pressure
        exit_m = self.rng.normal(
            self.p.mu_exit + self.p.kappa_exit * B_mkt_agm,
            self.p.sigma_exit, size=n_sims,
        )

        # 13) Switching cost
        if state.prev_action is not None:
            switch_cost = self.p.c_switch[state.prev_action][pkg.name]
        else:
            switch_cost = 0.0

        # ===== LOSS DECOMPOSITION =====
        impl_cost = self.p.c_impl[pkg.name]
        car_penalty = self.p.w_price * np.maximum(-car, 0.0)
        exit_penalty = self.p.w_exit * np.maximum(-exit_m, 0.0)

        # --- L_market (uses actual strike/revolt) ---
        if state.strike_count == 0:
            strike1_penalty_mkt = self.p.w_strike * strike
            revolt_penalty_mkt = self.p.w_revolt * revolt
            strike2_penalty_mkt = np.zeros(n_sims)
            spill_penalty_mkt = np.zeros(n_sims)
        else:
            strike1_penalty_mkt = np.zeros(n_sims)
            revolt_penalty_mkt = np.zeros(n_sims)
            strike2_penalty_mkt = self.p.w_strike2 * strike2
            spill_penalty_mkt = self.p.w_spill * spill

        loss_mkt = (
            impl_cost + car_penalty + strike1_penalty_mkt + revolt_penalty_mkt
            + exit_penalty + strike2_penalty_mkt + spill_penalty_mkt
        )

        # --- L_board (uses Board-PERCEIVED strike/revolt + CEO loyalty) ---
        car_penalty_board = self.p.w_price_board * np.maximum(-car, 0.0)
        exit_penalty_board = self.p.w_exit_board * np.maximum(-exit_m, 0.0)
        ceo_removal_penalty = self.p.w_ceo_removal_board * pkg.ceo_change

        if state.strike_count == 0:
            strike_penalty_board = self.p.w_strike_board * strike_board
            revolt_penalty_board = self.p.w_revolt_board * revolt_board
            strike2_penalty_board = np.zeros(n_sims)
            spill_penalty_board = np.zeros(n_sims)
        else:
            strike_penalty_board = np.zeros(n_sims)
            revolt_penalty_board = np.zeros(n_sims)
            strike2_penalty_board = self.p.w_strike2 * strike2_board
            spill_penalty_board = self.p.w_spill * spill_board

        loss_board = (
            impl_cost + car_penalty_board + strike_penalty_board + revolt_penalty_board
            + exit_penalty_board + strike2_penalty_board + spill_penalty_board
            + switch_cost + ceo_removal_penalty
        )

        # --- L_ceo ---
        car_penalty_ceo = self.p.w_price_ceo * np.maximum(-car, 0.0)
        loss_ceo = self.p.w_impl_ceo_multiplier * impl_cost + car_penalty_ceo
        ceo_agency = np.zeros(n_sims)
        ceo_agency[ceo_mode == 1] = self.p.w_ceo_resign_mgmt
        ceo_agency[ceo_mode == 2] = self.p.w_ceo_sacked_mgmt
        loss_ceo = loss_ceo + ceo_agency

        # --- U_ASA (uses actual strike/spill — market truth) ---
        if state.strike_count == 0:
            asa_util = self.p.u_asa_strike1 * strike - asa_action.cost
        else:
            asa_util = (
                self.p.u_asa_strike2 * strike2
                + self.p.u_asa_spill * spill
                - asa_action.cost
            )

        # Legacy management loss
        loss_mgmt = loss_mkt.copy()
        loss_mgmt[ceo_mode == 1] += self.p.w_ceo_resign_mgmt
        loss_mgmt[ceo_mode == 2] += self.p.w_ceo_sacked_mgmt

        return SimulationOutputs(
            package_name=pkg.name,
            loss=loss_mkt,
            strike_event=strike,
            revolt_event=revolt,
            car_m1=car,
            exit_m12=exit_m,
            B_mkt_path=Bm,
            B_mgmt_path=Bg,
            ceo_mode=ceo_mode,
            loss_mgmt=loss_mgmt,
            loss_board=loss_board,
            loss_ceo=loss_ceo,
            strike2_event=strike2,
            spill_event=spill,
            impl_cost_component=np.full(n_sims, impl_cost),
            car_penalty_component=car_penalty,
            strike1_penalty_component=strike1_penalty_mkt,
            revolt_penalty_component=revolt_penalty_mkt,
            exit_penalty_component=exit_penalty,
            strike2_penalty_component=strike2_penalty_mkt,
            spill_penalty_component=spill_penalty_mkt,
            switch_cost_component=np.full(n_sims, switch_cost),
            ceo_agency_component=ceo_agency,
            gamma_asa_eff=None,
            asa_utility=asa_util,
            asa_action_name=asa_action.name,
            ceo_action=ceo_action,
        )

    def solve_adversarial(
        self,
        packages: List[GovernancePackage],
        state: DecisionState,
        n_sims: int = 20000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Solve the 3-player Stackelberg game for one checkpoint.

        Move order: Board chooses D -> ASA best-responds A*(D) -> CEO best-responds CEO*(D,A*)
        Board anticipates both responses when ranking.

        Returns:
            table_a: ASA+CEO best responses — one row per D
            table_b: Board ranking under adversarial+CEO response
            full_grid: Complete D×A×CEO grid with all metrics
        """
        feasible = feasible_packages(packages, state)
        actions = asa_action_set(self.p, state)
        ceo_actions = ["Stay", "Resign"]

        grid_rows = []

        for pkg in feasible:
            for act in actions:
                for ca in ceo_actions:
                    out = self.simulate_DA(
                        pkg, act, state, n_sims=n_sims, ceo_action=ca,
                    )
                    agm_idx = state.agm_month - 1
                    grid_rows.append({
                        "Checkpoint": state.checkpoint_id,
                        "D": pkg.name,
                        "A": act.name,
                        "CEO_action": ca,
                        "asa_shift": act.vote_logit_shift,
                        "asa_cost": act.cost,
                        "E[L_board]": float(out.loss_board.mean()),
                        "E[L_market]": float(out.loss.mean()),
                        "E[L_ceo]": float(out.loss_ceo.mean()),
                        "E[U_ASA]": float(out.asa_utility.mean()),
                        "P(strike1)": float(out.strike_event.mean()),
                        "P(strike2)": float(out.strike2_event.mean()),
                        "P(spill)": float(out.spill_event.mean()),
                        "P(revolt)": float(out.revolt_event.mean()),
                        "P(CEO resign)": float((out.ceo_mode == 1).mean()),
                        "P(CEO sacked)": float((out.ceo_mode == 2).mean()),
                        "Mean CAR": float(out.car_m1.mean()),
                        "Mean exit": float(out.exit_m12.mean()),
                        "E[impl_cost]": float(out.impl_cost_component.mean()),
                        "E[car_penalty]": float(out.car_penalty_component.mean()),
                        "E[strike1_penalty]": float(out.strike1_penalty_component.mean()),
                        "E[revolt_penalty]": float(out.revolt_penalty_component.mean()),
                        "E[exit_penalty]": float(out.exit_penalty_component.mean()),
                        "E[strike2_penalty]": float(out.strike2_penalty_component.mean()),
                        "E[spill_penalty]": float(out.spill_penalty_component.mean()),
                        "E[switch_cost]": float(out.switch_cost_component.mean()),
                        "E[ceo_agency]": float(out.ceo_agency_component.mean()),
                        "Mean B_mkt_agm": float(out.B_mkt_path[:, agm_idx].mean()),
                        "strike_count": state.strike_count,
                        "prev_action": state.prev_action or "",
                    })

        full_grid = pd.DataFrame(grid_rows)

        # --- Solve backward: CEO -> ASA -> Board ---

        # For each (D, A): CEO*(D,A) = argmin E[L_ceo]
        # Then for each D: A*(D) = argmax E[U_ASA | D, A, CEO*(D,A)]
        # Board ranks by E[L_board | D, A*(D), CEO*(D,A*(D))]

        table_a_rows = []
        for pkg in feasible:
            pkg_grid = full_grid[full_grid["D"] == pkg.name]

            # Step 1: For each ASA action, find CEO best response
            best_by_asa = []
            for act in actions:
                da_rows = pkg_grid[pkg_grid["A"] == act.name]
                # CEO minimises L_ceo
                ceo_best_idx = da_rows["E[L_ceo]"].idxmin()
                best_by_asa.append(da_rows.loc[ceo_best_idx])

            # Step 2: Among (A, CEO*(D,A)) pairs, ASA maximises U_ASA
            best_asa_df = pd.DataFrame(best_by_asa)
            asa_best_idx = best_asa_df["E[U_ASA]"].idxmax()
            best = best_asa_df.loc[asa_best_idx]

            table_a_rows.append({
                "D": pkg.name,
                "A*(D)": best["A"],
                "CEO*(D)": best["CEO_action"],
                "E[U_ASA|D,A*]": best["E[U_ASA]"],
                "E[L_board|D,A*]": best["E[L_board]"],
                "E[L_market|D,A*]": best["E[L_market]"],
                "E[L_ceo|D,A*]": best["E[L_ceo]"],
                "P(strike1)": best["P(strike1)"],
                "P(strike2)": best["P(strike2)"],
                "P(spill)": best["P(spill)"],
            })
        table_a = pd.DataFrame(table_a_rows)

        # Table B: Board ranking under adversarial + CEO response
        table_b = table_a.sort_values("E[L_board|D,A*]").reset_index(drop=True)
        table_b["Rank"] = np.arange(1, len(table_b) + 1)
        table_b["board_optimal"] = table_b["Rank"] == 1

        return table_a, table_b, full_grid


# ----------------------------
# Main: sequential decision loop
# ----------------------------

def main() -> None:
    packages = default_packages()

    checkpoints = [
        {
            "id": "C0",
            "path": "data/checkpoints/belief_C0_2023-10-01.npz",
            "horizon_months": 2, "agm_month": 2,
            "V": 0.30, "imminent_agm": True, "strike_count": 0,
        },
        {
            "id": "C1",
            "path": "data/checkpoints/belief_C1_2023-10-10.npz",
            "horizon_months": 2, "agm_month": 2,
            "V": 0.30, "imminent_agm": True, "strike_count": 0,
        },
        {
            "id": "C2",
            "path": "data/checkpoints/belief_C2_2023-10-18.npz",
            "horizon_months": 1, "agm_month": 1,
            "V": 0.30, "imminent_agm": True, "strike_count": 0,
        },
        {
            "id": "C3",
            "path": "data/checkpoints/belief_C3_2023-11-03.npz",
            "horizon_months": 12, "agm_month": 12,
            "V": 0.30, "imminent_agm": False, "strike_count": 1,
        },
    ]

    prev_action: Optional[str] = None
    all_grids = []

    for cfg in checkpoints:
        cp = np.load(cfg["path"], allow_pickle=True)
        B0_mkt = cp["B_mkt"]
        B0_mgmt = cp["B_mgmt"] if "B_mgmt" in cp.files else None
        gamma_A = cp["gamma_A"] if "gamma_A" in cp.files else None

        # Build state
        state = DecisionState(
            checkpoint_id=cfg["id"],
            horizon_months=cfg["horizon_months"],
            agm_month=cfg["agm_month"],
            strike_count=cfg["strike_count"],
            prev_action=prev_action,
            V=cfg["V"],
            B0_mkt_draws=B0_mkt,
            B0_mgmt_draws=B0_mgmt,
            gamma_A_draws=gamma_A,
            imminent_agm=cfg["imminent_agm"],
        )

        # Sanity print
        q = np.quantile(B0_mkt, [0.05, 0.50, 0.95])
        print(f"\n{'='*70}")
        print(f"  {cfg['id']} | strike_count={state.strike_count} | prev_action={state.prev_action}")
        print(f"{'='*70}")
        print(f"  B0_mkt  mean={B0_mkt.mean():.4f}  p05={q[0]:.4f}  p50={q[1]:.4f}  p95={q[2]:.4f}")
        if B0_mgmt is not None:
            qg = np.quantile(B0_mgmt, [0.05, 0.50, 0.95])
            print(f"  B0_mgmt mean={B0_mgmt.mean():.4f}  p05={qg[0]:.4f}  p50={qg[1]:.4f}  p95={qg[2]:.4f}")

        # Build params and simulator
        params_ck = ToyParams(
            rho_mkt=0.90, sigma_B_mkt=0.45,
            rho_mgmt=0.96, sigma_B_mgmt=0.25,
        )
        params_ck.V = cfg["V"]

        sim = ARASimulator(
            params_ck,
            B0_draws=B0_mkt,
            rng=np.random.default_rng(42),
            B0_mgmt_draws=B0_mgmt,
        )

        # Adversarial Stackelberg solve
        table_a, table_b, full_grid = sim.solve_adversarial(
            packages, state, n_sims=20000,
        )

        # --- Table A: ASA + CEO best responses ---
        actions = asa_action_set(params_ck, state)
        action_names = [a.name for a in actions]
        print(f"\n  Table A — Best Responses  (ASA actions: {action_names}, CEO actions: [Stay, Resign])")
        print(f"  {'-'*80}")
        ta_cols = ["D", "A*(D)", "CEO*(D)", "E[U_ASA|D,A*]", "E[L_ceo|D,A*]",
                   "E[L_board|D,A*]", "P(strike1)", "P(strike2)", "P(spill)"]
        ta_display = [c for c in ta_cols if c in table_a.columns]
        print(table_a[ta_display].to_string(index=False))

        # --- Table B: Board ranking (under adversarial + CEO response) ---
        print(f"\n  Table B — Board Ranking (under ASA + CEO best response)")
        print(f"  {'-'*80}")
        tb_cols = ["Rank", "D", "A*(D)", "CEO*(D)", "E[L_board|D,A*]",
                   "E[L_market|D,A*]", "E[L_ceo|D,A*]", "board_optimal"]
        tb_display = [c for c in tb_cols if c in table_b.columns]
        print(table_b[tb_display].to_string(index=False))

        # Board-optimal action under adversarial response -> prev_action
        if len(table_b) > 0:
            board_opt = table_b.loc[table_b["board_optimal"]].iloc[0]
            prev_action = board_opt["D"]
            print(f"\n  >> Board-optimal (adversarial): {prev_action}")

        all_grids.append(full_grid)

    # Write adversarial CSV
    df_all = pd.concat(all_grids, ignore_index=True)
    df_all.to_csv("data/sim_summary_by_checkpoint_adversarial.csv", index=False)
    print(f"\nWrote: data/sim_summary_by_checkpoint_adversarial.csv")


if __name__ == "__main__":
    main()
