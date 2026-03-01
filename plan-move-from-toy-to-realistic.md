# Plan: Move from Toy to Structured ARA in 4 Phases

## Phase 0 — Freeze what's "toy" today (so you can replace pieces safely)

Create a single config manifest (CSV/XLSX-first, as you prefer) that enumerates every toy assumption currently embedded in code. In `ToyParams` you have all the big ones: AR(1) $\rho$, $\sigma_B$, feature shifts `delta`, mappings $(\alpha, \kappa)$ to votes and exit, CAR model, and loss weights.

**Deliverables:**

`data/model_manifest.csv` with columns:

- `component` (belief_dynamics, vote_link, car_link, exit_link, utility_weights, governance_effects)
- `parameter`, `current_value`, `source` (toy/stan/posterior/panel), `notes`

This becomes the checklist for systematically deleting toy assumptions.

## Phase 1 — Split beliefs + add information asymmetry

**Goal:** run the simulator with two states $B^{mkt}_t$ and $B^{mgmt}_t$, with different observation models.

### State evolution

**Market:**

$$B^{mkt}*t = \rho_m  B^{mkt}*{t-1} + \epsilon^m_t + s^{pub}_t$$

**Management:**

$$B^{mgmt}*t = \rho_g  B^{mgmt}*{t-1} + \epsilon^g_t + s^{pub}_t + s^{priv}_t$$

Where $s^{pub}_t$ are public shocks; $s^{priv}_t$ are private (ASA ↔ mgmt) shocks.

### Observation models

AGM rem vote and (if you want) proxy/director revolt depend on market belief:

$$\text{logit}(\text{remagainst}) = \alpha_{rem} + B^{mkt} + \text{noise}$$

(your anchored scale)

CAR and exit pressure depend on market belief (as now).

Management's internal "support belief" is not directly observed (or observed via internal proxies if you have them later).

### What changes in code

- Replace `B_path` with two paths: `B_mkt_path`, `B_mgmt_path`
- Replace `_governance_shift` with two effects:
  - `shift_mkt(D)` (what the market believes the package implies)
  - `shift_mgmt(D)` (what management privately thinks + what it hears privately)

This directly encodes: market doesn't see ASA private discussions; management does.

### Deliverable

New checkpoint files contain draws for both:

- `B_mkt` and `B_mgmt` (even if initially you set `B_mgmt = B_mkt + bias` as a placeholder)

## Phase 2 — Add management overconfidence / bias

**Goal:** make $B^{mgmt}$ systematically "less bad" than $B^{mkt}$ in a parameterized way.

Introduce a bias term:

$$B^{mgmt}_t = B^{true}*t - \delta*{bias} + \nu_t$$

or, if you don't want a "true" state:

$$B^{mgmt}_t = B^{mkt}*t - \delta*{bias} + \nu_t$$

with $\delta_{bias} > 0$ meaning management underestimates distrust severity.

Then make decision-making depend on management belief:

- The firm chooses $D$ to maximize management utility given $B^{mgmt}$ (not market's).
- The market reacts based on $B^{mkt}$ and observed $D$.

### Calibration options (from least to most ambitious)

- **2A (fast):** fixed prior for $\delta_{bias}$ (e.g., $\text{Normal}(0.5, 0.25)$) and sensitivity sweep.
- **2B:** infer $\delta_{bias}$ from "surprise gaps": management actions vs market reactions (needs data / proxies).
- **2C:** add internal signals (board minutes, etc.) if ever available (likely not).

### Deliverables

- `data/priors_overconfidence.csv` specifying prior on $\delta_{bias}$ by checkpoint or constant.
- Sensitivity grid results showing when bias flips optimal $D$.

## Phase 3 — Split utilities + model agency

**Goal:** different payoffs for market vs CEO/management, so the "optimal" action can diverge.

### Define two utility functions

**Market / shareholder utility $U^{mkt}$:**

- penalizes strike/revolt risk (pre-AGM checkpoints)
- penalizes downside CAR
- penalizes exit pressure / reputational capital loss
- does not include CEO personal costs

**Management / CEO utility $U^{mgmt}$:**

- includes a personal disutility of early exit (lost pay, reputation, control)
- includes private cost of admitting fault (e.g., clawbacks, governance constraints)
- may include "survival" / "retain role" benefit

### Mechanically

Keep your loss components (CAR downside, strike, revolt, exit) as the market part.

Add a CEO term:

```
+ w_ceo_exit * I(mode != stay)
```

or a continuous payoff.

Then: firm's chosen $D$ is argmin of management loss, while evaluation (what you report) can show both:

- Expected loss (market)
- Expected loss (management)
- plus an "agency gap" metric: $E[L^{mkt}] - E[L^{mgmt}]$

### Deliverables

- `data/utility_weights_market.csv`
- `data/utility_weights_mgmt.csv`
- A combined report table per checkpoint showing both rankings side-by-side.

## Phase 4 — Distinguish "CEO resigns" vs "CEO is sacked"

**Goal:** model the mode of transition as endogenous and differently perceived.

### Introduce a discrete CEO transition mode $M$

- $M = \text{stay}$
- $M = \text{resign}$
- $M = \text{sacked}$

### Model two things separately

**Market perception effects** (different CAR and belief shifts):

- Resignation might be read as "accountability / clean break" (bigger positive CAR, bigger drop in $B^{mkt}$)
- Sacking might be read as "board forced / crisis severity" (mixed: short-term pop vs signal of deep problems)

So replace the single `etaCEO * ceo_change` term with:

- `eta_resign`, `eta_sacked`

and similarly split governance shift on market belief:

- `delta_mkt_resign`, `delta_mkt_sacked`

**CEO/management incentives** (different mgmt utility):

- CEO hates "sacked" much more than "resign"
- Board cost differs too (legal/contractual, instability)

So management loss gets:

- `c_ceo_resign`, `c_ceo_sacked` (with `c_sacked > c_resign`)

### How to make $M$ happen

If $D$ includes CEO transition, sample mode via a logistic choice model depending on:

- management belief $B^{mgmt}$ (more distrust → higher chance of transition)
- board toughness parameter
- CEO "fight" parameter (agency)

### Deliverables

- `data/ceo_mode_params.csv` (priors + mapping)
- Simulator outputs: $P(\text{resign})$, $P(\text{sacked})$ and their contribution to CAR and belief updates.

---

# How to Migrate Away from "Toy" Assumptions Safely

You already have a clean separation: `ToyParams` + simulator class.

Use that to do incremental replacement:

## Replacement order (min risk → max risk)

1. **Belief draws:** extend checkpoint `.npz` to include `B_mgmt` draws (even if derived initially).
2. **Mappings:** keep anchored market vote mapping ($\text{logit}(\text{rem}) = \alpha + B^{mkt}$) as the identity anchor (good choice).
3. **CAR model:** replace toy linear CAR with an empirical regression/likelihood calibrated on QAN CAR around comparable announcements (or keep as sensitivity).
4. **Exit pressure:** replace toy linear normal with something tied to flows / volume / analyst downgrades (or keep as sensitivity).
5. **Governance effects ($\delta_*$):** stop treating them as fixed; model as uncertain draws (priors), later calibrate from panel.
6. **Utilities:** promote market vs mgmt utilities to first-class outputs.

## Sensitivity sweeps you should run immediately (before heavy calibration)

- **Overconfidence $\delta_{bias}$ sweep:** when does it flip D1 vs D3?
- **Agency cost sweep:** when does management refuse D3 even if market would prefer it?
- **CEO mode split:** when does "resign" dominate "sacked" in market utility vs management utility?

---

# Practical Implementation Checklist (What to Code Next)

## Data

Add `B_mgmt` into checkpoint generation:

- simplest: `B_mgmt = B_mkt - delta_bias + noise`
- keep `delta_bias` and noise as configurable priors/sweep inputs.

## Simulator

- Replace single `B_path` with two.
- Replace governance shift with two vectors: `delta_mkt`, `delta_mgmt`.
- Add a `simulate_ceo_mode()` function and propagate into CAR + utilities.

## Outputs

Table per checkpoint with:

- `E[L_market]`, `E[L_mgmt]`, `agency_gap`
- `P(strike)`, `P(revolt)` (only when pre-AGM)
- `P(resign)`, `P(sacked)` (when CEO transition is possible)
- mean CAR, mean exit pressure

---

# Minimal-Diff Implementation Sketch

Below is a **minimal-diff** sketch that keeps your current structure (a `ToyParams` container + `ARASimulator` with `simulate_package()` and `simulate_all()`), but adds:

1. **Two belief states**: `B_mkt` and `B_mgmt`
2. **CEO-mode split**: `stay / resign / sacked` with separate market vs management utilities
3. **Backwards compatibility**: if you don't provide `B0_mgmt_draws`, it is generated from `B0_mkt_draws` via an overconfidence bias.

I'm going to show this as "drop-in edits" rather than a rewrite.

---

## 1) ToyParams: minimal additions

Keep your existing fields, but add **parallel parameters** and **CEO-mode / utility** parameters. The goal is: *default behaviour stays close to your current toy*, but you can start separating beliefs/utilities immediately.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ToyParams:
    # --- existing fields (keep) ---
    rho: float = 0.95
    sigma_B: float = 0.50

    alpha_rem: float = -1.20
    kappa_rem: float = 1.00
    alpha_dir: float = -1.00
    kappa_dir: float = 0.80

    theta_strike: float = 0.25
    theta_revolt: float = 0.30

    # Loss weights (you already have these)
    w_impl: float = 1.0
    w_strike: float = 4.0
    w_revolt: float = 3.0
    w_car_down: float = 40.0
    w_exit_down: float = 2.0

    # Governance deltas (you already have something like this; keep structure)
    # e.g. {"review": -0.5, "transparency": -0.3, "clawback": -0.2, "ceo": -0.8}
    delta: dict = None

    # CAR parameters (toy)
    eta_review: float = 0.01
    eta_transparency: float = 0.005
    eta_clawback: float = 0.002
    eta_ceo: float = 0.02
    sigma_car: float = 0.02

    # exit pressure toy
    exit_beta: float = -0.50
    exit_sigma: float = 0.30

    # --- NEW: split belief dynamics (defaults = reuse existing) ---
    rho_mkt: float = None
    sigma_B_mkt: float = None
    rho_mgmt: float = None
    sigma_B_mgmt: float = None

    # --- NEW: management overconfidence mapping (mgmt belief derived if not provided) ---
    # mgmt underestimates distrust by bias > 0 (i.e. B_mgmt = B_mkt - bias + noise)
    mgmt_bias: float = 0.50
    mgmt_bias_sigma: float = 0.20

    # --- NEW: governance belief shifts split by audience ---
    # if None, fall back to delta for both
    delta_mkt: dict = None
    delta_mgmt: dict = None

    # --- NEW: CEO mode modelling (only relevant when CEO transition is on) ---
    # logits for mode selection (resign vs sacked) conditional on "transition happens"
    ceo_mode_intercept: float = 0.0
    ceo_mode_slope_Bmgmt: float = 0.5   # higher distrust -> more likely sacked (example)
    # probability that a CEO transition actually happens when package includes it
    ceo_transition_intercept: float = 1.5
    ceo_transition_slope_Bmgmt: float = 0.7

    # --- NEW: market perception differs by mode ---
    eta_ceo_resign: float = 0.03
    eta_ceo_sacked: float = 0.015
    # belief shifts differ by mode (market updates)
    delta_ceo_resign_mkt: float = -1.0
    delta_ceo_sacked_mkt: float = -0.7

    # --- NEW: management utility / agency costs ---
    # market loss is your existing loss; mgmt loss adds CEO personal costs
    w_ceo_resign_mgmt: float = 1.0
    w_ceo_sacked_mgmt: float = 3.0

    def finalize(self) -> None:
        """Call once after init to fill None defaults."""
        if self.delta is None:
            self.delta = {"review": -0.5, "transparency": -0.3, "clawback": -0.2, "ceo": -0.8}

        if self.rho_mkt is None: self.rho_mkt = self.rho
        if self.sigma_B_mkt is None: self.sigma_B_mkt = self.sigma_B
        if self.rho_mgmt is None: self.rho_mgmt = self.rho
        if self.sigma_B_mgmt is None: self.sigma_B_mgmt = self.sigma_B

        if self.delta_mkt is None: self.delta_mkt = dict(self.delta)
        if self.delta_mgmt is None: self.delta_mgmt = dict(self.delta)
```

**Why this is minimal:** you're not deleting any old params; you're only adding new optional ones + a `finalize()` helper.

---

## 2) ARASimulator: minimal interface changes

### 2.1 Constructor: accept market and management draws

```python
class ARASimulator:
    def __init__(
        self,
        p: ToyParams,
        B0_draws: np.ndarray,
        rng: np.random.Generator,
        B0_mgmt_draws: np.ndarray | None = None,
    ):
        self.p = p
        self.p.finalize()
        self.B0_mkt_draws = np.asarray(B0_draws)
        self.B0_mgmt_draws = None if B0_mgmt_draws is None else np.asarray(B0_mgmt_draws)
        self.rng = rng
```

Back-compat: you can keep your existing call sites that only pass `B0_draws`.

---

### 2.2 Helper: draw initial beliefs

```python
    def _draw_B0_pair(self, n_sims: int) -> tuple[np.ndarray, np.ndarray]:
        B0_mkt = self.rng.choice(self.B0_mkt_draws, size=n_sims, replace=True)

        if self.B0_mgmt_draws is not None:
            B0_mgmt = self.rng.choice(self.B0_mgmt_draws, size=n_sims, replace=True)
        else:
            # Derive mgmt belief from market belief + overconfidence bias
            bias = self.rng.normal(self.p.mgmt_bias, self.p.mgmt_bias_sigma, size=n_sims)
            B0_mgmt = B0_mkt - bias

        return B0_mkt, B0_mgmt
```

---

### 2.3 Governance shift: split into mkt/mgmt

Assuming you already have something like `_governance_shift(pkg)` that returns a scalar shift.

Replace it with:

```python
    def _governance_shift_pair(self, pkg) -> tuple[float, float]:
        # minimal: same feature flags, different lookup dicts
        shift_mkt = 0.0
        shift_mgmt = 0.0

        if pkg.review:
            shift_mkt += self.p.delta_mkt["review"]
            shift_mgmt += self.p.delta_mgmt["review"]
        if pkg.transparency:
            shift_mkt += self.p.delta_mkt["transparency"]
            shift_mgmt += self.p.delta_mgmt["transparency"]
        if pkg.clawback:
            shift_mkt += self.p.delta_mkt["clawback"]
            shift_mgmt += self.p.delta_mgmt["clawback"]
        if pkg.ceo_change:
            # IMPORTANT: CEO change is no longer a single shift; it is mode-dependent
            # so we do NOT apply it here; handle in CEO-mode block later
            pass

        return shift_mkt, shift_mgmt
```

This is the smallest change that gets you "two-belief" without messing with package definitions.

---

### 2.4 Belief path simulation: two paths

Replace your single `B_path` with:

```python
    def _simulate_belief_paths(
        self,
        B0_mkt: np.ndarray,
        B0_mgmt: np.ndarray,
        shift_mkt: float,
        shift_mgmt: float,
        months: int,
        persistent_shift: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = B0_mkt.size
        Bm = np.zeros((n, months))
        Bg = np.zeros((n, months))

        # t=0
        Bm[:, 0] = self.p.rho_mkt * B0_mkt + shift_mkt + self.rng.normal(0.0, self.p.sigma_B_mkt, size=n)
        Bg[:, 0] = self.p.rho_mgmt * B0_mgmt + shift_mgmt + self.rng.normal(0.0, self.p.sigma_B_mgmt, size=n)

        for t in range(1, months):
            add_mkt = shift_mkt if persistent_shift else 0.0
            add_mgmt = shift_mgmt if persistent_shift else 0.0

            Bm[:, t] = self.p.rho_mkt * Bm[:, t - 1] + add_mkt + self.rng.normal(0.0, self.p.sigma_B_mkt, size=n)
            Bg[:, t] = self.p.rho_mgmt * Bg[:, t - 1] + add_mgmt + self.rng.normal(0.0, self.p.sigma_B_mgmt, size=n)

        return Bm, Bg
```

---

### 2.5 CEO mode: new helper (stay/resign/sacked)

This is the "CEO resign vs sacked" split. It uses **management belief** to decide whether transition happens + which mode (agency).

```python
    def _simulate_ceo_mode(self, pkg, B_mgmt_agm: np.ndarray) -> np.ndarray:
        """
        Returns mode codes: 0=stay, 1=resign, 2=sacked
        Only meaningful if pkg.ceo_change.
        """
        n = B_mgmt_agm.size
        mode = np.zeros(n, dtype=int)

        if not pkg.ceo_change:
            return mode

        # Does a transition occur?
        p_trans = 1.0 / (1.0 + np.exp(-(self.p.ceo_transition_intercept + self.p.ceo_transition_slope_Bmgmt * B_mgmt_agm)))
        trans = self.rng.uniform(0, 1, size=n) < p_trans

        # Given transition, decide resign vs sacked
        # Higher B_mgmt -> more likely "sacked" (board forced) as an example
        p_sacked = 1.0 / (1.0 + np.exp(-(self.p.ceo_mode_intercept + self.p.ceo_mode_slope_Bmgmt * B_mgmt_agm)))
        sacked = self.rng.uniform(0, 1, size=n) < p_sacked

        mode[trans & ~sacked] = 1
        mode[trans & sacked] = 2
        return mode
```

You can later swap this out for a more theory-grounded model; interface stays.

---

### 2.6 Outcomes: votes depend on **market** belief; CEO effects depend on mode

Inside `simulate_package()` you currently compute `B_agm` and then map to vote probabilities and CAR. Minimal changes:

- Compute both `B_mkt_agm` and `B_mgmt_agm`
- Compute vote probs from `B_mkt_agm`
- Sample CEO mode from `B_mgmt_agm`
- Apply **mode-dependent** belief shift and CAR effect

Sketch:

```python
    def simulate_package(self, pkg, n_sims=20000, months=2, agm_month=2, observed_y_rem=None, observed_y_dir=None):
        B0_mkt, B0_mgmt = self._draw_B0_pair(n_sims)
        shift_mkt, shift_mgmt = self._governance_shift_pair(pkg)

        Bm, Bg = self._simulate_belief_paths(B0_mkt, B0_mgmt, shift_mkt, shift_mgmt, months, persistent_shift=True)
        B_mkt_agm = Bm[:, agm_month - 1]
        B_mgmt_agm = Bg[:, agm_month - 1]

        # CEO mode (new)
        ceo_mode = self._simulate_ceo_mode(pkg, B_mgmt_agm)

        # Apply mode-dependent market belief shift (only market sees the "accountability signal")
        # Minimal: adjust B_mkt_agm in-place AFTER mode draw.
        B_mkt_agm = B_mkt_agm.copy()
        B_mkt_agm[ceo_mode == 1] += self.p.delta_ceo_resign_mkt
        B_mkt_agm[ceo_mode == 2] += self.p.delta_ceo_sacked_mkt

        # Rem vote depends on market belief (anchored)
        p_rem = 1.0 / (1.0 + np.exp(-(self.p.alpha_rem + self.p.kappa_rem * B_mkt_agm)))
        p_dir = 1.0 / (1.0 + np.exp(-(self.p.alpha_dir + self.p.kappa_dir * B_mkt_agm)))

        # Realize vote outcomes (or override if observed)
        y_rem = np.full(n_sims, float(observed_y_rem)) if observed_y_rem is not None else noisy_against(p_rem)
        y_dir = np.full(n_sims, float(observed_y_dir)) if observed_y_dir is not None else noisy_against(p_dir)

        strike = (y_rem > self.p.theta_strike).astype(float)
        revolt = (y_dir > self.p.theta_revolt).astype(float)

        # CAR (now mode-dependent)
        mu_car = (
            (self.p.eta_review if pkg.review else 0.0)
            + (self.p.eta_transparency if pkg.transparency else 0.0)
            + (self.p.eta_clawback if pkg.clawback else 0.0)
        )
        # CEO effect by mode
        mu_car = np.full(n_sims, mu_car)
        mu_car[ceo_mode == 1] += self.p.eta_ceo_resign
        mu_car[ceo_mode == 2] += self.p.eta_ceo_sacked
        car_m1 = mu_car + self.rng.normal(0.0, self.p.sigma_car, size=n_sims)

        # Exit pressure (still based on market belief)
        exit_p = self.p.exit_beta * B_mkt_agm + self.rng.normal(0.0, self.p.exit_sigma, size=n_sims)

        # Implementation cost (keep your existing logic)
        impl = pkg.impl_cost

        # Market loss (existing components)
        loss_mkt = (
            self.p.w_impl * impl
            + self.p.w_strike * strike
            + self.p.w_revolt * revolt
            + self.p.w_car_down * np.maximum(0.0, -car_m1)
            + self.p.w_exit_down * np.maximum(0.0, -exit_p)
        )

        # Management loss = market loss + CEO personal agency costs (new)
        loss_mgmt = loss_mkt.copy()
        loss_mgmt[ceo_mode == 1] += self.p.w_ceo_resign_mgmt
        loss_mgmt[ceo_mode == 2] += self.p.w_ceo_sacked_mgmt

        return SimulationOutputs(
            # keep your existing fields...
            loss=loss_mkt,
            strike=strike,
            revolt=revolt,
            car_m1=car_m1,
            exit_p=exit_p,
            B_mkt_agm=B_mkt_agm,
            B_mgmt_agm=B_mgmt_agm,
            ceo_mode=ceo_mode,
            loss_mgmt=loss_mgmt,
        )
```

This is the minimal "two belief + CEO mode" insertion point without touching the rest of the plumbing.

---

## 3) SimulationOutputs: add a few fields (minimal)

Whatever your current outputs object is (dataclass or dict), add only what you need to summarize the new mechanics.

```python
@dataclass
class SimulationOutputs:
    loss: np.ndarray                 # market loss (existing)
    strike: np.ndarray
    revolt: np.ndarray
    car_m1: np.ndarray
    exit_p: np.ndarray

    # NEW:
    B_mkt_agm: np.ndarray
    B_mgmt_agm: np.ndarray
    ceo_mode: np.ndarray            # 0/1/2
    loss_mgmt: np.ndarray           # management loss
```

---

## 4) Summary table: add CEO mode stats + management utility (minimal edits)

In your summarizer (where you currently compute expected loss, quantiles, probs), add:

- `P(resign) = mean(ceo_mode==1)`
- `P(sacked) = mean(ceo_mode==2)`
- `Expected loss (mgmt) = mean(loss_mgmt)`
- `Agency gap = E[loss_mkt] - E[loss_mgmt]` (or reverse)

This does *not* require changing `simulate_all()`, only the aggregation.

---

## 5) Checkpoint plumbing: minimal change to allow mgmt draws later

Right now you load `B_mkt` from checkpoint `.npz`. Keep that, and optionally pass `B_mgmt` when you have it:

```python
cp = np.load(path)
B0_mkt = cp["B_mkt"]
B0_mgmt = cp["B_mgmt"] if "B_mgmt" in cp.files else None
sim = ARASimulator(params, B0_draws=B0_mkt, B0_mgmt_draws=B0_mgmt, rng=np.random.default_rng(42))
```

Until you generate `B_mgmt` properly, you still get separation via `mgmt_bias`.

---

## What This Buys You Immediately (Mapping to Your 4 Issues)

- **Market doesn't know ASA ↔ mgmt discussions:** you can add private shifts into `B_mgmt` only (later) without affecting market mapping.
- **Mgmt overconfidence:** `mgmt_bias` creates systematic divergence between $B^{mkt}$ and $B^{mgmt}$.
- **Different payoffs (agency):** `loss_mgmt` adds personal CEO costs, so management may prefer D1/D2 even when market prefers D3.
- **Resign vs sacked:** `ceo_mode` drives different CAR impacts and different belief shifts, and different mgmt costs.

# Resign Versus Sacked Parameterisation

Here’s a **concrete default parameterisation** for the *resign vs sacked* channel that is genuinely anchored to the facts/behavioural claims in your attached note, and scaled to your simulator’s existing “loss units”.

I’m going to assume your new two-belief + CEO-mode fields exist (per the minimal-diff sketch), and I’ll give you **numerical defaults** + the rationale for each one.

---

## 1) What we can legitimately infer from the attached note

### 1.1 CEO’s private cost: “sacked” is *much* worse than “resign”

Joyce’s *resignation / retirement* preserved “good leaver” status and eligibility for pro-rata vesting; headline FY23 remuneration was **$21.4m**. 
The board later applied malus/clawback and reduced by **$9.26m** to **~$14.4m**. 
A “termination for cause” scenario is described as **total forfeiture of all outstanding incentives** (no pro-rata vesting) and potentially no notice pay. 

**Implication for parameters:** management/CEO utility should assign a **big additional disutility** to “sacked” versus “resign”.

### 1.2 Mode selection: strong incentive to *frame* exits as voluntary

The note explicitly says CEOs rarely admit to being fired; boards and CEOs have mutual incentives to frame as voluntary; and ~**48%** of departures sit in an ambiguous “push-out” middle zone. 
Also: “stock price volatility increases with push-out score” — i.e. involuntary-looking exits trigger stronger market reactions. 

**Implication for parameters:**

- Conditional on “transition happens”, baseline should heavily favour **resign/retire framing** (intercept favouring resign).
- But as *internal* crisis severity rises, the probability of “sacked” should increase (positive slope on (B^{mgmt})).
- Market reaction should have **bigger variance / tail risk** for “sacked” than “resign” (at minimum, higher `sigma_car` conditional on mode; if you keep one sigma, then a more negative mean for sacked is the next best toy proxy).

---

## 2) Concrete parameter defaults (drop-in values)

These values are chosen so that:

- “Resign” is the common outcome when CEO transition is invoked (face-saving default),
- “Sacked” becomes plausible only when (B^{mgmt}) is very high,
- management utility penalises sacked ~3× resign (matching “partial reduction vs total forfeiture” spirit),
- market CAR mean is mildly positive for resign, mildly negative for sacked, and sacked has higher volatility proxy.

### 2.1 CEO transition probability (if package includes CEO transition)

If D3 includes `ceo_change=1`, you probably want transition to occur most of the time, *but* allow management resistance when (B^{mgmt}) is low.

```python
# Probability CEO transition occurs (given package has ceo_change=1)
ceo_transition_intercept = 2.0
ceo_transition_slope_Bmgmt = 0.6
```

Interpretation (logit):

- At (B^{mgmt}=0): p ≈ logistic(2.0)=0.88
- At (B^{mgmt}=2): p ≈ logistic(3.2)=0.96
- At (B^{mgmt}=4): p ≈ logistic(4.4)=0.99

This is consistent with “if the board goes for CEO transition, it usually happens”, but still allows some “CEO hangs on” mass when crisis is perceived internally as not catastrophic.

### 2.2 Mode selection: resign vs sacked (conditional on transition)

To reflect the “retired-or-fired ambiguity / face-saving” incentive, start with a strong resign bias (negative intercept), and let severity push it toward sacked.

```python
# Probability of sacked given transition
ceo_mode_intercept = -2.0
ceo_mode_slope_Bmgmt = 0.7
```

Interpretation:

- (B^{mgmt}=0): p(sacked|transition)=logistic(-2.0)=0.12 (mostly resign/retire framing)
- (B^{mgmt}=2): logistic(-0.6)=0.35
- (B^{mgmt}=4): logistic(0.8)=0.69 (sacking becomes common only at high crisis)

That matches the “mutual incentive to disguise” + the reality that truly forced removals happen when things are very bad. 

### 2.3 Market perception: belief shift by mode

You want “CEO transition” to reduce market distrust, but **resignation** reads as an orderly accountability step, while **sacking** reads as disorderly / “something must be very wrong” even if accountability is strong.

So: both reduce distrust; resignation reduces it more.

```python
delta_ceo_resign_mkt = -1.00
delta_ceo_sacked_mkt = -0.60
```

These are in your **anchored rem-vote logit units**, so they are big—but that’s OK because your current toy deltas are already of order 0.3–0.8 per feature. 
(And you already saw how sensitive strike probability is to ~1 logit unit shifts.)

### 2.4 Market CAR mean by mode (and a volatility proxy)

The note says investors react more dramatically to involuntary-looking exits. 
In your toy model CAR enters loss only on downside (`w_price * max(-car,0)`). 
So: model resign as mildly positive (rarely penalised), model sacked as mildly negative (often penalised), and increase uncertainty for sacked.

If you can afford per-mode sigma:

```python
eta_ceo_resign = +0.012
eta_ceo_sacked = -0.006

sigma_car_resign = 0.010
sigma_car_sacked = 0.018
```

If you *must* keep one `sigma_car`, set `sigma_car=0.014` and keep the mean split as above.

### 2.5 Management utility: agency cost by mode

Here we translate the remuneration asymmetry into **loss units**.

We know: “resign/retire” preserved substantial value; later clawback reduced by **$9.26m** but did not wipe incentives; “termination for cause” is described as **total forfeiture**, i.e. much worse. 

A simple, defensible mapping is: sacked costs management about **3×** resign, in the simulator’s utility scale.

```python
w_ceo_resign_mgmt = 0.8
w_ceo_sacked_mgmt = 2.4
```

This keeps agency costs in the same ballpark as your existing implementation costs (D3 impl cost is 0.70 loss units in your toy). 
So “sacking” is a big personal hit relative to any other governance lever—exactly what you want for agency.

---

## 3) One compact “parameter block” you can paste into ToyParams

```python
# --- CEO transition & mode ---
p.ceo_transition_intercept = 2.0
p.ceo_transition_slope_Bmgmt = 0.6

p.ceo_mode_intercept = -2.0
p.ceo_mode_slope_Bmgmt = 0.7

# --- Market belief effects (anchored units) ---
p.delta_ceo_resign_mkt = -1.00
p.delta_ceo_sacked_mkt = -0.60

# --- Market CAR effects ---
p.eta_ceo_resign = +0.012
p.eta_ceo_sacked = -0.006
# optional if you support it
p.sigma_car_resign = 0.010
p.sigma_car_sacked = 0.018

# --- Management agency costs ---
p.w_ceo_resign_mgmt = 0.8
p.w_ceo_sacked_mgmt = 2.4
```

---

## 4) Quick sanity checks you should see after wiring this

Once you add reporting of `P(resign)` and `P(sacked)`:

- At **C0/C1** (low (B^{mkt}) and likely lower (B^{mgmt})), D3 should mostly produce **resign/retire**, rarely sacked.
- At **C2/C3** (high (B^{mkt}), and with mgmt bias maybe still high), “sacked” should become non-trivial.
- **Market loss** should often prefer resignation over sacking because sacking creates downside CAR penalty + smaller trust improvement.
- **Management loss** should strongly prefer resignation over sacking due to the big agency cost ratio implied by the remuneration/forfeiture asymmetry.

# CEO Overconfidence and Deference to CEO

Here are the **additional structure changes** I’d make (beyond the two-belief + resign/sacked channel) to explicitly encode **(a) CEO overconfidence**, and **(b) board/executive “excessive deference”**—using only what’s in your attached note as behavioural justification. 

---

## 1) Structure changes (small, but high leverage)

### A) Add a “deference / challenge” latent variable (V_t) that gates what management *is allowed* to do

The Saar review finding that decisions “weren’t sufficiently challenged” due to “excessive deference”  is not just “mgmt belief differs”; it’s **a constraint on the action set**.

**Implementation (minimal):**

- Introduce (V_t \in [0,1]) (“board challenge intensity”; 0 = rubber-stamp, 1 = highly challenging).
- Let (V_t) enter:
  1. the probability that D3 is *actually feasible* (board will force it)
  2. the probability of “sacked” vs “resign” (board toughness vs face-saving)

**Why it matters:** with low (V_t), even very high (B^{mkt}) won’t produce decisive action because the governance system can’t self-correct.

---

### B) Make management “overprecision” explicit: mgmt belief is *less uncertain* than market belief

Overconfidence in the literature is not just overestimation; it also includes **overprecision** (excessive certainty). Your note explicitly frames hubris as a feedback loop where past success reduces challenge and increases inflated self-assessment. 

**Implementation (minimal):**

- Keep your `mgmt_bias`, but also set:
  - `sigma_B_mgmt < sigma_B_mkt` (mgmt thinks outcomes are more “under control”)
  - optionally `rho_mgmt > rho_mkt` (mgmt belief is stickier; “doubling down”)

This captures the “pattern of doubling down under pressure” behaviours described. 

---

### C) Add an “escalation-of-commitment” channel: when pressured, mgmt becomes *less responsive* to public belief

Your note lists repeated high-stakes “doubling down under pressure” decisions. 
That’s a behavioural signature that as external belief worsens, management may **not** linearly “do more”; they may **defend/deny** until forced.

**Implementation (minimal):**

- Add a single parameter `defensiveness` that *reduces* the effective impact of public signals on (B^{mgmt}), or reduces the *belief shift* mgmt attributes to governance actions.
- In code terms, this can be as simple as:
  - `effective_shift_mgmt = (1 - defensiveness) * shift_mgmt`
  - with defensiveness increasing when (B^{mkt}) is high.

This is the smallest way to encode “we’re in control” narratives even as public distrust rises.

---

### D) Let CEO self-interest enter the “action choice” stage, not only the payoff stage

Right now, you’re adding CEO costs in `loss_mgmt`. That’s necessary, but not sufficient: overconfident CEOs often don’t *perceive* the same trade-offs, or they believe they can “win it back”.

**Implementation (minimal):**

- Add one extra internal objective term that depends on CEO status:
  - `status_utility = +u_stay` if stay
  - `status_utility = -u_resign`, `-u_sacked`
- And let the firm’s *policy* be a softmax choice over D using management belief and status utility.

Even if you keep “simulate all packages and rank” externally, internally you can compute “what management would choose” as a separate diagnostic.

---

## 2) Parameter changes (concrete defaults consistent with your note)

These are *starting defaults* that match the story in `ceo-overconfidence.md`: confidence under impending ACCC heat , repeated doubling down , and excessive deference / weak challenge .

### A) Increase and tighten the overconfidence bias (overestimation + overprecision)

- **Bigger bias**: management underestimates distrust more.
- **Smaller bias sigma**: mgmt is more *certain* about its rosy view (overprecision).

```python
mgmt_bias = 0.90          # was 0.50
mgmt_bias_sigma = 0.12    # was 0.20
```

### B) Make management belief more persistent and less noisy than market belief

This encodes “stickiness” and “doubling down”.

```python
rho_mkt = 0.90
sigma_B_mkt = 0.45

rho_mgmt = 0.96           # stickier
sigma_B_mgmt = 0.25       # overprecision
```

### C) Add deference (V) and make it low by default at C0/C1

Given Saar’s finding, a plausible default is: early on, (V) is low (challenge weak); as crises accumulate it can increase (or not).

Minimal parameterisation:

```python
V0_mean = 0.25     # low challenge / high deference
V0_sd   = 0.10
rho_V   = 0.85     # institutional inertia
sigma_V = 0.05
```

And then **use (V)** to gate:

- “CEO transition happens” probability
- “sacked” probability

Example gating (conceptually):

- `ceo_transition_intercept += 2.0 * V`
- `ceo_mode_intercept += 1.5 * V` (higher V ⇒ more likely sacked vs resign)

### D) Make “sacked” harder under deference (face-saving + CEO power)

Your earlier mode parameters assumed severity drives sacking. With deference, you want: even at high (B^{mgmt}), if (V) is low, boards still prefer resign/retire framing.

So update the *effective* logit for sacked:

[
\text{logit }P(\text{sacked}) = a + b B^{mgmt} + c V
]

Concrete defaults (layered on top of what I gave you earlier):

```python
ceo_mode_intercept      = -2.4
ceo_mode_slope_Bmgmt    = 0.6
ceo_mode_slope_V        = 1.6   # NEW: challenge enables sacking
```

### E) Make management’s perceived effectiveness of governance actions smaller (defensiveness)

Overconfident, deferred systems tend to believe “PR + process” will work, but also discount the need for deeper accountability until late. You already separate `delta_mkt` vs `delta_mgmt`; now bias the management deltas to be *too optimistic about soft actions* and *too pessimistic about hard actions*.

A simple encoding:

```python
# mgmt thinks D1/D2 will work better than market thinks (optimistic self-efficacy)
delta_mgmt["indep_review"]   = 1.2 * delta_mkt["indep_review"]
delta_mgmt["timeline"]       = 1.2 * delta_mkt["timeline"]
delta_mgmt["transparency"]   = 1.1 * delta_mkt["transparency"]

# mgmt dislikes/discounts effectiveness of hard accountability moves
delta_mgmt["clawback"]       = 0.7 * delta_mkt["clawback"]
# CEO transition handled by mode; but mgmt will resist it via feasibility gating (V) + agency costs
```

This matches “excessive deference + past crises overcome” leading to overbelief that the same playbook works again. 

---

## 3) One extra output you should add immediately: “governance miscalibration gap”

To validate that you’re actually capturing the behavioural story, print (per checkpoint, per package):

- `E[B_mkt_agm] - E[B_mgmt_agm]`  (belief divergence)
- `E[loss_mkt] - E[loss_mgmt]`    (agency divergence)
- `P(transition)` and `P(sacked)` as functions of (V)

If those aren’t moving in the directions implied by the note (e.g., low (V) suppresses sacking early; mgmt belief is rosier and more certain), the structure isn’t actually doing the job.

---

## Deference

**(V)** is *board challenge intensity*, so low (V) = high deference, high (V) = strong independent challenge.

Making it a **constant per checkpoint** is the right first step. It keeps the system tractable and lets you do sensitivity sweeps without introducing another AR(1) layer.

Below is the **minimal structural change** needed to implement constant (V) cleanly.

---

# 1️⃣ Conceptual role of V (constant)

Think of (V \in [0,1]) as:

- 0.0 → rubber-stamp board (max deference)
- 0.5 → normal governance
- 1.0 → highly independent, activist board

It should affect:

1. **Probability CEO transition actually occurs**
2. **Probability transition is “sacked” vs “resign”**
3. (Optionally later) magnitude of governance credibility shifts

It should **not** directly affect market belief — market does not observe internal challenge intensity directly.

---

# 2️⃣ Minimal additions to ToyParams

Add one scalar and one slope:

```python
# --- NEW: board challenge / deference ---
V: float = 0.30                 # constant per checkpoint
ceo_transition_slope_V: float = 1.5
ceo_mode_slope_V: float = 1.8

```

Interpretation:

- With V = 0.30 → relatively deferential (consistent with “excessive deference” finding).
- As V increases, board more likely to force transition and more willing to sack.

---

# 3️⃣ Modify CEO transition probability

Current transition logit:

```python
logit_p_trans = ceo_transition_intercept
               + ceo_transition_slope_Bmgmt * B_mgmt_agm

```

Change to:

```python
logit_p_trans = (
    self.p.ceo_transition_intercept
    + self.p.ceo_transition_slope_Bmgmt * B_mgmt_agm
    + self.p.ceo_transition_slope_V * self.p.V
)

```

### Interpretation

- When V is low (0.30), board is reluctant to force removal.
- When V is high (0.70+), transition probability jumps materially.

---

# 4️⃣ Modify resign vs sacked logit

Current:

```python
logit_p_sacked = (
    self.p.ceo_mode_intercept
    + self.p.ceo_mode_slope_Bmgmt * B_mgmt_agm
)

```

Change to:

```python
logit_p_sacked = (
    self.p.ceo_mode_intercept
    + self.p.ceo_mode_slope_Bmgmt * B_mgmt_agm
    + self.p.ceo_mode_slope_V * self.p.V
)

```

### Interpretation

- Low V → even high crisis tends to be framed as “resign”
- High V → board willing to classify as “sacked”

This captures exactly the “excessive deference” channel in the note.

---

# 5️⃣ Recommended default values (calibrated to your narrative)

Given the governance review found excessive deference, start with:

```python
V = 0.30   # baseline

```

Slopes:

```python
ceo_transition_slope_V = 1.5
ceo_mode_slope_V = 1.8

```

Keep your earlier severity slopes:

```python
ceo_transition_slope_Bmgmt = 0.6
ceo_mode_slope_Bmgmt = 0.6

```

And intercepts:

```python
ceo_transition_intercept = 1.8
ceo_mode_intercept = -2.4

```

---

# 6️⃣ What you should observe immediately

With V = 0.30:

- Early checkpoints (C0/C1):
  - High probability of “resign” if D3 chosen
  - Very low probability of “sacked”
- Late checkpoints (C2/C3):
  - Transition probability rises due to B_mgmt
  - But sacked remains moderate unless V increases

If you rerun with:

```python
V = 0.70

```

You should see:

- Higher P(transition)
- Much higher P(sacked)
- Larger agency gap (management hates sacked)

That sensitivity difference is the empirical test that your deference channel is working.

---

# 7️⃣ Why constant V is actually powerful

Keeping V constant per checkpoint allows you to:

- Treat deference as a **governance regime parameter**
- Run scenario sweeps:
  - “What if the board had been more independent?”
  - “What if governance reform increased V?”

It becomes a structural stress-testing lever — exactly aligned with your adversarial risk framework.

---

If you want, I can next:

- formalise the exact equations in compact mathematical notation (so this layer is publication-ready), or
- show you how to integrate V into your CSV “single source of truth” workflow so it’s controlled externally rather than hard-coded.

