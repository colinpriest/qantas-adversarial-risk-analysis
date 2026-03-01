Excellent. Now we move from theory → production system.

What you need is **one unified adversarial engine** that:

• Shares Bayesian state estimation (Stan)
• Shares chance models (vote + review)
• Switches focal actor (Board vs ASA)
• Switches opponent roles automatically
• Executes node-indexed ARA recursion
• Remains checkpoint-consistent (01-Oct-2023 etc.)

Below is a **complete, implementation-grade software specification**.
This is written so a numerate engineer can implement without asking conceptual questions.

---

# I. SYSTEM ARCHITECTURE OVERVIEW

```
project_root/
│
├── data/
│   ├── checkpoints/                 # belief .npz draws (existing)
│   ├── governance_spec.xlsx         # action sets, thresholds, utilities
│   ├── opponent_priors.xlsx         # priors for Board/ASA/CEO parameters
│   └── stan_inputs/                 # JSON for Stan
│
├── stan/
│   ├── belief_model.stan
│   ├── vote_link_model.stan
│   ├── review_model.stan
│   └── fit_models.py
│
├── engine/
│   ├── state.py
│   ├── chance_models.py
│   ├── utilities.py
│   ├── predictive.py
│   ├── tree.py
│   ├── solver.py
│   └── modes.py
│
├── run/
│   ├── run_board_mode.py
│   ├── run_asa_mode.py
│   └── sensitivity.py
│
└── outputs/
```

The system separates:

**Layer A — Bayesian estimation (Stan)**
**Layer B — Adversarial tree engine (Python)**

---

# II. STAN LAYER SPECIFICATION

This layer produces posterior draws and checkpoint belief states.

---

## 1️⃣ Belief Model (belief_model.stan)

### Purpose

Estimate latent distrust process (B_t) up to checkpoint date.

### Model

AR(1) state equation:
[
B_t = \rho B_{t-1} + \beta shock_t + \sigma_B \epsilon_t
]

Observation channels:

* Abnormal returns
* Rem vote %

Vote anchoring:
[
\text{logit}(rem_vote_t) = \alpha_{\text{rem}} + B_t + \sigma_{rem}\eta_t
]

### Outputs required

Export posterior draws:

```
B_t_draws[N_draws]
rho_draws
sigma_B_draws
alpha_rem_draws
sigma_rem_draws
```

At checkpoint date only.

Save as:

```
data/checkpoints/belief_CX_YYYY-MM-DD.npz
```

Containing:

```
B_mkt[N]
B_mgmt[N]
parameter_draw_indices
```

---

## 2️⃣ Vote Mapping Model (optional separate Stan file)

If desired, fit separate hierarchical mapping:

[
\text{logit}(V) = \alpha + B_{\text{agm}} + \gamma_A A_2 + \gamma_D D_1 + \epsilon
]

Export:

```
alpha_vote_draws
gamma_A_draws
gamma_D_draws
sigma_vote_draws
```

---

## 3️⃣ Review Findings Model

Binary:
[
R \sim Bernoulli(\pi_R)
]
or
[
\text{logit}(\pi_R) = \delta_0 + \delta_1 D_1 + \delta_2 V + \epsilon
]

Export draws.

---

# III. DATA CONTRACTS (STRICT)

All configuration must live in Excel/CSV.

---

## governance_spec.xlsx

### Sheet: action_sets

| node | action | feasibility_rule |
| ---- | ------ | ---------------- |
| D1   | D0     | always           |
| D1   | D1     | not_reviewed     |
| D1   | D2     | CEO_present      |
| ...  | ...    | ...              |

---

### Sheet: vote_thresholds

| name         | value |
| ------------ | ----- |
| first_strike | 0.25  |
| overwhelming | 0.50  |

---

### Sheet: utilities_board

Weights for:

* vote_penalty_weight
* review_penalty_weight
* implementation_cost
* CEO_loss_cost
* spill_risk_cost

---

### Sheet: utilities_asa

Weights for:

* governance_change_reward
* vote_reward
* mobilisation_cost
* reputational_reward

---

### Sheet: utilities_ceo

Weights for:

* job_loss_cost
* reputational_cost
* resignation_cost
* forced_removal_cost

---

## opponent_priors.xlsx

For each actor:

| actor | parameter         | distribution | param1 | param2 |
| ----- | ----------------- | ------------ | ------ | ------ |
| ASA   | mobilisation_cost | normal       | μ      | σ      |
| CEO   | job_loss_cost     | lognormal    | μ      | σ      |
| Board | deference         | normal       | μ      | σ      |

These define (p_B(\Theta_A)), (p_A(\Theta_B)), etc.

---

# IV. PYTHON ENGINE SPECIFICATION

---

# 1️⃣ Core Data Structures

---

## class DecisionState

Fields:

```
CEO_present: bool
review_commissioned: bool
review_completed: bool
CEO_removed: bool
checkpoint_id: str
```

Methods:

```
feasible_actions(node_name) -> list[str]
apply(node_name, action) -> new DecisionState
```

Feasibility rules read from governance_spec.xlsx.

---

## class BeliefBundle

Loads:

```
B_mkt[N]
B_mgmt[N]
vote_parameter_draws
review_parameter_draws
```

Indexed by draw i.

---

## class ParameterSampler

For each actor j:

```
sample_parameters(actor, focal_actor) -> Θ_j_draw
```

Uses opponent_priors.xlsx.

---

# 2️⃣ Chance Models

---

## vote_model.sample(draw_i, history, state)

Steps:

1. Construct AGM belief:

[
B_{\text{agm}} =
B_0^{mkt}[i]

* \gamma_A^{(i)} 1_{A_2=RecStrike}
* \gamma_D^{(i)} f(D_1)
  ]

2. Compute:

[
\text{logit}(V) \sim \mathcal{N}(\alpha_{vote}^{(i)} + B_{\text{agm}}, \sigma_{vote}^{(i)})
]

3. Return:

```
{
  vote_percent,
  strike_indicator,
  overwhelming_indicator
}
```

---

## review_model.sample(draw_i, history, state)

If review not commissioned → deterministic “None”.

If commissioned:

[
R \sim Bernoulli(\pi_R^{(i)})
]

Return:

```
{review_adverse: 0/1}
```

---

# 3️⃣ Utility Functions

---

## utilities.board(outcome, Θ_B)

Compute:

```
vote_penalty = weight_vote * loss_function(vote_percent)
review_penalty = weight_review * review_adverse
implementation_cost = ...
ceo_loss_cost = ...
```

Return scalar.

---

## utilities.asa(outcome, Θ_A)

Reward for:

* high vote opposition
* CEO removal
* review adverse

Minus mobilisation cost if RecStrike.

---

## utilities.ceo(outcome, Θ_C)

Penalty for:

* job loss
* adverse review
* overwhelming vote
* forced removal vs voluntary resignation

---

# 4️⃣ Predictive Distribution Engine

File: predictive.py

---

## Generic function

```
predict(node_name, history, focal_actor, belief_draw_i, state)
```

Algorithm:

1. Identify owner of node.
2. Sample K opponent parameter draws Θ_j^(k).
3. For each k:

   * For each feasible action x:

     * Compute opponent expected utility Ψ_j(x; h, Θ_j^(k))
       (calls recursive tree evaluation assuming action fixed)
   * Determine best response x*(k)
4. Return empirical distribution over x*(k).

K default = 200 (tunable).

---

# 5️⃣ Tree Engine (node-indexed recursion)

File: tree.py

---

Define function:

```
value(node_name, history, state, focal_actor, draw_i)
```

Switch on node:

### If terminal:

Return utility for focal actor.

### If decision node owned by focal:

Return max over feasible actions.

### If decision node owned by opponent:

Get predictive distribution.
Return weighted sum over actions.

### If chance node:

Sample/integrate accordingly.

---

Node order:

```
D1
A2
V
M_agm
D_rev
R
M_rev
D4
Terminal
```

---

# 6️⃣ Solver

File: solver.py

---

## solve(focal_actor, checkpoint_id)

Steps:

1. Load BeliefBundle.
2. For each candidate initial action:

   * For each belief draw i:

     * Compute value via tree recursion.
   * Average over i.
3. Return:

```
{
  EU_per_action,
  optimal_action,
  outcome_distributions,
  diagnostics
}
```

---

# 7️⃣ Modes

File: modes.py

---

```
MODE_BOARD:
    focal = "Board"
    opponents = {"ASA": "ARA", "CEO": "ARA"}

MODE_ASA:
    focal = "ASA"
    opponents = {"Board": "ARA", "CEO": "ARA"}
```

Optional:

```
MODE_ASA_POLICY_BOARD:
    opponents = {"Board": "Policy", "CEO": "ARA"}
```

---

# V. PERFORMANCE & NUMERICS

---

Monte Carlo structure:

Outer draws:
N ≈ 5000 (belief draws)

Inner opponent draws:
K ≈ 100–300

Total complexity per scenario:
[
O(N \times K \times \text{tree depth})
]

Use:

* NumPy vectorisation where possible
* JAX optional later
* Cache subtree values conditional on action histories

---

# VI. OUTPUTS

---

## outputs/game_summary.csv

| checkpoint | focal | action | EU | Pr_strike | Pr_overwhelming | Pr_CEO_removed | ... |

---

## outputs/simulation_draws.parquet

One row per belief_draw × initial_action.

Contains:

* realised V
* realised R
* realised CEO actions
* utilities for all actors
* state flags

---

# VII. SENSITIVITY ENGINE

Run:

```
sensitivity.py
```

Grid over:

* belief persistence ρ
* mobilisation shock γ_A
* CEO job_loss_cost
* Board deference

Re-run solve and store optimal policy shifts.

---

# VIII. VALIDATION CHECKLIST

Engineer must confirm:

* Board-mode D1* changes when γ_A large.
* ASA-mode mobilisation increases when CEO removal reward large.
* Symmetry: swapping focal switches which nodes are max vs sum.
* Removing CEO reduces future CEO decision nodes automatically.

---

# IX. RESULTING SYSTEM PROPERTIES

You now have:

✔ Single Bayesian state engine (Stan)
✔ Single adversarial tree engine (Python)
✔ Dual focal perspectives
✔ Full ARA opponents
✔ Vote endogenous but non-strategic
✔ Review endogenous
✔ Temporal checkpoint discipline
✔ Sensitivity analysis capability

Below are the three add-ons you asked for:

1. a **formal component diagram (UML-ish)**
2. **computational pseudocode** for evaluating (\Psi_j(\cdot)) (opponent expected utilities used in predictive distributions)
3. a clean design for **Level-1 vs Level-2 opponent modelling** (and how to implement both without rewriting the engine)

I’m going to stay consistent with the tree order you’re using:
(D_1 \to A_2 \to V \to M_{\text{agm}} \to D_{\text{rev}} \to R \to M_{\text{rev}} \to D_4).

---

## 1) UML-style component diagram

### 1.1 High-level components and dependencies

```text
+---------------------------------------------------------------+
|                           CLI / run/                           |
|  run_board_mode.py  run_asa_mode.py  sensitivity.py            |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                        Orchestrator                            |
|  - loads checkpoint, selects focal, configures mode            |
|  - runs solve() across candidate initial actions               |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                          Solver                                |
|  solve(focal, checkpoint, mode_config)                         |
|  - loops over initial actions & Monte Carlo draws              |
|  - calls TreeEvaluator.value(root, ...)                        |
|  - aggregates EU + diagnostics                                 |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                        TreeEvaluator                           |
|  value(node, history, state, draw_i, focal, mode)              |
|  - terminal -> Utility                                          |
|  - focal decision -> max over actions                          |
|  - opponent decision -> PredictiveDist + expectation           |
|  - chance -> ChanceModels integrate/sample                     |
+-------------------+----------------------+---------------------+
                    |                      |
                    v                      v
+------------------------+         +----------------------------+
|    PredictiveDist      |         |       ChanceModels          |
|  predict(node, h, ...) |         |  vote_model, review_model   |
|  - samples Θ_owner     |         |  sample_v(), sample_r()     |
|  - computes Ψ_owner    |         |  (uses BeliefBundle draws)  |
|  - returns pmf over x  |         +----------------------------+
+-----------+------------+
            |
            v
+---------------------------------------------------------------+
|                   OpponentModel (Levels)                       |
|  compute_Psi(owner, action, h, state, draw_i, focal, level)    |
|  - Level-1: owner assumes others follow fixed policies         |
|  - Level-2: owner models (one) other actor as ARA too          |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                       UtilityModels                            |
|  u_board(Z, Θ_B)  u_asa(Z, Θ_A)  u_ceo(Z, Θ_C)                 |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                     Data + Parameter Layer                     |
|  BeliefBundle (.npz + Stan draws)                              |
|  Priors (opponent_priors.xlsx)                                 |
|  Governance spec (governance_spec.xlsx)                        |
|  ParameterSampler                                                |
+---------------------------------------------------------------+

+---------------------------------------------------------------+
|                            Stan Layer                           |
|  belief_model.stan / vote_link_model.stan / review_model.stan   |
|  fit_models.py -> exports checkpoint .npz + parameter draws      |
+---------------------------------------------------------------+
```

### 1.2 Class-level diagram (key interfaces)

```text
class DecisionState
  + checkpoint_id: str
  + CEO_present: bool
  + review_commissioned: bool
  + review_completed: bool
  + CEO_removed: bool
  + feasible_actions(node_name) -> list[str]
  + apply(node_name, action) -> DecisionState

class BeliefBundle
  + N: int
  + B_mkt[N], B_mgmt[N]
  + vote_params_draws[N]  # alpha_vote, gamma_A, gamma_D, sigma_vote, etc.
  + review_params_draws[N]
  + get_draw(i) -> dict

class ChanceModels
  + sample_vote(draw_i, history, state) -> vote_outcome
  + sample_review(draw_i, history, state) -> review_outcome

class UtilityModels
  + u(actor, terminal_outcome, actor_params) -> float

class ParameterSampler
  + sample(actor, perspective_actor, rng) -> Theta_actor

class OpponentModel
  + compute_Psi(owner, action, context) -> float
    (uses TreeEvaluator in a controlled way)

class PredictiveDist
  + predict(node, history, state, draw_i, focal, mode, level) -> dict[action, prob]

class TreeEvaluator
  + value(node, history, state, draw_i, focal, mode) -> float

class Solver
  + solve(focal, checkpoint, mode, level) -> results
```

---

## 2) Computational pseudocode for (\Psi_j) evaluation

You need (\Psi_j(x;h,\Theta_j)) inside the predictive distribution (p_i(X\mid h)). That’s the “opponent’s expected utility if they take action (x) now” under the focal actor’s world model.

There are two practical ways to compute (\Psi):

* **A) exact via recursive evaluation** (calls the same TreeEvaluator but with a “forced action” at the current node and a different utility function)
* **B) approximate via rollout simulation** (sample downstream once or a few times; faster but noisier)

You can implement both behind the same interface.

### 2.1 Context object used everywhere

```text
Context:
  focal_actor: "Board" or "ASA"
  mode_config: which nodes are focal decisions vs opponent decisions vs chance
  level: 1 or 2 (opponent modelling depth)
  draw_i: Monte Carlo index for belief/stan parameter draw
  history h
  state S
  forced_action: optional (node_name, action)
  utility_target_actor: whose utility we are evaluating inside Psi (owner of node)
  rng
```

### 2.2 A) “Exact” (\Psi) by reuse of tree recursion

This is the cleanest because it remains consistent with your node-indexed Bellman setup.

```text
function COMPUTE_PSI_EXACT(owner j, node X, action x, context):
    # context includes history h at node X, state S, draw_i, etc.

    # 1) Create a modified evaluation context:
    ctx2 = context.clone()
    ctx2.forced_action = (X, x)
    ctx2.utility_target_actor = j

    # 2) Evaluate the continuation value from node X forward:
    #    Use TreeEvaluator but:
    #    - at node X, do NOT max/expect; just apply forced action x
    #    - at nodes owned by j later: choose according to owner policy assumption:
    #         Level-1: owner uses a fixed policy π_j (rule or precomputed)
    #         Level-2: owner uses ARA predictive modelling of one other actor (see section 3)
    #    - chance nodes use ChanceModels with draw_i
    #
    # Important: the evaluation should end at terminal and use u_j(·).

    psi = TREE_VALUE_FROM_NODE(node=X, ctx=ctx2)

    return psi
```

Key implementation detail: `TREE_VALUE_FROM_NODE` is the same as `TreeEvaluator.value()` but with two switches:

* `utility_target_actor` (use (u_j) not (u_{\text{focal}}))
* `forced_action` at this node

### 2.3 B) Approximate (\Psi) by stochastic rollouts

This is useful if the exact recursion is slow because you’re doing it K times per predictive distribution.

```text
function COMPUTE_PSI_ROLLOUT(owner j, node X, action x, context, R_rollouts):
    total = 0
    for r in 1..R_rollouts:
        h = context.history
        S = context.state
        # apply forced action
        (h, S) = APPLY_ACTION(X, x, h, S)

        # simulate forward:
        while not TERMINAL(h, S):
            node = NEXT_NODE(h, S)

            if node is chance:
                outcome = SAMPLE_CHANCE(node, context.draw_i, h, S)
                (h, S) = APPLY_CHANCE(node, outcome, h, S)

            else if node is decision:
                owner2 = OWNER(node)
                if owner2 == j:
                    a2 = POLICY_FOR_OWNER(j, node, h, S, context.level)
                else:
                    a2 = POLICY_FOR_OTHER(owner2, node, h, S, context.mode_config, context.level)

                (h, S) = APPLY_ACTION(node, a2, h, S)

        total += UTILITY(j, TERMINAL_OUTCOME(h,S), Theta_j=context.Theta_owner_draw)
    return total / R_rollouts
```

Rollout policies can be:

* deterministic heuristics (fast)
* or “one-step ARA” at each decision (slower but closer)

Most teams do: **exact recursion for focal evaluation**, rollouts for inner (\Psi).

### 2.4 How (\Psi) is used inside the predictive distribution

```text
function PREDICT_ACTION_DIST(focal i, opponent owner j, node X, history h, draw_i, state S, level, K):
    counts[action]=0

    for k in 1..K:
        Theta_j ~ p_i(Theta_j)         # focal's belief about opponent params

        best_action = None
        best_value  = -inf

        for x in FEASIBLE_ACTIONS(X, h, S):
            context = {focal=i, owner=j, Theta_j, h, S, draw_i, level}
            psi = COMPUTE_PSI_EXACT(...) or COMPUTE_PSI_ROLLOUT(...)
            if psi > best_value:
                best_value = psi
                best_action = x

        counts[best_action] += 1

    return counts / K
```

That’s the full computational definition of (p_i(\cdot\mid h)).

---

## 3) Level-1 vs Level-2 opponent modelling (clean spec)

This is where most implementations get tangled. The trick is: **never change TreeEvaluator**.
Instead: change **what OpponentModel assumes** when computing (\Psi_j).

### 3.1 Definitions

* **Level-1 ARA**:
  Focal actor models each opponent as EU-maximising under uncertain (\Theta), but when the opponent computes their own expected utilities (\Psi_j), they do **not** recursively model others as ARA optimisers. They assume **fixed response policies** for other actors.

* **Level-2 ARA** (practical, not infinite regress):
  Opponent (j), inside (\Psi_j), treats **one other key actor (k)** as strategic via a predictive distribution, but treats remaining actors as fixed policies. This gives “I think you think” flavour without exploding.

This is enough for your storytelling and consulting realism.

### 3.2 Implementation pattern: pluggable “decision rule providers”

Define a single interface:

```text
DecisionRuleProvider.get_action(owner, node, history, state, context) -> action or distribution
```

You will have three providers:

1. `FocalMaximiserProvider` (only used when owner == focal and we are solving the main problem)
2. `OpponentARAProvider(level=1 or 2)` (used when owner is being modelled strategically)
3. `FixedPolicyProvider` (used as the “default behaviour” inside someone else’s (\Psi))

**Crucially:** `OpponentARAProvider` calls `PredictiveDist.predict()` which calls `OpponentModel.compute_Psi()` which calls back into the tree recursion. That loop must be controlled with the `level` parameter.

### 3.3 Level-1 spec (exact)

Inside (\Psi_j), when future nodes occur:

* if the future node is owned by (j): pick ( \arg\max ) using the same (\Psi_j) mechanism (but you can also simplify by assuming myopic choices at later nodes)
* if the future node is owned by **anyone else**: use a fixed policy (\pi(\cdot)) (or a calibrated stochastic policy)

So you need:

```text
Fixed policies required in Level-1:
  π_B(node, h, S) for Board actions (if Board is not focal in this Ψ)
  π_A(node, h, S) for ASA actions (if ASA is not focal in this Ψ)
  π_C(node, h, S) for CEO actions (if CEO is not focal in this Ψ)
```

These policies can be as simple as:

* “resign iff expected forced removal risk > threshold”
* “commission review iff vote > τ1 and not yet commissioned”
* “ASA recommend strike iff belief > threshold and mobilisation cost low”

They live in `policies/` or a `policy_models.py` and take parameters from priors if you want heterogeneity.

### 3.4 Level-2 spec (controlled recursion)

Pick a single “strategic counterpart” for each owner when computing (\Psi).

Recommended mapping for your case:

* When Board is focal and modelling **ASA’s** move (computing (\Psi_A)):
  ASA models **CEO** strategically (Level-2) OR models Board strategically (but not both).

  * For storytelling realism: ASA models Board (because ASA is pressuring Board).
  * For operational realism: ASA models CEO only weakly.

* When Board is focal and modelling **CEO** (computing (\Psi_C)):
  CEO models **Board** strategically (this is the obvious one).

* When ASA is focal and modelling **Board** (computing (\Psi_B)):
  Board models **ASA** strategically (again obvious).

So in Level-2, inside (\Psi_j):

* if node owner is the chosen strategic counterpart (k): action is drawn from predictive distribution (p_j(\cdot\mid h)) induced by (k)’s EU-max (from (j)’s view)
* all other non-(j) nodes use fixed policies

This yields a depth-2 “I think you think” but avoids infinite regress.

### 3.5 Concrete algorithmic control to avoid infinite loops

You must include a recursion guard. Here is the spec:

* `context.level` is an integer.
* `OpponentARAProvider` when computing (\Psi_j) calls predictive modelling on counterpart (k) only if `context.level >= 2`.
* When it does so, it calls `PredictiveDist.predict(..., level=context.level-1)` (decrement level).
* When `level == 1`, predictive modelling is not allowed inside (\Psi); all non-owner decisions are fixed policies.

This guarantees termination.

### 3.6 Pseudocode for Level-2 inside (\Psi_j)

```text
function OWNER_DECISION_RULE(owner=j, node, history h, state S, ctx):
    # used when node owner is j
    # j chooses best action for itself at that node
    return argmax_x Psi_j(x; h, Theta_j)

function OTHER_DECISION_RULE(owner=other, node, h, S, ctx):
    if ctx.level >= 2 and other == ctx.strategic_counterpart:
        # model "other" as strategic, one level down
        dist = PredictiveDist.predict(node, h, S,
                                      focal_actor=j,
                                      level=ctx.level-1)
        return sample(dist)
    else:
        return FixedPolicyProvider.get_action(other, node, h, S, ctx)
```

### 3.7 Recommended defaults for your project

* Main solver for the client (Board or ASA): **exact recursion** for `TreeEvaluator.value`.

* Predictive distributions: use **K=200** opponent parameter samples.

* Inner (\Psi) calculation:

  * Start with **rollouts R=20** to get the system working.
  * Move to “exact recursion (\Psi)” only if needed for accuracy.

* Depth:

  * Default: **Level-1** (most stable, easiest to debug)
  * “Premium narrative”: **Level-2** only for CEO↔Board and Board↔ASA pairs, with strict recursion guard.

---

## Deliverable checklist for the engineer

To be “done”, the codebase must include:

1. **Mode switch** (`focal="Board"` vs `focal="ASA"`) that flips max vs expectation at nodes per your recursion.
2. Predictive distribution code that:

   * samples (\Theta_j) from the correct perspective prior (p_i(\Theta_j))
   * computes (\Psi_j) and best responses
   * returns a pmf over actions
3. (\Psi) evaluator that supports:

   * `method="exact"` and `method="rollout"`
   * `level=1` and `level=2` with recursion guard
4. Policies for Level-1 (and non-strategic actors in Level-2) that are:

   * deterministic or stochastic
   * parameterised
   * stored in config or priors (not hardcoded magic numbers)
5. Output artifacts:

   * `game_summary.csv`
   * `simulation_draws.parquet`
   * plus logs of predictive pmfs at each node for debugging (“why did it think ASA would do X?”)

---

Excellent. This is where projects either become robust infrastructure or descend into chaos.

Below is a **fully explicit I/O schema specification** for:

1. `governance_spec.xlsx`
2. `opponent_priors.xlsx`
3. `.npz` checkpoint metadata JSON schema

This is written so an engineer can validate inputs programmatically and fail fast if anything deviates.

---

# PART I — `governance_spec.xlsx` SCHEMA

This workbook MUST contain the following sheets (case-sensitive):

```
action_sets
node_order
vote_thresholds
utilities_board
utilities_asa
utilities_ceo
policy_parameters
```

All numeric columns must be float64 unless specified.

---

## 1️⃣ Sheet: `node_order`

Defines canonical tree order.

### Columns

| column_name | type | permissible_values                                | required |
| ----------- | ---- | ------------------------------------------------- | -------- |
| order_index | int  | 0–99                                              | yes      |
| node_name   | str  | {D1, A2, V, M_agm, D_rev, R, M_rev, D4, Terminal} | yes      |
| node_type   | str  | {decision, chance, terminal}                      | yes      |
| owner       | str  | {Board, ASA, CEO, Nature}                         | yes      |

### Constraints

* order_index must be unique and strictly increasing.
* Terminal must appear exactly once.
* V and R must have owner = Nature.

---

## 2️⃣ Sheet: `action_sets`

Defines all feasible actions and feasibility rules.

### Columns

| column_name      | type | permissible_values                                       | required |
| ---------------- | ---- | -------------------------------------------------------- | -------- |
| node_name        | str  | must match node_order.node_name where node_type=decision | yes      |
| action_name      | str  | arbitrary string without spaces                          | yes      |
| feasibility_code | str  | see list below                                           | yes      |
| description      | str  | free text                                                | no       |

### Permissible feasibility_code values

| code                    | meaning                                                    |
| ----------------------- | ---------------------------------------------------------- |
| always                  | action always feasible                                     |
| CEO_present             | state.CEO_present == True                                  |
| CEO_not_removed         | state.CEO_removed == False                                 |
| review_not_commissioned | state.review_commissioned == False                         |
| review_commissioned     | state.review_commissioned == True                          |
| review_completed        | state.review_completed == True                             |
| not_reviewed_yet        | review_commissioned == False AND review_completed == False |

The engine must map these codes to boolean functions.

---

## 3️⃣ Sheet: `vote_thresholds`

### Columns

| column_name    | type  | permissible_values           | required |
| -------------- | ----- | ---------------------------- | -------- |
| threshold_name | str   | {first_strike, overwhelming} | yes      |
| value          | float | 0 < value < 1                | yes      |

Constraints:

* first_strike < overwhelming
* both strictly between 0 and 1

---

## 4️⃣ Sheet: `utilities_board`

Defines Board utility weights.

### Columns

| column_name    | type  | permissible_values | required |
| -------------- | ----- | ------------------ | -------- |
| parameter_name | str   | see list below     | yes      |
| value          | float | any real           | yes      |

### Permissible parameter_name values

```
vote_penalty_weight
review_penalty_weight
implementation_cost_review
implementation_cost_sack
ceo_loss_cost
spill_risk_weight
overwhelming_penalty_weight
reputational_spill_weight
```

If any missing → error.

---

## 5️⃣ Sheet: `utilities_asa`

### parameter_name permissible values

```
vote_reward_weight
ceo_removal_reward
review_adverse_reward
mobilisation_cost
reputational_gain_weight
overwhelming_reward_weight
```

Same column structure as utilities_board.

---

## 6️⃣ Sheet: `utilities_ceo`

### parameter_name permissible values

```
job_loss_cost
reputational_cost_weight
resignation_cost
forced_removal_cost
overwhelming_vote_cost_weight
adverse_review_cost_weight
```

---

## 7️⃣ Sheet: `policy_parameters`

Used for Level-1 fixed policy behaviours.

### Columns

| column_name    | type  | permissible_values | required |
| -------------- | ----- | ------------------ | -------- |
| actor          | str   | {Board, ASA, CEO}  | yes      |
| node_name      | str   | decision node      | yes      |
| parameter_name | str   | arbitrary          | yes      |
| value          | float | any                | yes      |

Example parameters:

For Board:

```
review_vote_threshold
sack_vote_threshold
```

For CEO:

```
resign_vote_threshold
resign_adverse_prob_threshold
```

For ASA:

```
mobilise_vote_threshold
```

Engine must allow arbitrary parameter_name but document usage in policy code.

---

# PART II — `opponent_priors.xlsx` SCHEMA

Single sheet: `priors`

---

## Columns

| column_name       | type  | permissible_values                        | required |
| ----------------- | ----- | ----------------------------------------- | -------- |
| perspective_actor | str   | {Board, ASA}                              | yes      |
| target_actor      | str   | {Board, ASA, CEO}                         | yes      |
| parameter_name    | str   | free string                               | yes      |
| distribution      | str   | {normal, lognormal, beta, uniform, gamma} | yes      |
| param1            | float | depends on distribution                   | yes      |
| param2            | float | depends on distribution                   | yes      |
| param3            | float | optional                                  | no       |

---

## Distribution parameter interpretation

| distribution | param1  | param2 | param3 |
| ------------ | ------- | ------ | ------ |
| normal       | mean    | sd     | -      |
| lognormal    | meanlog | sdlog  | -      |
| beta         | alpha   | beta   | -      |
| uniform      | lower   | upper  | -      |
| gamma        | shape   | scale  | -      |

---

## Validation rules

* perspective_actor ≠ target_actor allowed
* Must contain priors for:

  * Board perspective on ASA and CEO
  * ASA perspective on Board and CEO
* If missing → error
* param2 must be > 0 where applicable
* uniform lower < upper

---

# PART III — `.npz` CHECKPOINT METADATA SCHEMA

Each checkpoint file:

```
belief_CX_YYYY-MM-DD.npz
```

Must contain:

```
B_mkt              float64[N]
B_mgmt             float64[N]
alpha_vote         float64[N]
gamma_A            float64[N]
gamma_D            float64[N]
sigma_vote         float64[N]
review_param_1     float64[N]
review_param_2     float64[N]
draw_id            int64[N]
metadata_json      str (UTF-8 JSON string)
```

---

## JSON Schema for metadata_json

### JSON structure

```json
{
  "schema_version": "1.0",
  "checkpoint_id": "C0",
  "checkpoint_date": "2023-10-01",
  "n_draws": 8000,
  "stan_model_versions": {
    "belief_model": "v1.3.2",
    "vote_model": "v1.0.1",
    "review_model": "v1.0.0"
  },
  "priors_used": {
    "gamma_A_prior_source": "asa_panel_estimation_v2.csv",
    "review_prior_source": "historical_reviews_2010_2022.csv"
  },
  "ar1_parameters_summary": {
    "rho_mean": 0.82,
    "sigma_B_mean": 0.14
  },
  "random_seed": 123456,
  "generation_timestamp_utc": "2026-03-01T04:32:00Z"
}
```

---

## JSON Schema (formal)

```json
{
  "type": "object",
  "required": [
    "schema_version",
    "checkpoint_id",
    "checkpoint_date",
    "n_draws",
    "stan_model_versions",
    "priors_used",
    "random_seed",
    "generation_timestamp_utc"
  ],
  "properties": {
    "schema_version": {"type": "string"},
    "checkpoint_id": {"type": "string"},
    "checkpoint_date": {"type": "string", "format": "date"},
    "n_draws": {"type": "integer", "minimum": 1},
    "stan_model_versions": {
      "type": "object",
      "properties": {
        "belief_model": {"type": "string"},
        "vote_model": {"type": "string"},
        "review_model": {"type": "string"}
      }
    },
    "priors_used": {"type": "object"},
    "ar1_parameters_summary": {"type": "object"},
    "random_seed": {"type": "integer"},
    "generation_timestamp_utc": {"type": "string", "format": "date-time"}
  }
}
```

---

# ENGINE VALIDATION REQUIREMENTS

At load time:

1. Validate governance_spec.xlsx schema strictly.
2. Validate opponent_priors.xlsx strictly.
3. Validate .npz array lengths equal N.
4. Validate metadata_json.n_draws == N.
5. Validate checkpoint_date ≤ decision_date (no leakage).
6. Fail fast if any required parameter missing.

---

# FINAL RESULT

With these schemas:

* No hidden assumptions
* No magic constants
* Fully versioned checkpoints
* Deterministic reproducibility
* Easy peer review
* Easy extension to multi-stakeholder later

---

