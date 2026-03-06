# Game Tree Timing Structure — Qantas ARA V2

Sequential structure of the Qantas governance crisis game tree.
Branching is **conditional** on game state (CEO presence, review status, review findings, headline incident).

---

## Node order (12 nodes)

```
D0_ceo → D1 → A2 → V → M_agm → D4 → D_rev → R → M_rev → D4_post_review → D_rev_post_review → Terminal
```

| Index | Node | Type | Owner | Purpose |
|-------|------|------|-------|---------|
| 0 | D0_ceo | decision | CEO | Pre-game resignation choice |
| 1 | D1 | decision | Board | Governance reform package |
| 2 | A2 | decision | ASA | Strike recommendation |
| 3 | V | chance | Nature | Shareholder vote (logit-normal) |
| 4 | M_agm | chance | Nature | Post-AGM market reaction (pass-through) |
| 5 | D4 | decision | CEO | CEO response to AGM outcome |
| 6 | D_rev | decision | Board | Board response post-AGM |
| 7 | R | chance | Nature | Review findings (adverse rating + CAR) |
| 8 | M_rev | chance | Nature | Post-review market reaction (pass-through) |
| 9 | D4_post_review | decision | CEO | CEO response to adverse review |
| 10 | D_rev_post_review | decision | Board | Board response to adverse review |
| 11 | Terminal | terminal | Nature | Compute utilities for all actors |

---

## Pre-game: D0_ceo (CEO pre-emptive resignation, 05-Sep-2023)

CEO decides whether to resign before the AGM sequence begins.

| Action | Feasibility | State effect |
|--------|-------------|--------------|
| CEO_resign | always | CEO_present ← false, CEO_removed ← true, CEO_resigned_early ← true |
| CEO_stay | always | No change (CEO_present = true) |

Modelled as a strategic decision. Solver evaluates both scenarios and computes Pr(CEO_resign) via ARA using a Beta(12.5, 0.5) prior (Jeffreys + 12 Australian no-remorse CEO observations, mean ≈ 96.2%).

---

## Phase 1: Pre-AGM decisions

**D1 — Board governance reform** (Board)

| Action | Feasibility | State effect |
|--------|-------------|--------------|
| D0_minimal | always | none |
| D1_review | always | review_commissioned ← true |
| D3_ceo_transition | CEO_present | CEO_present ← false, CEO_removed ← true |

**A2 — ASA strike recommendation** (ASA)

| Action | Feasibility | State effect |
|--------|-------------|--------------|
| A2_no_strike | always | none |
| A2_rec_strike | always | none |

---

## Phase 2: AGM

**V — Shareholder vote** (Nature, chance)

Vote model V2 — logit-normal with endogenous governance effects:

```
B_agm = B_mkt[i]
      + gamma_A[i] × 1{A2=RecStrike}
      + gamma_AH[i] × 1{A2=RecStrike} × 1{headline=1}
      + gamma_D[i] × f(D1)

logit(V) ~ N(alpha_vote[i] + B_agm, sigma_vote[i])
```

Governance effects (epistemic — drawn once per belief draw, held fixed across MC vote samples):
- D0_minimal: f = 0 (baseline)
- D1_review: f ~ U(0, 1) — review reduces protest
- D3_ceo_transition: f ~ U(−1, 0.5) — ambiguous, asymmetric bounds

Crisis floor (when `headline_incident = true`):
```
V_floor ~ Beta(50, 150)     [mean 0.25, drawn once per belief draw]
V_final = max(V_logit_normal, V_floor)
```

Derived indicators: `strike = 1[V ≥ 0.25]`, `overwhelming = 1[V ≥ 0.50]`.

With overconfidence bias: governance effect bounds shift (Board overestimates effectiveness), and `sigma_vote` is scaled by `sigma_scale = 1/√κ` (Board underestimates vote uncertainty).

**M_agm — Post-AGM market reaction** (Nature, pass-through)

No stochastic outcome — propagates to next node.

---

## Phase 3: Post-AGM CEO response

**D4 — CEO responds to AGM outcome** (CEO)

Feasibility: all actions require `CEO_present = true`. **Entire node skipped if CEO already removed.**

| Action | Feasibility | State effect |
|--------|-------------|--------------|
| D4_stay | CEO_present | none |
| D4_resign | CEO_present | CEO_present ← false, CEO_removed ← true |
| D4_negotiate_exit | CEO_present | CEO_present ← false, CEO_removed ← true |

CEO utility distinguishes departure modes: `D_sacked = 100` (worst), `D_resign_late = 60` (mid-game), `D_negotiate = 45` (face-saving).

---

## Phase 4: Post-AGM Board response

**D_rev — Board response** (Board)

Action availability depends on CEO_present and review_commissioned.

| Action | Feasibility | State effect |
|--------|-------------|--------------|
| Drev_no_action | always | none |
| Drev_commission_review | review_not_commissioned | review_commissioned ← true |
| Drev_sack_ceo | CEO_present | CEO_present ← false, CEO_removed ← true |

Board fixed policy uses `sack_vote_threshold = 0.25` — Board sacks CEO after any first strike.

---

## Phase 5: Review findings

**R — Governance review findings** (Nature, chance)

Two-component model:

1. **Outcome rating** (adverse vs positive):
   ```
   p_adverse ~ Beta(10, 5)     [mean = 2/3, drawn once per belief draw]
   adverse ~ Bernoulli(p_adverse)
   ```
   With bias: `β_biased = 5 × (1 + 10 × review_car_bias)` — Board believes positive outcomes more likely.

2. **CAR** (market reaction to findings release):
   ```
   μ_f ~ Student-t(ν=4, −0.05, 0.03)
   σ_f ~ Half-Normal(0.10)
   CAR ~ Student-t(ν=3, μ_f, σ_f)
   ```
   With bias: `μ_f_biased = μ_f + review_car_bias` (Board perceives CAR ~3pp more favourable).

3. **Direct cost** (epistemic — drawn once per scenario):
   ```
   C_direct ~ Gamma(α=4.55, rate=4741)     [mean ≈ 9.6 bps]
   ```

If review not commissioned: deterministic pass-through (CAR = 0, adverse = false).

State transition on adverse outcome: if `review_adverse = true` AND `CEO_present = true`, then `post_review_round ← true`.

**M_rev — Post-review market reaction** (Nature, pass-through)

No stochastic outcome — propagates to next node.

---

## Phase 6: Post-review response (conditional)

**Condition:** `post_review_round = true` (review adverse AND CEO was present at review completion)

If this condition does **not** hold: both D4_post_review and D_rev_post_review are skipped (no feasible actions) → proceed directly to Terminal.

**D4_post_review — CEO responds to adverse review** (CEO)

| Action | Feasibility | State effect |
|--------|-------------|--------------|
| D4_stay | post_review_round | none |
| D4_resign | post_review_round | CEO_present ← false, CEO_removed ← true |
| D4_negotiate_exit | post_review_round | CEO_present ← false, CEO_removed ← true |

CEO reacts to adverse findings before Board can act.

**D_rev_post_review — Board responds to adverse review** (Board)

| Action | Feasibility | State effect |
|--------|-------------|--------------|
| Drev_no_action | post_review_round | none |
| Drev_sack_ceo | post_review_round AND CEO_present | CEO_present ← false, CEO_removed ← true |
| Drev_commission_review | post_review_round AND review_not_commissioned | review_commissioned ← true |

If CEO departed at D4_post_review, only Drev_no_action (and possibly Drev_commission_review) remain feasible.

---

## Terminal

Compute utilities for all actors based on full game history and final state.

- **Board:** minimises opposition & disruption; penalised by `ceo_loss_cost = 1.5` for CEO removal and `adverse_review_ceo_present_penalty = 5.0` when review adverse AND CEO still present
- **ASA:** maximises accountability; `mobilisation_cost = 0.3` for recommending strike
- **CEO:** CRRA wealth utility with loss aversion (`loss_aversion = 2.25`); departure-mode-specific penalties (`D_sacked = 100`, `D_resign_late = 60`, `D_negotiate = 45`)

---

## Full tree: CEO stays, D1 = do nothing or commission review

```
D0_ceo: CEO_stay
  └─ D1 (Board: D0_minimal or D1_review)
      └─ A2 (ASA: no_strike or rec_strike)
          └─ V (Vote: logit-normal + crisis floor)
              └─ M_agm (pass-through)
                  └─ D4 (CEO: stay / resign / negotiate_exit)
                      │
                      ├─ CEO departs (resign or negotiate)
                      │   └─ D_rev (Board: no_action / commission_review)
                      │       └─ R (review findings)
                      │           └─ M_rev (pass-through)
                      │               └─ [D4_post_review skipped — CEO absent]
                      │                   └─ [D_rev_post_review skipped — CEO absent]
                      │                       └─ Terminal
                      │
                      └─ CEO stays
                          └─ D_rev (Board: no_action / commission_review / sack_ceo)
                              │
                              ├─ Board sacks CEO
                              │   └─ R → M_rev → [post-review skipped] → Terminal
                              │
                              └─ Board no_action or commission_review
                                  └─ R (review findings)
                                      └─ M_rev (pass-through)
                                          │
                                          ├─ Not adverse (or CEO absent) → Terminal
                                          │
                                          └─ Adverse AND CEO present
                                              └─ D4_post_review (CEO: stay / resign / negotiate)
                                                  │
                                                  ├─ CEO departs
                                                  │   └─ D_rev_post_review (Board: no_action / commission_review)
                                                  │       └─ Terminal
                                                  │
                                                  └─ CEO stays
                                                      └─ D_rev_post_review (Board: sack / no_action / commission_review)
                                                          └─ Terminal
```

## Simplified tree: CEO resigned or forced out early

```
D0_ceo: CEO_resign  (or  D1: D3_ceo_transition)
  └─ D1 (Board: D0_minimal or D1_review)
      └─ A2 (ASA: no_strike or rec_strike)
          └─ V (Vote: logit-normal + crisis floor)
              └─ M_agm (pass-through)
                  └─ [D4 skipped — CEO not present]
                      └─ D_rev (Board: no_action / commission_review)
                          └─ R (review findings)
                              └─ M_rev (pass-through)
                                  └─ [D4_post_review skipped]
                                      └─ [D_rev_post_review skipped]
                                          └─ Terminal
```

---

## Key design principles

1. **D4 comes BEFORE D_rev**: CEO reacts to AGM results first, then Board responds. The CEO has initiative to resign before the Board decides whether to act.

2. **Distinct post-review nodes**: D4_post_review and D_rev_post_review are separate nodes (not reuse of D4/D_rev) with their own feasibility rules gated by `post_review_round`. This avoids ambiguity in the node order and simplifies recursion.

3. **Conditional post-review round**: After adverse review findings, if the CEO is still present, D4_post_review and D_rev_post_review activate. This captures the governance escalation dynamic where the Board can force out a CEO who survived the AGM.

4. **Feasibility-driven pruning**: When CEO is removed (by any mechanism — early resignation, D3, D4 departure, Board sack), all subsequent CEO decision nodes become infeasible and are skipped. The `post_review_round` flag further gates Phase 6 nodes.

5. **Epistemic vs aleatoric separation**: Governance effects, crisis floor, adverse probability, and review direct cost are drawn once per belief draw (epistemic uncertainty). Vote samples and review samples are drawn multiple times within each draw (aleatoric noise). This prevents double-counting uncertainty.

6. **Actual 2023 path**: D0_ceo=CEO_resign → D1=D1_review → A2=A2_rec_strike → V=82.93% (strike, overwhelming) → M_agm → [D4 skipped] → D_rev=Drev_no_action → R=adverse → M_rev → [post-review skipped] → Terminal
