# Game Tree Timing Structure — Qantas ARA V2

Sequential structure of the Qantas governance crisis game tree.
Branching is **conditional** on game state (CEO presence, review status, review findings).

---

## Pre-game: D0_ceo (CEO pre-emptive resignation, 05-Sep-2023)

CEO decides whether to resign before the AGM sequence begins.

| Action | State effect |
|--------|-------------|
| CEO_resign | CEO_present ← false, CEO_removed ← true, CEO_resigned_early ← true |
| CEO_stay | No change (CEO_present = true) |

Modelled as a strategic decision. Solver evaluates both scenarios and computes Pr(CEO_resign) via ARA.

---

## Phase 1: Pre-AGM decisions

**D1 — Board governance reform** (Board)

| Action | Feasibility | State effect |
|--------|------------|--------------|
| D0_minimal | always | none |
| D1_review | always | review_commissioned ← true |
| D3_ceo_transition | CEO_present | CEO_present ← false, CEO_removed ← true |

**A2 — ASA strike recommendation** (ASA)

| Action | Feasibility |
|--------|------------|
| A2_no_strike | always |
| A2_rec_strike | always |

## Phase 2: AGM

**V — Shareholder vote** (Nature, chance)
Vote percentage V ∈ [0,1] via logit-normal. Derived: strike = 1[V ≥ 0.25], overwhelming = 1[V ≥ 0.50].

**M_agm — Post-AGM market reaction** (Nature, pass-through)

## Phase 3: Post-AGM CEO response

**D4 — CEO responds to AGM outcome** (CEO)
Condition: CEO_present = true. **Skipped if CEO already removed.**

| Action | State effect |
|--------|-------------|
| D4_stay | none |
| D4_resign | CEO_present ← false, CEO_removed ← true |
| D4_negotiate_exit | CEO_present ← false, CEO_removed ← true |

## Phase 4: Post-AGM Board response

**D_rev — Board response** (Board)
Action availability depends on CEO_present and review_commissioned.

| Action | Feasibility | State effect |
|--------|------------|--------------|
| Drev_no_action | always | none |
| Drev_commission_review | review_not_commissioned | review_commissioned ← true |
| Drev_sack_ceo | CEO_present | CEO_present ← false, CEO_removed ← true |

## Phase 5: Review findings

**R — Governance review findings** (Nature, chance)
- If review_commissioned: CAR ~ t_3(μ_f, σ_f), review_adverse = 1[CAR < 0]
- If not commissioned: deterministic (CAR = 0, adverse = false)

**M_rev — Post-review market reaction** (Nature, pass-through)

## Phase 6: Post-review response (conditional)

**Condition:** review_adverse = true AND CEO_present = true

If this condition holds, an additional round of decisions occurs:

**D4 — CEO responds to adverse review** (CEO)
Same action set and feasibility as Phase 3. CEO may resign in light of adverse findings.

**D_rev — Board responds to adverse review** (Board)
If CEO still present after D4 above, Board may sack CEO. Otherwise limited to no action / commission review.

If the condition does not hold (review not adverse, or CEO not present): skip directly to Terminal.

## Terminal

Compute utilities for all actors based on full game history and final state.

---

## Full tree: CEO stays, D1 = do nothing or commission review

```
D0_ceo: CEO_stay
  └─ D1 (Board: D0_minimal or D1_review)
      └─ A2 (ASA: strike rec)
          └─ V (Vote)
              └─ M_agm
                  └─ D4 (CEO: stay/resign/negotiate)
                      │
                      ├─ CEO resigns or negotiates exit
                      │   └─ D_rev (Board: no action / commission review)
                      │       └─ R → M_rev → Terminal
                      │
                      └─ CEO stays
                          └─ D_rev (Board: no action / commission review / sack CEO)
                              │
                              ├─ Board sacks CEO
                              │   └─ R → M_rev → Terminal
                              │
                              └─ Board no action or commission review
                                  └─ R (review findings)
                                      └─ M_rev
                                          │
                                          ├─ Not adverse → Terminal
                                          │
                                          └─ Adverse AND CEO present
                                              └─ D4 (CEO: stay/resign)
                                                  │
                                                  ├─ CEO resigns → Terminal
                                                  │
                                                  └─ CEO stays
                                                      └─ D_rev (Board: sack/no action)
                                                          └─ Terminal
```

## Simplified tree: CEO resigned or forced out early

```
D0_ceo: CEO_resign  (or  D1: D3_ceo_transition)
  └─ D1 (Board: D0_minimal or D1_review)
      └─ A2 (ASA: strike rec)
          └─ V (Vote)
              └─ M_agm
                  └─ [D4 skipped — CEO not present]
                      └─ D_rev (Board: no action / commission review)
                          └─ R → M_rev → Terminal
```

---

## Key design principles

1. **D4 comes BEFORE D_rev**: CEO reacts to AGM results first, then Board responds. The CEO has initiative to resign before the Board decides whether to act.

2. **Conditional post-review round**: After adverse review findings, if the CEO is still present, another round of CEO → Board decisions occurs. This captures the governance escalation dynamic.

3. **Feasibility-driven pruning**: When CEO is removed (by any mechanism — early resignation, D3, D4 resignation, Board sack), all subsequent CEO decision nodes become infeasible and are skipped.

4. **Actual 2023 path**: D0_ceo=CEO_resign → D1=D1_review → A2=A2_rec_strike → V=82.93% (strike, overwhelming) → M_agm → [D4 skipped] → D_rev=Drev_no_action → R=adverse → M_rev → Terminal
