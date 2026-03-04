# CEO Utility at D4: Post-AGM Decision

## Context

D4 is the CEO's decision node after the AGM vote (V) and market reaction (M_agm). At this point the CEO knows:
- The vote result (strike/no-strike, percentage)
- Whether the ASA recommended a strike
- What governance action the Board took at D1 (minimal/review/CEO transition)

The CEO chooses: **stay**, **resign**, or **negotiate exit**.

## Node Position

```
D0_ceo → D1 → A2 → V → M_agm → [D4] → D_rev → R → M_rev → ...
```

D4 comes BEFORE D_rev (Board's post-AGM response). The CEO has first-mover advantage: resigning or negotiating before the Board can sack prevents the worst non-monetary penalties.

## Utility Function

CEO utility follows reference-dependent CRRA with loss aversion:

```
U_total = U_money(W) − λ_D · D_raw
```

### Monetary Component (W)

| Departure mode | W formula | Typical value |
|:---------------|:----------|:-------------|
| Stay (not sacked) | W_stay_kept | 7.0 |
| Negotiate exit at D4 | (W_stay_sacked + W_stay_kept) / 2 | 3.75 |
| Resign at D4 | W_stay_sacked × 1.3 | 0.65 |
| Sacked by Board | W_stay_sacked | 0.50 |

W < W_ref triggers loss aversion:
```
U_money = λ · CRRA(W) − (λ−1) · CRRA(W_ref)
```

### Non-Monetary Component (D_raw)

D_raw is additive, starting from baseline crisis cost:

```
D_raw = D_stay (25)
      + D_departure_mode   (depends on how CEO leaves)
      + D_agm (30)         if vote > 25%
      + D_disgrace (30)    if overwhelming vote
      + D_adverse_review (10) if review adverse
```

**Departure-mode penalties** (the key calibration):

| Departure mode | Parameter | Value | Rationale |
|:---------------|:----------|:-----:|:----------|
| Negotiate exit | D_negotiate | 45 | CEO controls narrative, gets face-saving terms. Close to D_resign (40) but mid-crisis rather than pre-crisis. |
| Resign voluntarily | D_resign_late | 60 | CEO controls narrative but post-AGM — too late for "taking responsibility gracefully." |
| Sacked by Board | D_sacked | 100 | Maximum reputational destruction. Public firing, no control over narrative, career damage. |

Compare with D0 (pre-game resignation):

| | D0 resign | D4 resign | D4 negotiate | D4 sacked |
|:--|:---------:|:---------:|:------------:|:---------:|
| W | 8.0 | 0.65 | 3.75 | 0.50 |
| D_raw (base) | 40 | 25+60 = 85 | 25+45 = 70 | 25+100 = 125 |
| D_raw (+ D_agm) | — | 115 | 100 | 155 |

D0 resignation is unambiguously better: higher W, lower D, avoids AGM entirely.

## Strategic Dynamics at D4

The CEO's D4 choice depends on the **expected probability of being sacked at D_rev**:

- If Board unlikely to sack (no strike): **stay** dominates.
- If Board very likely to sack (strike): **negotiate** dominates (D_negotiate=45 vs D_sacked=100, W=3.75 vs W=0.50).

The CEO's belief about Board sacking is computed via ARA predictive distribution. In rollouts, the Board's fixed policy at D_rev uses `sack_vote_threshold = 0.25` (first strike threshold), so after any strike the Board is predicted to sack.

### Worked Example (strike at 33% vote)

| D4 action | D_raw | λ_D × D | W | U_money | U_total |
|:----------|:-----:|:-------:|:-:|:-------:|:-------:|
| Stay (not sacked) | 55 | 123.75 | 7.0 | −0.97 | −124.72 |
| Stay (sacked at D_rev) | 155 | 348.75 | 0.5 | −5.74 | −354.49 |
| Negotiate exit | 100 | 225.00 | 3.75 | −1.70 | −226.70 |
| Resign late | 115 | 258.75 | 0.65 | −4.96 | −263.71 |

If Pr(Board sacks at D_rev) = 0.7 after a strike:
```
E[stay] = 0.3 × (−124.72) + 0.7 × (−354.49) = −285.56
E[negotiate] = −226.70
```

Negotiate is 59 utility points better → CEO rationally negotiates.

## Calibration Sources

- **D_resign = 40 (D0)**: "Taking responsibility" framing, pre-crisis exit. Joyce resigned 5 Sep 2023 (before AGM on 3 Nov 2023).
- **D_negotiate = 45 (D4)**: Slightly worse than D0 but CEO retains some bargaining power. Comparable to negotiated departures (e.g., AMP CEO Francesco De Ferrari, 2021).
- **D_resign_late = 60 (D4)**: Post-AGM voluntary resignation. CEO controls narrative but damage already done by AGM. Comparable to Rio Tinto CEO Jean-Sébastien Jacques (2020) — resigned after Juukan Gorge but only after board pressure became overwhelming.
- **D_sacked = 100 (Board fires)**: Maximum stigma. Comparable to AMP CEO Craig Meller (2018, Royal Commission) or Crown directors removed after Bergin Inquiry.
- **Loss aversion λ_D = 2.25**: Tversky & Kahneman (1992). CEO evaluates all penalties relative to expected status as a powerful executive.
