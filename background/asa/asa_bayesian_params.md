# Bayesian Parameter Tables — ASA A2 Nodes (Qantas 2023)

## 1. Strike Probability Distributions — Beta(α, β)

Each A2 node's P(Recommend strike) is modelled as a Beta distribution. The α+β sum reflects evidential concentration: higher sums indicate less uncertainty. Parameters are derived from the utility dynamics analysis and the information state at each node.

| A2 Node | Path | α | β | Mean | Mode | SD | 90% CI |
|---------|------|---|---|------|------|-----|--------|
| A2-1 | CEO resigns → Do nothing | 18 | 2 | 0.900 | 0.944 | 0.065 | [0.79, 0.98] |
| A2-2 | CEO resigns → Commission review | 14 | 3 | 0.824 | 0.867 | 0.088 | [0.67, 0.95] |
| A2-3 | CEO stays → Do nothing | 24 | 1 | 0.960 | 0.979 | 0.038 | [0.90, 1.00] |
| A2-4 | CEO stays → Commission review | 15 | 2 | 0.882 | 0.917 | 0.078 | [0.74, 0.98] |
| A2-5 | CEO stays → Board forces exit | 9 | 6 | 0.600 | 0.615 | 0.130 | [0.38, 0.82] |

### Rationale for α+β concentrations

- **A2-3** has the highest concentration (25): the information state at this node leaves virtually no room for a no-strike outcome; every utility argument is at or near its floor.
- **A2-5** has the lowest concentration (15) and highest variance: this is the only scenario where the board has proactively imposed consequences, and ASA's decision involves genuine trade-off reasoning. The 90% CI spans 0.44 units, reflecting real model uncertainty.
- **A2-1 vs A2-2**: The review shifts the mean by ~0.076 and increases the SD slightly (0.065 → 0.088), because the review introduces genuine ambiguity about whether ASA will give new leadership benefit of the doubt.

---

## 2. Utility Score Distributions — Truncated Normal N(μ, σ) on [1, 5]

Scores are on a 1–5 Likert scale where 5 = strong positive utility for ASA and 1 = strong negative utility. The scale midpoint (3) represents a neutral or ambiguous signal. Distributions are truncated at [1, 5].

### A2-Node 1: CEO resigns → Do nothing

| Dimension | Weight | μ | σ | Reasoning |
|-----------|--------|---|---|-----------|
| FW | 0.10 | 2.1 | 0.5 | Share price recovering but ACCC legal exposure unresolved; retail investors face residual class action risk |
| PPL | 0.30 | 1.3 | 0.4 | A$21.4M confirmed against ACCC action and service failures; no clawback; dominant negative driver |
| TD | 0.10 | 1.8 | 0.5 | Remuneration report public but no conduct-linked STI gating disclosed; opacity on clawback |
| EGR | 0.15 | 1.5 | 0.5 | ACCC court action is regulatory validation of ESG risk; labour law violation on record |
| BA | 0.20 | 2.3 | 0.6 | CEO resignation is a partial positive; board otherwise unchanged and defensive in posture |
| OL | 0.10 | 2.0 | 0.5 | ASA member base directly affected as Qantas customers; visible inaction costly to member trust |
| PF | 0.05 | 3.0 | 0.4 | No specific AGM procedural violations yet observed; June share sale is a latent concern |

**Weighted mean utility: 1.84** | **Weighted utility SD: ~0.22**

---

### A2-Node 2: CEO resigns → Commission review

| Dimension | Weight | μ | σ | Reasoning |
|-----------|--------|---|---|-----------|
| FW | 0.10 | 2.2 | 0.5 | Marginally better than node 1; governance premium partially recovering |
| PPL | 0.30 | 1.3 | 0.4 | Remuneration report unchanged; review does not alter historical pay structure |
| TD | 0.10 | 2.2 | 0.5 | Review acknowledgment of governance failure is a partial transparency positive |
| EGR | 0.15 | 1.9 | 0.5 | Review signals ESG acknowledgment; ACCC case still live |
| BA | 0.20 | 2.9 | 0.6 | CEO gone + review + Mullen accountability statements = meaningful BA improvement |
| OL | 0.10 | 2.4 | 0.5 | ASA can be seen to have contributed to accountability outcomes |
| PF | 0.05 | 3.0 | 0.4 | No change from node 1 |

**Weighted mean utility: 2.09** | **Weighted utility SD: ~0.22**

---

### A2-Node 3: CEO stays → Do nothing

| Dimension | Weight | μ | σ | Reasoning |
|-----------|--------|---|---|-----------|
| FW | 0.10 | 1.7 | 0.4 | Maximum brand and legal risk unaddressed; long-run franchise damage to a consumer stock |
| PPL | 0.30 | 1.2 | 0.3 | Worst possible PPL state: confirmed high pay, confirmed misconduct, zero corrective action |
| TD | 0.10 | 1.5 | 0.4 | Board defensiveness amplifies disclosure concerns; share sale opacity still unaddressed |
| EGR | 0.15 | 1.3 | 0.4 | All ESG risk signals confirmed; board is validating rather than correcting the conduct record |
| BA | 0.20 | 1.2 | 0.3 | Zero accountability: CEO in post, pay received, board unchanged, no review |
| OL | 0.10 | 1.4 | 0.4 | ASA's retail shareholder constituency at maximum dissatisfaction; inaction would be reputationally fatal for ASA |
| PF | 0.05 | 2.9 | 0.4 | Minor: no AGM procedural issues yet, but share sale concern amplified |

**Weighted mean utility: 1.43** | **Weighted utility SD: ~0.18**

---

### A2-Node 4: CEO stays → Commission review

| Dimension | Weight | μ | σ | Reasoning |
|-----------|--------|---|---|-----------|
| FW | 0.10 | 1.9 | 0.5 | CEO in post preserves reputational uncertainty despite review announcement |
| PPL | 0.30 | 1.2 | 0.3 | As per node 3: remuneration report fixed; review is forward-looking and cannot alter it |
| TD | 0.10 | 2.0 | 0.5 | Review acknowledges governance gap but CEO's presence undermines its credibility |
| EGR | 0.15 | 1.6 | 0.5 | Review partially addresses EGR signal; ACCC still live |
| BA | 0.20 | 1.9 | 0.5 | Review is BA-positive but CEO remaining in post with full FY23 pay is a significant BA override |
| OL | 0.10 | 1.7 | 0.4 | Some OL credit for triggering a review; offset by CEO remaining |
| PF | 0.05 | 2.9 | 0.4 | Unchanged |

**Weighted mean utility: 1.67** | **Weighted utility SD: ~0.19**

---

### A2-Node 5: CEO stays initially → Board forces exit

| Dimension | Weight | μ | σ | Reasoning |
|-----------|--------|---|---|-----------|
| FW | 0.10 | 2.5 | 0.5 | Forced exit is a stronger governance signal than voluntary departure; franchise risk reduced |
| PPL | 0.30 | 1.6 | 0.5 | Historical FY23 pay already received; forced exit does not constitute clawback. Partial positive: signals PPL norms will be enforced in future |
| TD | 0.10 | 2.3 | 0.5 | Board action is itself a disclosure of accountability standards |
| EGR | 0.15 | 2.2 | 0.5 | Board proactively addressing the governance risk rather than defending it |
| BA | 0.20 | 3.3 | 0.6 | Board-imposed consequence is qualitatively different from managed exit; strongest BA signal in the scenario set |
| OL | 0.10 | 2.7 | 0.5 | ASA can credit itself with forcing a board-level accountability decision |
| PF | 0.05 | 3.0 | 0.4 | Unchanged; potential partial credit if clawback discussion proceeds |

**Weighted mean utility: 2.27** | **Weighted utility SD: ~0.22**

---

## 3. Summary Comparison Table

| A2 Node | Path | P(Strike) Mean | P(Strike) SD | Weighted Utility μ | Weighted Utility SD |
|---------|------|----------------|--------------|-------------------|---------------------|
| A2-1 | Resigns → Do nothing | 0.900 | 0.065 | 1.84 | 0.22 |
| A2-2 | Resigns → Review | 0.824 | 0.088 | 2.09 | 0.22 |
| A2-3 | Stays → Do nothing | 0.960 | 0.038 | 1.43 | 0.18 |
| A2-4 | Stays → Review | 0.882 | 0.078 | 1.67 | 0.19 |
| A2-5 | Stays → Forced exit | 0.600 | 0.130 | 2.27 | 0.22 |

---

## 4. Structural Notes for ARA Integration

**On the non-monotonicity of P(Strike) and utility:**
Note that A2-1 (resigns, do nothing) has *higher* P(Strike) and *lower* utility than A2-3 (stays, do nothing) is reversed: CEO staying produces *higher* P(Strike) but *lower* absolute utility. This is because OL and BA interact: when the CEO stays and does nothing, ASA's utility is lower, but the strike probability is also higher because the case for a strike is even clearer. The relationship between utility and strike probability is monotone within each branch (resign vs stay) but not across them.

**On the PPL argument as the binding constraint:**
The key modelling insight is that PPL (weight 0.30) is essentially fixed at ~1.3 across all nodes where the remuneration report is public and no clawback has been committed. Board actions (CEO departure, governance review) shift BA but leave PPL unchanged. PPL is the binding constraint; BA is the swing variable. This explains why P(Strike) does not fall below 0.60 even in the most accountability-positive scenario (A2-5).

**On using the prompt vs Beta parameters:**
- Use the **Beta parameters** for closed-form integration into the decision tree (e.g., Monte Carlo rollback using the Beta distributions as priors)
- Use the **prompt** to generate a posterior sample that can update those priors with LLM reasoning variation
- At T=1.0 with n=50 runs per node, the empirical P(Strike) from the prompt should converge to within ±0.05 of the Beta mean — use divergence as a calibration check
