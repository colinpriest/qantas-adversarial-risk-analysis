# Data in `ranked_voting_recommendations.csv`

The `voting_diff` column represents the **year-on-year change in shareholder dissatisfaction**.

It measures the difference between the percentage of votes cast *"Against"* a resolution (typically the Remuneration Report) in the current year compared to the previous year.

---

## The Formula

$$
\text{voting\_diff} = \text{rem\_against\_pct} - \text{prior\_year\_pct}
$$

---

## How to Interpret the Numbers

- **Positive Number (e.g., +0.30):**
  An **escalation in protest** — the "Against" vote increased by 30 percentage points year-on-year. Typical when a Headline Incident occurs or when shareholders feel previous concerns were not addressed.
- **Negative Number (e.g., -0.15):**
  A **cooling off** — the "Against" vote dropped by 15 percentage points, suggesting board actions or improved performance regained shareholder trust.
- **Zero:**
  The level of dissent was unchanged.

---

## Board Action Effects on Voting Dissent

`voting_diff` tests whether **Board Corrective Actions** work:

| Board action | Avg `voting_diff` | Interpretation |
|---|---|---|
| D0 — No action | +32.8% | Baseline escalation when board is passive |
| D1 — Review/announcement | +14.9% | Most effective mitigation — dampens protest growth |
| D3 — CEO exit | +46.7% | Signals crisis severity; sacking itself amplifies protest |

The non-monotonic pattern (D1 mitigates, D3 escalates) is embedded in the `VoteModel._governance_effect()` method in `engine/chance_models.py`.

---

## Bayesian Prior for ASA Strike Recommendation (A2 node)

The data informs a **Beta-Binomial prior** on the probability that ASA recommends a strike vote, conditioned on Board's governance action (D1 node) and restricted to moral-reputational headline incidents.

### Filtering Rules

Two filters are applied before counting:

1. **`headline_incident = 1` only** — moral-reputational crises (environmental destruction, regulatory breaches, wage theft, management misconduct). Pure pay-quantum disputes have materially lower strike rates and are not representative of the Qantas scenario.
2. **Qantas (QAN, 2023) excluded** — the subject of the model cannot be used to inform its own prior. Circular inclusion would bias ASA's predicted strike probability toward the known outcome.

### Eligible Observations (n=14)

After both filters:

| Company | Year | Board action | `asa_against` | Note |
|---|---|---|---|---|
| Rio Tinto | 2021 | D3 (CEO exit) | 1 | Juukan Gorge cultural destruction |
| AGL Energy | 2021 | D1 (review) | 1 | Climate/emissions governance |
| Westpac | 2021 | D1 (review) | 1 | AUSTRAC money-laundering scandal |
| IAG | 2021 | D1 (review) | 1 | Insurance pricing misconduct |
| Downer EDI | 2022 | D0 (no action) | 1 | |
| ASX Ltd | 2022 | D3 (CEO exit) | 1 | Technology governance failure |
| Harvey Norman | 2023 | D0 (no action) | 1 | Wage theft scandal |
| Fortescue | 2023 | D0 (no action) | 0 | Headline incident; ASA did not recommend against |
| Woolworths | 2023 | D0 (no action) | 1 | Cost-of-living pricing scandal |
| Elders | 2023 | D0 (no action) | 1 | |
| Mineral Resources | 2023 | D0 (no action) | 1 | |
| Perpetual | 2023 | D0 (no action) | 1 | |
| Atlas Arteria | 2023 | D0 (no action) | 1 | |
| Macquarie Group | 2023 | D0 (no action) | 1 | |

**Qantas 2023 (board_action=2, asa_against=1) is excluded from all counts.**

### Count Summary by Board Action

| Board action | n | k (asa_against=1) | k/n |
|---|---|---|---|
| D0 — no action | 9 | 8 | 88.9% |
| D1 — review | 3 | 3 | 100.0% |
| D3 — CEO exit | 2 | 2 | 100.0% |

Fortescue is the only headline-incident case in which ASA did not recommend against (D0 bucket); all other cases are unanimous.

### Prior Specification

A **Jeffreys prior Beta(0.5, 0.5)** is used — the standard non-informative prior for a binomial proportion, consistent with the D0_ceo prior derivation. It is symmetric around 0.5 and places equal prior weight on "never recommends" and "always recommends."

$$
p(\theta) \propto \theta^{-0.5}(1-\theta)^{-0.5}, \quad \theta \in (0,1)
$$

### Posterior Derivation

With k successes in n trials, the posterior is:

$$
\theta \mid \text{data} \sim \text{Beta}(k + 0.5,\ n - k + 0.5)
$$

| Board action | Posterior | Mean | 95% credible interval |
|---|---|---|---|
| D0 — no action | Beta(8.5, 1.5) | 0.850 | [0.59, 0.98] |
| D1 — review | Beta(3.5, 0.5) | 0.875 | [0.53, 1.00] |
| D3 — CEO exit | Beta(2.5, 0.5) | 0.833 | [0.44, 1.00] |

All posterior means exceed 0.5. The **MAP estimate always recommends strike** regardless of Board action — consistent with the 100% first-strike rate observed in all headline-incident cases.

### Implementation

In `engine/predictive.py`, `_level1_policy()` for `actor="ASA"`, `node_name="A2"`:

```python
# Two-stage Beta-Binomial: sample p_strike from posterior, then Bernoulli
if d1_action == "D3_ceo_transition":
    p_strike = rng.beta(2.5, 0.5)   # Beta(k+0.5, n-k+0.5) with n=2, k=2
elif d1_action == "D1_review":
    p_strike = rng.beta(3.5, 0.5)   # n=3, k=3
else:
    p_strike = rng.beta(8.5, 1.5)   # n=9, k=8 (Fortescue abstained)
```

---

## Market Alignment as ASA Utility

The data reveals a further structural fact: **in 100% of headline-incident cases, the market votes a first strike** (`first_strike = 1` for all 14 eligible observations). This is not captured by the prior alone — it is a utility consideration.

ASA's institutional credibility and future influence depend on being aligned with mainstream institutional investor behaviour. When ASA recommends a strike and the market validates that recommendation, ASA's standing as the leading governance advocacy body is enhanced. When ASA stays silent while the market acts without it, credibility erodes.

This is modelled in `engine/utilities.py` as `market_alignment_bonus`:

```python
if outcome.a2_action == "A2_rec_strike" and outcome.strike_indicator:
    u += params.get("market_alignment_bonus", 1.5)
```

The parameter is stored in `governance_spec.xlsx` sheet `utilities_asa` (`market_alignment_bonus = 1.5`).
