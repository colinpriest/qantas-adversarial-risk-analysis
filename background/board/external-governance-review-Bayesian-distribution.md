# Bayesian Model: External Governance Review Outcome Ratings

This document specifies two conditional Dirichlet distributions for the ordinal outcome rating (Negative < Neutral < Positive) of external governance reviews of ASX 100 companies, one for each review classification:

- **P(Outcome | Regulatory)**: the distribution over outcomes when the review is regulatory-initiated
- **P(Outcome | Non-Regulatory)**: the distribution over outcomes when the review is board-initiated or otherwise non-regulatory

The sole monotonicity constraint is first-order stochastic dominance: the regulatory distribution must place weakly more mass on worse outcomes than the non-regulatory distribution at every cumulative threshold.

---

## 1. Data Extraction

### 1.1 Regulatory reviews (APRA 36 treated as a single event)

The APRA 36 Self-Assessment program (ANZ, NAB, MQG, SUN, IAG, QBE, MPL, BEN) was a single coordinated exercise initiated under one trigger (the CBA inquiry), conducted simultaneously under the same template. Treating it as 8 independent observations overstates the effective sample. Conservatively, it is counted as 1 event.

| Review event | Outcome |
|:-------------|:-------:|
| APRA Prudential Inquiry (CBA, 2017-2018) | Negative |
| APRA 36 Self-Assessment Program (2018-2019, pooled) | Negative |
| APRA/ASIC Compliance & Governance Review (AMP, 2019-2021) | Negative |
| Promontory Independent Review / CORE (WBC, 2020-2024) | Negative |
| Bergin Inquiry (CWN, 2020-2021) | Negative |
| Bell / Gotterson Inquiry (SGR, 2021-2022) | Negative |
| ASIC Technology Governance & CHESS Inquiries (ASX, 2018/2023) | Negative |
| Remediation EU (BOQ, 2023) | Negative |
| APRA Governance Remediation EU (ALL, 2021) | Negative |
| Compliance & Sales Practice Review (TLS, 2018-2023) | Negative |
| Underpayment Governance Review (WOW, 2020) | Negative |
| Regulatory Compliance Review (TAH, 2021-2022) | Negative |

**Regulatory total: 12 events, 12 Negative, 0 Neutral, 0 Positive.**

### 1.2 Non-regulatory reviews

| Review event | Outcome |
|:-------------|:-------:|
| Board Review of Cultural Heritage (RIO, 2020) | Negative |
| Governance Framework Gap Analysis (MIN, 2022) | Neutral |
| Independent Board Performance Review (BOQ, 2020) | Positive |

**Non-regulatory total: 3 events, 1 Negative, 1 Neutral, 1 Positive.**

### 1.3 Cross-tabulation

| Classification | Negative | Neutral | Positive | Total |
|:--------------:|:--------:|:-------:|:--------:|:-----:|
| Regulatory     | 12       | 0       | 0        | 12    |
| Non-Regulatory | 1        | 1       | 1        | 3     |
| **Total**      | **13**   | **1**   | **1**    | **15**|

---

## 2. Monotonicity Constraint

The constraint is first-order stochastic dominance: regulatory reviews produce weakly worse outcomes.

```
P(Y <= Negative | Regulatory) >= P(Y <= Negative | Non-Regulatory)
P(Y <= Neutral | Regulatory)  >= P(Y <= Neutral | Non-Regulatory)
```

This does not impose any ordering within the non-regulatory distribution. In particular, it does not require P(Neg | Non-Reg) > P(Neu | Non-Reg) > P(Pos | Non-Reg). The non-regulatory distribution is free to place more mass on Neutral than Negative, or more mass on Positive than Neutral, provided the cumulative thresholds remain below the corresponding regulatory values.

---

## 3. Domain Analysis

### 3.1 Why regulatory reviews are near-deterministically negative

The 12/12 negative rate reflects three structural mechanisms:

**Selection mechanism.** APRA, ASIC, and AUSTRAC initiate formal reviews in response to identified or strongly suspected failures. The APRA Prudential Inquiry into CBA was triggered by the AUSTRAC money-laundering case. The Bergin Inquiry into Crown followed media reporting of junket operator links to organised crime. The APRA 36 program was triggered by CBA's findings. In each case, the review was initiated because the regulator had reason to believe governance had failed. The review then documents the extent of a known or strongly suspected problem.

**Diagnostic framing.** Regulatory reviews are structured to identify deficiencies, not to certify adequacy. The APRA self-assessment template asked entities to benchmark against CBA's failures. The Bergin and Gotterson inquiries applied a "suitability" test presuming the burden of proof lay with the entity. This framing makes a negative finding the default outcome.

**Regulator incentive asymmetry.** Under the "Why not litigate?" posture, a regulator that initiates a review and finds nothing faces institutional embarrassment. False negatives (missed failures) carry career risk; false positives (finding problems later shown to be minor) do not. This pushes outcomes toward negative.

A non-negative regulatory outcome is structurally possible (a sector-wide screening could catch a well-governed entity, or pre-existing remediation could be sufficiently advanced) but rare. Estimated P(Negative | Regulatory) floor: 0.90.

### 3.2 Why non-regulatory reviews are heterogeneous

The non-regulatory category mixes fundamentally different review types:

- **Reactive** (RIO 2020): initiated after a crisis, resembling regulatory reviews in outcome. The Juukan Gorge review was technically non-regulatory but was conducted under intense parliamentary and public pressure.
- **Proactive** (BOQ 2020, MIN 2022): board-initiated before external crisis. These tend to identify specific improvement areas (Neutral) or confirm governance adequacy (Positive). A company with the self-awareness to commission a proactive review is, by selection, more likely to have at least partially functional governance.

With n=3 and one observation per outcome, the data cannot separate these subcategories. The distribution should reflect this genuine uncertainty without imposing an artificial within-category ordering.

### 3.3 Relevance to Qantas

The scope is publicly disclosed reviews, because public disclosure is the relevant condition for the Qantas game tree. Qantas announced the commissioning of its governance review, making it a public event that enters the information set of shareholders, the ASA, and the media. Undisclosed reviews are outside scope.

---

## 4. Recommended Distributions

Two distinct Dirichlet distributions are recommended, one per classification.

### 4.1 P(Outcome | Regulatory): Dirichlet(57, 2, 1)

| Outcome | Alpha | Mean | Mode | 90% Marginal CI |
|:-------:|:-----:|:----:|:----:|:---------------:|
| Negative | 57 | 0.950 | 0.966 | [0.906, 0.982] |
| Neutral | 2 | 0.033 | 0.017 | [0.003, 0.078] |
| Positive | 1 | 0.017 | 0.000 | [0.000, 0.050] |

**Effective sample size:** 60.

**Calibration logic:**

- **Mean 0.950.** The data gives 12/12 = 1.000. The prior pulls this to 0.95, allowing for the possibility that approximately 1 in 20 regulatory reviews produces a non-negative outcome. This allowance covers scenarios where a regulatory review is initiated prophylactically (e.g., sector-wide screenings) or where advanced pre-existing remediation leads the review to confirm adequacy. The 0.95 figure aligns with the floor estimate from Section 3.1.

- **P(Neutral) > P(Positive).** Among the rare non-negative regulatory outcomes, a mixed-findings neutral result is more plausible than an unqualified positive. Even when remediation is partially complete, the diagnostic framing of regulatory reviews makes unqualified endorsement unlikely. The Westpac CORE program, the closest case to a positive regulatory outcome, took 30 months and 12 quarterly reports to transition from negative to positive.

- **ESS of 60.** This exceeds the 12 actual observations because the domain logic (selection mechanism, diagnostic framing, regulator incentives) provides independent support for the high P(Negative). The ESS is not as high as 100 (used in the earlier analysis before the APRA 36 consolidation) because the reduced sample (12 vs 19) warrants modestly less confidence. The 60 figure is calibrated so that the 90% lower bound for P(Negative) sits just above 0.90.

- **90% CI [0.906, 0.982].** The lower bound of 0.906 means the model never assigns less than approximately 91% probability to negative outcomes for regulatory reviews, even in the most favourable draw. This is the minimum defensible concentration given the structural arguments.

### 4.2 P(Outcome | Non-Regulatory): Dirichlet(5, 5, 5)

| Outcome | Alpha | Mean | Mode | 90% Marginal CI |
|:-------:|:-----:|:----:|:----:|:---------------:|
| Negative | 5 | 0.333 | 0.308 | [0.116, 0.586] |
| Neutral | 5 | 0.333 | 0.308 | [0.116, 0.586] |
| Positive | 5 | 0.333 | 0.308 | [0.116, 0.586] |

**Effective sample size:** 15.

**Calibration logic:**

- **Symmetric Dirichlet.** The data is 1/1/1 across three categories. No within-category ordering is imposed by the constraint, and the data provides no evidence for one. A symmetric prior honestly represents the state of knowledge: all three outcomes are roughly equally likely for non-regulatory reviews, with wide uncertainty.

- **Concentration parameter alpha = 5 per category.** This is higher than the uninformative Dirichlet(1, 1, 1), which would give an ESS of 3 and credible intervals spanning nearly the entire [0, 1] range. The alpha of 5 reflects a modest domain belief that non-regulatory outcomes are genuinely heterogeneous (not dominated by any single category) and prevents extreme draws (e.g., P(Negative) > 0.90 or P(Positive) > 0.90) from dominating the simulation. The 90% marginal CI of [0.116, 0.586] is wide but bounded: no single outcome can absorb more than approximately 60% or less than approximately 12% of probability in the central 90% of draws.

- **ESS of 15.** Moderate: large enough to prevent a single new observation from swinging the distribution dramatically, small enough to be updated meaningfully by a batch of 5-10 new non-regulatory observations. With only 3 actual observations, the prior carries most of the weight, which is appropriate when the data cannot distinguish reactive from proactive subtypes.

- **Why not tilt toward negative?** The overall base rate is 13/15 = 0.867 negative, which might suggest tilting the non-regulatory prior. However, this base rate is dominated by regulatory reviews (12 of 15). The non-regulatory data is 1/1/1, and the domain argument is that proactive non-regulatory reviews (the majority subtype, 2 of 3 in the data) tend to produce neutral or positive outcomes. Tilting toward negative on the basis of the overall rate would conflate the two populations, which is precisely what the conditional model is designed to avoid.

### 4.3 Stochastic ordering verification

At the posterior means:

| Cumulative threshold | Regulatory | Non-Regulatory | Gap | Satisfied |
|:--------------------:|:----------:|:--------------:|:---:|:---------:|
| P(Y <= Negative) | 0.950 | 0.333 | 0.617 | Yes |
| P(Y <= Neutral) | 0.983 | 0.667 | 0.317 | Yes |

The gaps are large. Because the two Dirichlet distributions are specified independently, individual draws can occasionally violate the ordering. With mean gaps of 0.617 and 0.317 and the tight regulatory concentration, violations will occur in fewer than 1% of paired draws. A rejection sampler (draw from both, discard pairs that violate ordering) is computationally trivial and guarantees pointwise compliance.

---

## 5. Cumulative Odds Representation

For integration with the logit-scale game tree.

### 5.1 Cumulative probabilities and logits at the mean

**Regulatory:**

| Threshold | Cumulative P | Cumulative logit |
|:---------:|:------------:|:----------------:|
| Y <= Negative | 0.950 | 2.944 |
| Y <= Neutral | 0.983 | 4.068 |

**Non-Regulatory:**

| Threshold | Cumulative P | Cumulative logit |
|:---------:|:------------:|:----------------:|
| Y <= Negative | 0.333 | -0.693 |
| Y <= Neutral | 0.667 | 0.693 |

### 5.2 Proportional odds form

```
logit(P(Y <= j | x)) = alpha_j - beta * x
```

where x = 1 for Non-Regulatory, x = 0 for Regulatory:

- Threshold 1: beta_1 = 2.944 - (-0.693) = 3.637
- Threshold 2: beta_2 = 4.068 - 0.693 = 3.375

Average beta approximately 3.51. The mild departure from proportional odds (beta_1 = 3.64 vs beta_2 = 3.38) indicates the classification effect is slightly stronger at the Negative/Neutral boundary than at the Neutral/Positive boundary. For practical purposes, a single beta of 3.5 is adequate.

**Prior for beta (if proportional odds is required):**

```
beta ~ N(3.5, 0.9)    truncated below at 0
```

The truncation enforces monotonicity. The 0.9 SD spans beta from approximately 2.0 (moderate effect) to 5.0 (near-deterministic effect) at the 90% level.

---

## 6. Sensitivity Analysis

### 6.1 Regulatory distribution sensitivity

| Specification | Dirichlet | P(Neg) | 90% CI for P(Neg) | Interpretation |
|:--------------|:---------:|:------:|:------------------:|:---------------|
| Base case | (57, 2, 1) | 0.950 | [0.906, 0.982] | Domain-informed, ESS 60 |
| Data only | (13, 1, 1) | 0.867 | [0.717, 0.971] | Dirichlet(1,1,1) prior + 12 observations |
| Higher confidence | (95, 3, 2) | 0.950 | [0.914, 0.977] | ESS 100, narrower interval |
| Lower floor | (46, 3, 1) | 0.920 | [0.862, 0.964] | Allows 8% non-negative rate |

The base case represents a middle ground: higher confidence than the data alone justifies (because domain logic provides independent support), but not as high as the pre-consolidation ESS of 100.

### 6.2 Non-regulatory distribution sensitivity

| Specification | Dirichlet | P(Neg), P(Neu), P(Pos) | 90% CI for each | Interpretation |
|:--------------|:---------:|:-----------------------:|:----------------:|:---------------|
| Base case | (5, 5, 5) | 0.333, 0.333, 0.333 | [0.116, 0.586] | Symmetric, ESS 15 |
| Tilted negative | (7, 5, 3) | 0.467, 0.333, 0.200 | Varies | Reflects base rate pull |
| More diffuse | (3, 3, 3) | 0.333, 0.333, 0.333 | [0.058, 0.672] | ESS 9, wider intervals |
| More concentrated | (8, 8, 8) | 0.333, 0.333, 0.333 | [0.157, 0.530] | ESS 24, tighter intervals |

The symmetric specification is robust to concentration scaling (the mean does not change), and a negative tilt is available if future data or argument supports it.

### 6.3 Stochastic ordering robustness

The minimum gap at either cumulative threshold across all sensitivity combinations:

- Regulatory "lower floor" (P(Neg) = 0.920) vs Non-Reg "tilted negative" (P(Neg) = 0.467): gap = 0.453. Still large.
- Regulatory "data only" (P(Neg) = 0.867) vs Non-Reg base case (P(Neg) = 0.333): gap = 0.534.

The ordering is robust across all plausible specifications. A rejection sampler would discard fewer than 2% of draws in the worst case.

---

## 7. Limitations

### 7.1 Sample size

The total sample is 15 events (after APRA 36 consolidation), with only 3 non-regulatory. The regulatory distribution is well-identified by the combination of data and domain logic. The non-regulatory distribution is dominated by the prior. Any downstream analysis should note that the non-regulatory probabilities are assumed, not estimated.

### 7.2 Regime change

The report documents a shift from "comply or explain" to "active supervision" around 2018. Pre-2018 regulatory reviews may have had different outcome distributions. The recommended distributions are calibrated to the current regime and should not be applied to a hypothetical future relaxation of regulatory intensity.

### 7.3 Remediation trajectories

The distributions capture the initial review outcome. The Westpac case demonstrates that outcomes can transition from Negative to Positive through sustained remediation. If the downstream model requires remediation dynamics, this initial-outcome model should be coupled with a separate transition specification.

---

## 8. Summary

### 8.1 Recommended distributions

| Classification | Distribution | P(Neg) | P(Neu) | P(Pos) | ESS |
|:--------------:|:------------:|:------:|:------:|:------:|:---:|
| **Regulatory** | **Dirichlet(57, 2, 1)** | **0.950** | **0.033** | **0.017** | **60** |
| **Non-Regulatory** | **Dirichlet(5, 5, 5)** | **0.333** | **0.333** | **0.333** | **15** |

### 8.2 Key properties

- Two distinct distributions, one per classification, providing different outcome probabilities for regulatory versus non-regulatory reviews.
- Stochastic ordering satisfied at mean level with large gaps (0.617 and 0.317). Pointwise violations under independent draws are rare (< 1%) and handled by rejection sampling.
- The regulatory distribution is tightly concentrated on negative outcomes, reflecting 12/12 data and strong structural arguments about regulatory selection, diagnostic framing, and incentive asymmetry.
- The non-regulatory distribution is symmetric, reflecting 1/1/1 data and the absence of evidence for any within-category ordering.
- No ordering is imposed within the non-regulatory category. The constraint applies only between categories.
