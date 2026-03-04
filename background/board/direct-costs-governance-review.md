# Direct Costs of Commissioning an External Board Governance Review

## Estimation Framework Scaled to Cumulative Abnormal Returns

**Working Paper — Draft for Discussion**

Colin [Surname], School of Risk and Actuarial Studies, UNSW Business School

---

## 1. Purpose and Scope

This report estimates the direct, observable costs incurred by a board of an ASX-listed company when it commissions an external governance review with a published report. It covers three cost components: independent reviewer fees, management distraction, and internal resource consumption. These are the costs the board can reasonably anticipate at the point of decision.

All estimates are expressed as decimal CAR values, where a CAR of -0.0001 represents a 1 basis point reduction in cumulative abnormal returns. The reference entity is an ASX-listed company with a market capitalisation of approximately AUD 10 billion, consistent with Qantas Airways Limited during the period October 2023 to August 2024.

Indirect and information-related costs — litigation exposure from published findings, regulatory invitation effects, competitive intelligence leakage, and signalling costs — are excluded from this report. These are addressed separately and are substantially larger in magnitude but considerably harder to estimate with precision.

---

## 2. Component A: Independent Reviewer Fees

### 2.1 Description

This component covers the fees paid directly to the external reviewer and any supporting consultants, legal advisers to the review, and administrative costs of the review process (document management, transcription, travel, report production).

### 2.2 Estimation Methodology

Board-initiated governance reviews in the Australian panel are typically structured as senior advisory engagements led by one or two prominent individuals with supporting staff. The fee structure depends on the reviewer's profile, the scope of the terms of reference, the number of interviewees, and the duration of the engagement.

**Observable data points from the reference panel:**

Crown Resorts disclosed a contribution of AUD 12.5 million towards the costs of the Bergin Inquiry, but this was a regulator-initiated statutory inquiry running 60 days of public hearings over 18 months with full legal teams — not comparable to a board-initiated review.

The Qantas review was conducted by Tom Saar, an experienced McKinsey business adviser, over approximately 10 months (October 2023 to August 2024), producing 32 recommendations. The review involved interviews with board members, the CEO, the Group Executive team, and key functional leaders, together with document review and benchmarking. No public disclosure of fees has been identified.

The Westpac Advisory Panel comprised three eminent individuals (Switkowski, Schott, Carter) operating over approximately six months. Again, fees were not separately disclosed.

The Rio Tinto review was led by a single NED (L'Estrange) with internal support, which would have reduced external fees but increased internal resource costs.

**Estimation by analogy:** A senior independent advisory engagement of this nature — one lead adviser, a small supporting team, 10 months duration, 30+ interviews, a published report — would typically command fees in the range of AUD 2 million to AUD 5 million. This range accounts for the lead adviser's personal fees (AUD 500K–1.5M for a 10-month part-time engagement at this seniority), supporting consultant time (2–4 FTEs at professional services rates for portions of the engagement), independent legal advice to the review (AUD 200K–500K), and production and administrative costs (AUD 100K–300K).

The lower bound (AUD 2M) assumes a lean engagement with a single lead adviser and minimal supporting staff, comparable to the Rio Tinto structure. The upper bound (AUD 5M) assumes a more substantial team with dedicated legal support, comparable to the Westpac Advisory Panel structure. A central estimate of AUD 3 million is adopted.

### 2.3 Conversion to Decimal CAR

For a company with market capitalisation M = AUD 10 billion:

$$CAR_{fees} = -\frac{\text{Fee cost}}{M}$$


| Scenario | Fee estimate (AUD) | Decimal CAR |
| -------- | ------------------ | ----------- |
| Low      | 2,000,000          | -0.00020    |
| Central  | 3,000,000          | -0.00030    |
| High     | 5,000,000          | -0.00050    |


### 2.4 Confidence Assessment

**High.** Although the exact fees are not publicly disclosed for board-initiated reviews, the range of plausible values is narrow and the conversion to CAR produces values that are economically immaterial. The estimate is insensitive to reasonable variation in assumptions. Even doubling the upper bound to AUD 10 million produces a CAR of only -0.00100, which remains small relative to the other cost components and trivial relative to the utility factors in the benefit side of the model.

---

## 3. Component B: Management Distraction

### 3.1 Description

This component captures the opportunity cost of senior executive and board member time diverted from operational and strategic activities to participate in the review process. It includes time spent in interviews, preparing documents for the reviewer, reviewing drafts, deliberating on findings and recommendations, and planning and overseeing remediation.

Management distraction is the largest of the three direct cost components. Unlike reviewer fees, it is not directly observable in financial statements and must be estimated indirectly.

### 3.2 Estimation Methodology

Three independent estimation approaches are developed below. The final estimate is a weighted combination.

#### Approach B1: Executive Time Costing (Bottom-Up)

This approach estimates the hours consumed by the review at each organisational level and applies a fully-loaded hourly cost.

**Board members.** For a board of 8–10 members, each participating in a 2–3 hour interview, reviewing the draft report (4–8 hours), and participating in 3–5 dedicated board sessions on the review (12–20 hours total), the per-director time commitment is approximately 20–30 hours. For the Chair and any board committee chairs with additional involvement (meetings with the reviewer, stakeholder communications), add 30–50 hours. Total board time: approximately 250–400 hours.

Non-executive director fees for ASX 50 companies are typically AUD 250K–350K per annum for base fees, with the Chair receiving AUD 600K–800K. The effective hourly rate (assuming approximately 400–600 hours of committed time per year) is AUD 500–700 per hour.

Board time cost: 300 hours × AUD 600/hour ≈ AUD 180,000.

**CEO and Group Executives.** The CEO is typically the most heavily engaged executive, participating in multiple interviews, reviewing findings, leading the management response, and overseeing remediation planning. Estimated time commitment: 80–150 hours over the review period. Each Group Executive (typically 8–12 in an ASX 50 company) would participate in at least one interview (2–3 hours), prepare briefing materials (4–8 hours), and contribute to remediation planning (8–16 hours). Estimated time per Group Executive: 15–25 hours. Total Group Executive time: approximately 200–400 hours.

Total compensation for a Group Executive at Qantas is approximately AUD 2–4 million per annum (including fixed pay, short-term incentive at target, and long-term incentive at fair value). The CEO's total compensation is higher. Using a blended average of AUD 3 million and approximately 2,500 productive hours per year, the fully-loaded hourly rate is AUD 1,200.

C-suite time cost: 350 hours × AUD 1,200/hour ≈ AUD 420,000.

**Total direct time cost (B1):** AUD 180,000 + AUD 420,000 ≈ AUD 600,000.

This produces a CAR of -0.00006, which is clearly too low because it captures only the direct interview and deliberation time, not the cognitive bandwidth and strategic distraction that occurs throughout the review period.

#### Approach B2: Cognitive Bandwidth Multiplier

The executive time costing approach measures only the hours directly attributable to the review. It does not capture the broader productivity drag that occurs when senior leaders are operating under the cognitive load of a governance review — anticipating findings, managing internal politics, hedging against adverse outcomes, and modulating their behaviour in ways that may reduce decision-making quality.

Research on executive distraction during M&A processes, proxy contests, and regulatory investigations suggests that the total productivity impact is typically 3–5 times the direct time cost. This multiplier reflects reduced quality of non-review decisions, delayed strategic initiatives, risk aversion in operational decision-making during the review period, and internal political dynamics (executives positioning for the review's conclusions).

Applying a multiplier of 4× to the direct time cost:

**Adjusted distraction cost (B2):** AUD 600,000 × 4 = AUD 2,400,000.

CAR = -0.00024.

#### Approach B3: Operational Performance Deviation

This approach estimates the distraction cost indirectly by examining whether operational performance deteriorated during the review period relative to a pre-review baseline and sector benchmark. If the company's operational KPIs (on-time performance, customer satisfaction, load factors, complaints per passenger, employee engagement) declined during the review period by more than can be explained by external factors, the residual may be partially attributable to management distraction.

This approach has a fundamental identification problem: the same underlying governance failures that triggered the review are also causing the operational underperformance. Disentangling the effect of the review process from the effect of the underlying problems requires an instrument or a natural experiment, neither of which is readily available.

However, the approach is useful as an upper bound. If operational underperformance during the review period was equivalent to, say, a 0.5% revenue shortfall relative to the counterfactual (a plausible magnitude given the Qantas context), and operating margins were approximately 15%, this translates to a profit impact of approximately AUD 13.5 million (= AUD 18B revenue × 0.005 × 0.15), or a CAR of -0.00135. But only a fraction of this is attributable to the review process rather than the underlying problems. If 10–30% is attributable to the review distraction, this gives a CAR of -0.00014 to -0.00041.

This is broadly consistent with the cognitive bandwidth estimate (B2).

#### Approach B4: Analogy from M&A and Proxy Fight Distraction Literature

The corporate finance literature estimates the management distraction cost of contested M&A transactions at approximately 20–50 bps of firm value (CAR of -0.0020 to -0.0050) over the contest period. However, these estimates include the distraction of the full organisation including operational staff, not just the C-suite, and they apply to processes that are considerably more consuming than a governance review (due diligence, negotiation, regulatory approvals, integration planning).

A governance review is less operationally disruptive than an M&A process but more strategically sensitive (because it directly evaluates the board and CEO). A reasonable scaling factor is 10–30% of the M&A distraction estimate, giving a CAR of -0.0002 to -0.0015.

### 3.3 Synthesis


| Approach                  | CAR estimate         | Weight            |
| ------------------------- | -------------------- | ----------------- |
| B1: Direct time costing   | -0.00006             | 0.10 (floor only) |
| B2: Cognitive bandwidth   | -0.00024             | 0.35              |
| B3: Operational deviation | -0.00014 to -0.00041 | 0.25              |
| B4: M&A analogy           | -0.00020 to -0.00150 | 0.30              |


The weighted central estimate is approximately **-0.00040**, with a plausible range of -0.00015 to -0.00100.

### 3.4 Confidence Assessment

**Medium.** The direct time costing component is well-identified but captures only a fraction of the true cost. The cognitive bandwidth multiplier and M&A analogy approaches rest on transferability assumptions that are difficult to validate. The operational deviation approach has an endogeneity problem. The range is wide (roughly a factor of 7 between the low and high estimates), reflecting genuine uncertainty about the magnitude of the distraction effect. The central estimate of -0.00040 is the author's informed judgment.

---

## 4. Component C: Internal Resource Consumption

### 4.1 Description

This component covers the time of internal staff below the Group Executive level who are drawn into the review process. The primary functions affected are the company secretariat and governance team, the legal department (in-house counsel), investor relations and external communications, human resources (providing culture, engagement, and people data), risk and compliance, and internal audit.

These teams prepare documents for the reviewer, coordinate interview schedules, compile data on the matters under review, review draft findings for factual accuracy, prepare the public release and communications strategy, and subsequently manage the implementation of recommendations.

### 4.2 Estimation Methodology

The estimation follows a similar structure to the executive time costing but applied to functional teams.

**Company Secretariat / Governance (3–5 people, 30–50% of time, 10 months):** Central estimate: 4 FTEs × 0.40 × (10/12) year × AUD 200,000 average comp = AUD 267,000.

**Legal (2–4 people, 20–40% of time, 10 months):** Central estimate: 3 FTEs × 0.30 × (10/12) year × AUD 250,000 average comp = AUD 188,000.

**Investor Relations / Communications (2–3 people, 15–25% of time, 10 months):** Central estimate: 2.5 FTEs × 0.20 × (10/12) year × AUD 200,000 average comp = AUD 83,000.

**HR / People & Culture (2–3 people, 10–20% of time, 6 months):** Central estimate: 2.5 FTEs × 0.15 × (6/12) year × AUD 180,000 average comp = AUD 34,000.

**Risk and Compliance / Internal Audit (2–3 people, 10–20% of time, 6 months):** Central estimate: 2.5 FTEs × 0.15 × (6/12) year × AUD 200,000 average comp = AUD 38,000.

**Total internal resource cost:** approximately AUD 610,000.

Post-report, the implementation of 32 recommendations will consume additional internal resources over 12–24 months. However, much of this remediation work addresses genuine governance deficiencies that would need to be fixed regardless of whether a formal review was commissioned. The incremental cost attributable to the review (as opposed to the underlying problems) is the overhead of structuring the remediation as a formal response to published recommendations rather than as organic improvement. This overhead is estimated at 20–30% of the total remediation cost.

If total remediation consumes AUD 5–10 million in internal resources over 24 months (a reasonable estimate for 32 recommendations across governance, risk, culture, and remuneration frameworks), the incremental overhead is AUD 1–3 million.

**Total Component C cost:** AUD 610,000 + AUD 2,000,000 (remediation overhead, central) ≈ AUD 2,600,000.

### 4.3 Conversion to Decimal CAR


| Scenario | Cost estimate (AUD) | Decimal CAR |
| -------- | ------------------- | ----------- |
| Low      | 1,200,000           | -0.00012    |
| Central  | 2,600,000           | -0.00026    |
| High     | 4,500,000           | -0.00045    |


### 4.4 Confidence Assessment

**Medium-High.** The pre-report resource consumption is well-bounded because the functional teams involved and their approximate compensation are observable. The remediation overhead estimate is less precise but the incremental attribution (20–30% of total remediation cost) is a conservative assumption. The range is moderate (roughly a factor of 4 between low and high).

---

## 5. Aggregation

### 5.1 Point Estimates


| Component                 | Low          | Central      | High         |
| ------------------------- | ------------ | ------------ | ------------ |
| A: Reviewer fees          | -0.00020     | -0.00030     | -0.00050     |
| B: Management distraction | -0.00015     | -0.00040     | -0.00100     |
| C: Internal resources     | -0.00012     | -0.00026     | -0.00045     |
| **Total direct cost**     | **-0.00047** | **-0.00096** | **-0.00195** |


The total direct cost of commissioning an external board governance review, for an ASX-listed company with a market capitalisation of AUD 10 billion, is estimated at a CAR of approximately **-0.00096** (9.6 basis points), with a plausible range of -0.00047 to -0.00195.

In dollar terms, this corresponds to a central estimate of AUD 9.6 million, with a range of AUD 4.7 million to AUD 19.5 million.

### 5.2 Interpretation

Three observations are noteworthy.

First, the direct costs are dominated by management distraction (Component B), which accounts for approximately 40% of the central estimate and an even larger share of the uncertainty. Reviewer fees, the most visible cost and the one most likely to be debated in boardroom discussions about whether to commission a review, account for less than a third of the total and are the component estimated with the highest confidence.

Second, the total direct cost of approximately -0.0010 is small relative to the factors on the benefit side of the utility function. For comparison, the reputation restoration factor alone was assigned a CAR impact of +0.0040 in the utility model, conditional on a positive review outcome. This means the direct cost of commissioning the review is recovered if there is even a modest probability (~25%) of meaningful reputation improvement.

Third, these direct costs are also small relative to the market capitalisation loss that typically motivates the review. Qantas lost approximately AUD 3 billion (~3,000 bps) in market capitalisation during the crisis period. Spending 10 bps to address a 3,000 bps problem is a cost-to-problem ratio of roughly 0.3%, which suggests that direct costs should rarely be the binding constraint on the decision to commission a review.

---

## 6. Recommended Prior Distribution for Total Direct Cost

### 6.1 Distribution Selection

The total direct cost is the sum of three components with different uncertainty profiles. Component A (fees) is approximately lognormal with low variance. Component B (distraction) is approximately lognormal with moderate-to-high variance and a heavy right tail (distraction can escalate significantly if the review becomes contested or extended). Component C (internal resources) is approximately lognormal with moderate variance.

The sum of lognormal random variables does not have a closed-form distribution, but is well-approximated by a lognormal distribution when the components have similar orders of magnitude, which they do in this case.

However, a **log-normal distribution** is not ideal here because it constrains the cost to be strictly positive (which is correct — costs cannot be negative) but has a right tail that may be too heavy for the fee and internal resource components while not heavy enough for the distraction component.

The recommended prior is a **Gamma distribution**, which offers several advantages for this application. It is defined on the positive real line, which is appropriate since costs are strictly non-negative. It has two parameters (shape α and rate β) that can be calibrated to match the estimated mean and variance. It accommodates moderate right-skewness, reflecting the possibility that distraction costs escalate beyond the central estimate. It is conjugate to several common likelihood functions, facilitating Bayesian updating as empirical data becomes available. Its shape is flexible — at low α it approximates an exponential (heavy right tail); at high α it approaches a normal distribution.

### 6.2 Parameter Calibration

Working in decimal CAR units (costs expressed as positive values, with the understanding that these are subtracted from the utility function):

**Target moments from the analysis above:**

The central estimate (mean) of the total direct cost is 0.00096.

The range of -0.00047 to -0.00195 is interpreted as an approximate 90% credible interval. This implies a standard deviation of approximately (0.00195 - 0.00047) / (2 × 1.645) ≈ 0.00045.

**Gamma parameterisation:**

For a Gamma distribution with shape α and rate β:

$$\text{Mean} = \frac{\alpha}{\beta}, \quad \text{Variance} = \frac{\alpha}{\beta^2}$$

Solving:

$$\alpha = \frac{\mu^2}{\sigma^2} = \frac{(0.00096)^2}{(0.00045)^2} = \frac{9.216 \times 10^{-7}}{2.025 \times 10^{-7}} \approx 4.55$$

$$\beta = \frac{\mu}{\sigma^2} = \frac{0.00096}{2.025 \times 10^{-7}} \approx 4{,}741$$

### 6.3 Recommended Prior

$$C_{direct} \sim \text{Gamma}(\alpha = 4.55, ; \beta = 4741)$$

where $C_{direct}$ is expressed as a positive decimal CAR value (to be subtracted from the total utility).

**Properties of this distribution:**


| Property           | Value   |
| ------------------ | ------- |
| Mean               | 0.00096 |
| Standard deviation | 0.00045 |
| Mode               | 0.00075 |
| 5th percentile     | 0.00031 |
| 25th percentile    | 0.00063 |
| Median             | 0.00089 |
| 75th percentile    | 0.00122 |
| 95th percentile    | 0.00185 |
| Skewness           | 0.94    |


The positive skewness (0.94) reflects the asymmetric risk profile: direct costs are more likely to exceed the central estimate than to fall below it, primarily because management distraction can escalate if the review becomes prolonged or contested, but there is a natural floor on costs (the review cannot cost less than the reviewer's fees).

### 6.4 Sensitivity to Market Capitalisation

The Gamma parameters above are calibrated for M = AUD 10 billion. For a company with a different market capitalisation, the distribution can be rescaled by adjusting the rate parameter:

$$\beta_{adj} = \beta \times \frac{M}{M_{ref}} = 4741 \times \frac{M}{10 \times 10^9}$$

The shape parameter α remains unchanged, as it reflects the relative uncertainty structure of the cost components rather than their absolute magnitude. This assumes that reviewer fees and internal resource costs scale less than proportionally with firm size (a reasonable assumption — the review process is similar regardless of whether the company has a $5B or $50B market cap), while management distraction scales approximately proportionally (larger firms have more executives but also more complex operations to distract from).

### 6.5 Bayesian Updating

As empirical observations from future governance reviews become available, the Gamma prior can be updated using a Gamma-Poisson or Gamma-Gamma conjugate structure, depending on the likelihood model adopted. Alternatively, if direct cost observations become available in dollar terms (from annual report disclosures or litigation discovery), these can be converted to CAR units and used to update the distribution via standard Bayesian methods.

The recommended update strategy is to maintain a running posterior using each new observation from the reference panel as it becomes available (e.g., if the Qantas review costs are eventually disclosed in sufficient detail). With the current panel providing at most 2–3 usable observations for board-initiated reviews, the prior will dominate for the foreseeable future, which reinforces the importance of the calibration exercise above.

---

## 7. Limitations

Several limitations should be noted. First, the management distraction estimate (Component B) rests on transferability assumptions from the M&A distraction literature that have not been directly validated for governance reviews. Second, the remediation overhead attribution (20–30% of total remediation cost) is a judgment-based estimate. Third, the analysis assumes a single review engagement of approximately 10 months; a review that extends beyond this period or that triggers a supplementary review (as occurred with Star Entertainment's Bell One and Bell Two) would have substantially higher costs. Fourth, the Gamma distribution is a parametric convenience; the true cost distribution may have features (e.g., a point mass at zero for the scenario where the review is commissioned but subsequently abandoned) that are not captured. Fifth, all estimates are conditional on a market capitalisation of AUD 10 billion and would need to be recalibrated for firms of substantially different size.

---

*This report addresses direct costs only. Indirect costs — including litigation exposure from published findings, regulatory invitation effects, competitive intelligence leakage, and signalling costs — are excluded and are estimated to be 3–10× larger in magnitude than the direct costs reported here.*