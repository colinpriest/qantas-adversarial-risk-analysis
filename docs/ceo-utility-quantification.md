# CEO Utility Quantification

## 1. Overview

The CEO's utility function and decision probabilities are **literature-calibrated** rather than estimated via an LLM+Stan pipeline (unlike the Board and ASA actors). This reflects two constraints:

1. **No revealed-preference data.** The Board's D1 action and ASA's A2 recommendation are observable choices from which utility weights can be estimated via softmax maximum likelihood. The CEO's D0 decision (resign or stay) is a single binary observation — insufficient to identify a multi-parameter utility function econometrically.

2. **Rich qualitative record.** Joyce's compensation structure, clawback provisions, contract terms, and departure circumstances are publicly documented in detail (ASX remuneration reports, governance review findings, media accounts). Combined with established behavioural economics (prospect theory, CRRA preferences, evaluative stigma), these data support credible calibration without statistical estimation.

The CEO utility function is a **reference-dependent CRRA** specification with Kahneman–Tversky loss aversion, applied to both monetary and non-monetary components.

---

## 2. Functional Form

$$U_{\text{total}} = U_{\text{money}}(W) - \lambda_D \cdot D_{\text{raw}}$$

### 2.1 Monetary Component

The monetary utility uses constant relative risk aversion (CRRA) with reference-dependent loss aversion:

$$\text{CRRA}(W) = \frac{W^{1-\gamma}}{1 - \gamma}, \quad \gamma \neq 1$$

Reference dependence (Tversky & Kahneman, 1992):

$$U_{\text{money}}(W) = \begin{cases} \text{CRRA}(W) & W \geq W_{\text{ref}} \\ \lambda \cdot \text{CRRA}(W) - (\lambda - 1) \cdot \text{CRRA}(W_{\text{ref}}) & W < W_{\text{ref}} \end{cases}$$

This is continuous at $W_{\text{ref}}$ and amplifies the utility drop below the reference point by exactly $\lambda$. In practice, all game outcomes have $W < W_{\text{ref}}$, so the CEO is always in the loss domain.

### 2.2 Non-Monetary Component

$D_{\text{raw}}$ captures reputational damage, public humiliation, career destruction, and legal exposure. It is additive across game outcomes and scaled by a separate loss aversion coefficient $\lambda_D$:

$$D_{\text{raw}} = D_{\text{base}} + \sum_k D_k \cdot \mathbb{I}[\text{condition}_k]$$

The $\lambda_D$ multiplier reflects that an executive with high status and ego evaluates reputational losses relative to their expected position as a powerful CEO — making sacking, public disgrace, and AGM humiliation feel disproportionately worse than their raw values.

---

## 3. Parameter Specification

### 3.1 Preference Parameters

| Parameter | Value | Source | Description |
|:----------|:-----:|:-------|:------------|
| $\gamma$ | 1.5 | Tversky & Kahneman (1992); truncated to [0.5, 3.0] | Risk aversion coefficient |
| $\lambda$ | 2.25 | Tversky & Kahneman (1992) cumulative prospect theory | Loss aversion on monetary outcomes |
| $W_{\text{ref}}$ | 16.0 | Pre-crisis expected total compensation (A$M) | Reference point for loss aversion |
| $\lambda_D$ | 2.25 | Default equals $\lambda$; can be set independently | Loss aversion on non-monetary penalties |

**Calibration rationale for $\gamma = 1.5$.** CRRA with $\gamma$ around 1.5 makes the CEO dislike variance in wealth outcomes. This is centred on the canonical estimate for moderately risk-averse agents in the executive compensation literature. The prior $\gamma \sim \mathcal{N}(1.5, 0.5^2)$, truncated to [0.5, 3.0], is used when opponents model the CEO's risk preferences.

**Calibration rationale for $\lambda = 2.25$.** This is the canonical loss aversion coefficient from cumulative prospect theory (Tversky & Kahneman, 1992). Malmendier & Tate (2005) document that overconfident CEOs with high media profiles exhibit stronger loss aversion in corporate decisions, supporting a value at or above the population estimate for Joyce specifically.

**Calibration rationale for $W_{\text{ref}} = 16.0$.** Joyce's pre-crisis expected total compensation (salary + STI + LTI) was approximately A$16M. His initial FY23 remuneration was reported at A$21.4M before clawbacks, and he executed a A$16.85M share sale in June 2023. The reference point reflects what Joyce expected to receive absent the governance crisis.

### 3.2 Wealth Outcomes (A$M Equivalent Units)

Wealth depends on the CEO's departure mode. All values reflect post-ACCC calibration: the Board had flagged clawback of up to A$14.4M, LTIs were frozen, STIs were under scrutiny, and legal costs were mounting.

| Departure Mode | Parameter | Value | Formula | Rationale |
|:---------------|:----------|:-----:|:--------|:----------|
| Pre-game resign (D0) | $W_{\text{resign}}$ | 8.0 | Direct | "Good leaver" status preserved. Partial bonus retained, controlled narrative, moderate clawback. Joyce exits with some pay but forfeits significant LTI. |
| Stay & kept | $W_{\text{stay\_kept}}$ | 7.0 | Direct | Even keeping the job post-crisis: frozen LTIs, reduced STI, legal costs. Markedly lower than $W_{\text{ref}} = 16$ because ACCC proceedings ensure ongoing financial drag. |
| Stay & negotiate (D4) | $W_{\text{negotiate}}$ | 3.75 | $(W_{\text{sacked}} + W_{\text{kept}}) / 2$ | Midpoint between sacked and kept — CEO retains some bargaining power but accepts reduced terms. |
| Stay & resign late (D4) | $W_{\text{resign\_late}}$ | 1.95 | $W_{\text{sacked}} \times 1.3$ | Slightly better than sacked (30% premium for voluntary departure) but post-AGM timing eliminates most "good leaver" protections. |
| Stay & sacked | $W_{\text{stay\_sacked}}$ | 0.5 | Direct | Full clawback after forced removal. Board demonstrated willingness to claw back A$9.26M even in the voluntary scenario; forced sacking after sustained defiance triggers near-total forfeiture. Residual represents only base salary for months worked. |

**Key insight:** With ACCC-era calibration, even the best stay outcome ($W = 7.0$) is below $W_{\text{ref}} = 16.0$. Combined with loss aversion ($\lambda = 2.25$), the monetary component alone slightly favours resignation. The non-monetary penalties then amplify this asymmetry.

### 3.3 Non-Monetary Penalties

$D_{\text{raw}}$ is constructed additively from a baseline plus conditional terms:

| Parameter | Value | Condition | Rationale |
|:----------|:-----:|:----------|:----------|
| $D_{\text{stay}}$ | 25 | CEO stays (baseline crisis cost) | Ongoing legal exposure (ACCC), hostile media scrutiny, shareholder activism, board tension, months of uncertainty |
| $D_{\text{resign}}$ | 40 | CEO resigns at D0 | Moderate stigma — "taking responsibility" framing. Calibrated against Karpoff, Lee & Martin (2008): 70–90% CEO turnover within two years of enforcement actions |
| $D_{\text{sacked}}$ | 100 | Board fires CEO (additive to $D_{\text{stay}}$) | Maximum reputational destruction. Public firing, no control over narrative, career damage. Comparable to AMP CEO Craig Meller (2018, Royal Commission) or Crown directors removed after Bergin Inquiry |
| $D_{\text{resign\_late}}$ | 60 | CEO resigns at D4 (additive to $D_{\text{stay}}$) | Post-AGM voluntary resignation — controls narrative but damage already done. Comparable to Rio Tinto CEO Jean-Sébastien Jacques (2020, Juukan Gorge) |
| $D_{\text{negotiate}}$ | 45 | CEO negotiates exit at D4 (additive to $D_{\text{stay}}$) | Face-saving terms, close to $D_{\text{resign}} = 40$ but mid-crisis. Comparable to AMP CEO Francesco De Ferrari (2021) |
| $D_{\text{agm}}$ | 30 | Vote > 25% (first strike) | AGM humiliation — public rejection by shareholders |
| $D_{\text{disgrace}}$ | 30 | Overwhelming vote indicator | Additional public disgrace from extreme vote result |
| $D_{\text{adverse\_review}}$ | 10 | Review commissioned AND negative outcome | Adverse governance review findings add to stigma |

**Resign path total:** $D_{\text{raw}} = D_{\text{resign}} = 40$

**Stay path construction:**
$$D_{\text{raw}}^{\text{stay}} = D_{\text{stay}} + D_{\text{departure\_mode}} + D_{\text{agm}} \cdot \mathbb{I}[V > 0.25] + D_{\text{disgrace}} \cdot \mathbb{I}[\text{overwhelming}] + D_{\text{adverse}} \cdot \mathbb{I}[\text{review\_negative}]$$

**Worst-case stay path:** $D_{\text{raw}} = 25 + 100 + 30 + 30 + 10 = 195$ (sacked after overwhelming vote and adverse review). With $\lambda_D = 2.25$: effective penalty = $2.25 \times 195 = 438.75$, compared to resign path $2.25 \times 40 = 90$.

---

## 4. Departure-Mode Resolution

The CEO has two decision points where departure can occur:

### 4.1 D4 — Post-AGM Decision

After the AGM vote (V) and market reaction (M_agm), the CEO chooses: **stay**, **resign**, or **negotiate exit**. The `effective_d4` variable determines which departure-mode penalty applies:

```python
effective_d4 = outcome.d4_action
if outcome.d4_post_review_action in ("D4_resign", "D4_negotiate_exit"):
    effective_d4 = outcome.d4_post_review_action
```

This ensures that if the CEO acts at `D4_post_review` (after adverse review findings), that later action overrides the earlier D4 decision.

### 4.2 Departure-Mode Comparison

| | D0 resign | D4 resign | D4 negotiate | D4 sacked |
|:--|:---------:|:---------:|:------------:|:---------:|
| W | 8.0 | 1.95 | 3.75 | 0.50 |
| $D_{\text{raw}}$ (base) | 40 | 85 | 70 | 125 |
| $D_{\text{raw}}$ (+ first strike) | — | 115 | 100 | 155 |
| $D_{\text{raw}}$ (+ overwhelming + adverse) | — | 155 | 140 | 195 |

D0 resignation is unambiguously better: higher W, lower D, avoids the AGM entirely.

### 4.3 Worked Example — Strike at 33% Vote

With $\gamma = 1.5$, $\lambda = 2.25$, $W_{\text{ref}} = 16.0$, $\lambda_D = 2.25$:

| D4 Action | $D_{\text{raw}}$ | $\lambda_D \times D$ | W | $U_{\text{money}}$ | $U_{\text{total}}$ |
|:----------|:-----:|:-------:|:-:|:-------:|:-------:|
| Stay (not sacked) | 55 | 123.75 | 7.0 | −0.97 | −124.72 |
| Stay (sacked at D_rev) | 155 | 348.75 | 0.5 | −5.74 | −354.49 |
| Negotiate exit | 100 | 225.00 | 3.75 | −1.70 | −226.70 |
| Resign late | 115 | 258.75 | 0.65 | −4.96 | −263.71 |

If $\Pr(\text{Board sacks at D\_rev}) = 0.7$ after a strike:

$$\mathbb{E}[\text{stay}] = 0.3 \times (-124.72) + 0.7 \times (-354.49) = -285.56$$
$$\mathbb{E}[\text{negotiate}] = -226.70$$

Negotiate is ~59 utility points better — the CEO rationally negotiates.

---

## 5. D0_ceo: Pre-Game Resignation Decision

### 5.1 Bayesian Prior

The D0_ceo prediction is anchored by an empirical Bayesian prior derived from ASX moral-reputational crisis events.

**Prior family:** Jeffreys prior $\text{Beta}(0.5, 0.5)$ updated with 12 Australian observations of no-remorse CEOs facing moral-reputational crises — all 12 resigned.

$$p_{\text{departure}} \sim \text{Beta}(0.5 + 12, \; 0.5 + 0) = \text{Beta}(12.5, \; 0.5)$$

| Statistic | Value |
|:----------|:------|
| Prior mean | 0.962 |
| Prior mode | 0.958 |
| 90% credible interval | [0.85, 0.998] |
| Effective sample size | 13 |

**Joyce as archetype.** Joyce maps to the "no contrition" archetype: combative public posture throughout COVID complaints and ghost flights, record bonus while workers were stood down, lobbying against competition, and zero accountability signalling. In the filtered ASX data, the conditional departure rate without contrition strategy is 12/12 (100%). The $\beta = 0.5$ pseudo-count for survival is Jeffreys-type regularisation preventing prior degeneracy.

### 5.2 Level-2 ARA Prediction

The CEO's resign-or-stay decision is evaluated via **Level-2 Adversarial Risk Analysis**. This is necessary because the CEO's choice depends on what the Board will do *after* the CEO stays — and Level-1 treatment gets this badly wrong.

**Why Level-1 fails.** At Level-1, the CEO's rollouts use fixed policies for the Board's D1 action. The default Board policy returns `D0_minimal` (do nothing). The CEO therefore "sees" a world where the Board takes no action, faces no AGM or review risk, and rationally prefers to stay. This produces ~35% resign / ~65% stay — the opposite of what actually occurred.

**Level-2 mechanism.** When the CEO evaluates "what happens if I stay?", the simulation triggers a *nested* ARA `predict()` call at the Board's D1 decision node:

1. Sample Board utility parameters from the **CEO's priors about the Board** (see §7.1)
2. For each sampled Board type, simulate the full game tree forward from D1 to find the Board's best response
3. Aggregate across $K$ parameter samples to produce a predictive distribution over Board actions

The CEO now sees that the Board will very likely choose `D1_review`, which triggers higher expected vote percentages, possible adverse review findings, and risk of being sacked post-review. These downstream consequences make $\mathbb{E}[U(\text{stay})]$ substantially worse, pushing the predicted resignation probability above 70%.

### 5.3 Computational Parameters

Level-2 prediction is computationally expensive: the outer loop samples CEO types, each with rollouts, and each rollout reaching a Board decision node triggers an inner Level-1 prediction.

| Parameter | Value | Comparison |
|:----------|:-----:|:-----------|
| $K_{\text{d0\_ceo}}$ | 50 | vs $K = 200$ for main solve |
| $R_{\text{d0\_ceo}}$ | 10 | vs $R = 20$ for main solve |

The inner Board prediction reuses the same $K$ and $R$ from the D0_ceo `PredictiveDistribution` instance.

### 5.4 Prior–Evidence Combination

The engine combines the Beta prior with ARA-computed evidence via pseudo-count addition:

$$\alpha_{\text{post}} = 12.5 + \sum_{i=1}^{N} \Pr(\text{resign} \mid \text{draw}_i)$$
$$\beta_{\text{post}} = 0.5 + \sum_{i=1}^{N} \Pr(\text{stay} \mid \text{draw}_i)$$
$$\Pr(\text{resign}) = \frac{\alpha_{\text{post}}}{\alpha_{\text{post}} + \beta_{\text{post}}}$$

With $N = 100$ ARA draws, the prior has weight $13 / (13 + 100) = 11.5\%$. This is appropriate: the prior provides a meaningful anchor from empirical data while allowing the game-theoretic analysis to dominate.

---

## 6. Fixed Policies for Predictive Rollouts

When the CEO is an *opponent* (not the focal actor), the engine uses fixed policy thresholds from `governance_spec.xlsx` to determine the CEO's actions in rollouts:

### 6.1 D4 Policy

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| `resign_vote_threshold` | 0.40 | CEO resigns if vote percentage exceeds 40% |
| `resign_adverse_prob_threshold` | 0.60 | CEO resigns if probability of adverse review exceeds 60% |

Below these thresholds, the CEO's fixed policy is to stay.

### 6.2 D4_post_review Policy

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| `resign_vote_threshold` | 0.40 | Same threshold applied at post-review decision |

### 6.3 D_rev Sack Threshold

The Board's `sack_vote_threshold = 0.25` (first strike threshold) is critical for CEO incentives. After any first strike, the Board's fixed policy sacks the CEO. This means the CEO "sees" near-certain sacking risk after a strike, making negotiate/resign viable at D4. Typical D4 predictive distribution after a strike: negotiate ~72%, resign ~26%, stay ~2%.

---

## 7. Opponent Priors on CEO Parameters

When other actors model the CEO's behaviour, they sample CEO utility parameters from prior distributions stored in `opponent_priors.xlsx`.

### 7.1 Board's and ASA's Priors about CEO

Both the Board and ASA hold nearly identical priors about the CEO's utility parameters:

| Parameter | Distribution | Mean | SD | Description |
|:----------|:------------|:----:|:--:|:------------|
| $\gamma$ | $\mathcal{N}(1.5, 0.5)$ | 1.5 | 0.5 | Risk aversion coefficient |
| $W_{\text{resign}}$ | $\mathcal{N}(8.0, 2.0)$ | 8.0 | 2.0 | Wealth if pre-game resign |
| $W_{\text{stay\_sacked}}$ | $\mathcal{N}(0.5, 0.3)$ | 0.5 | 0.3 | Wealth if sacked |
| $W_{\text{stay\_kept}}$ | $\mathcal{N}(7.0, 2.0)$ | 7.0 | 2.0 | Wealth if kept in position |
| $D_{\text{resign}}$ | $\mathcal{N}(40, 15)$ | 40 | 15 | Pre-game resign penalty |
| $D_{\text{sacked}}$ | $\mathcal{N}(100, 30)$ | 100 | 30 | Sacked penalty |
| $D_{\text{resign\_late}}$ | $\mathcal{N}(60, 20)$ | 60 | 20 | Late resign penalty |
| $D_{\text{negotiate}}$ | $\mathcal{N}(45, 15)$ | 45 | 15 | Negotiate exit penalty |
| $D_{\text{agm}}$ | $\mathcal{N}(30, 10)$ | 30 | 10 | First strike AGM penalty |
| $D_{\text{disgrace}}$ | $\mathcal{N}(30, 10)$ | 30 | 10 | Overwhelming vote penalty |
| $D_{\text{adverse\_review}}$ | $\mathcal{N}(10, 5)$ | 10 | 5 | Adverse review penalty |
| $D_{\text{stay}}$ | $\mathcal{N}(25, 8)^*$ | 25 | 8* | Baseline crisis cost |
| $\lambda$ | $\mathcal{N}(2.25, 0.3)$ | 2.25 | 0.3 | Loss aversion (monetary) |
| $W_{\text{ref}}$ | $\mathcal{N}(16, 3)$ | 16 | 3 | Reference point |
| $\lambda_D$ | $\mathcal{N}(2.25, 0.3)$ | 2.25 | 0.3 | Loss aversion (non-monetary) |

\* ASA's prior on $D_{\text{stay}}$ uses SD = 10 (wider uncertainty) vs Board's SD = 8.

Prior means are centred on the spec defaults from `governance_spec.xlsx`. Standard deviations reflect meaningful uncertainty — opponents are unsure about the CEO's exact risk preferences, wealth situation, and sensitivity to different departure modes.

### 7.2 CEO's Priors about the Board

For Level-2 ARA at D0_ceo, the CEO holds priors about the Board's utility parameters:

| Parameter | Distribution | Mean | SD |
|:----------|:------------|:----:|:--:|
| `vote_penalty_weight` | $\mathcal{N}(2.0, 0.5)$ | 2.0 | 0.5 |
| `ceo_loss_cost` | $\mathcal{N}(1.5, 0.5)$ | 1.5 | 0.5 |
| `spill_risk_weight` | $\mathcal{N}(2.5, 0.5)$ | 2.5 | 0.5 |
| `review_car_weight` | $\mathcal{N}(15.0, 3.0)$ | 15.0 | 3.0 |
| `review_direct_cost_weight` | $\mathcal{N}(15.0, 3.0)$ | 15.0 | 3.0 |
| `implementation_cost_sack` | $\mathcal{N}(0.3, 0.1)$ | 0.3 | 0.1 |
| `board_passivity_after_departure` | $\mathcal{N}(0.5, 0.2)$ | 0.5 | 0.2 |
| `second_strike_spill_penalty` | $\mathcal{N}(8.0, 2.0)$ | 8.0 | 2.0 |
| `board_regulatory_liability` | $\mathcal{N}(5.0, 1.5)$ | 5.0 | 1.5 |
| `board_d1_liability` | $\mathcal{N}(4.0, 1.0)$ | 4.0 | 1.0 |
| `qantas_legal_d1_penalty` | $\mathcal{N}(3.0, 1.0)$ | 3.0 | 1.0 |
| `qantas_legal_d_rev_penalty` | $\mathcal{N}(2.0, 0.8)$ | 2.0 | 0.8 |
| `negative_review_finding_penalty` | $\mathcal{N}(5.0, 2.0)$ | 5.0 | 2.0 |
| `balanced_review_finding_penalty` | $\mathcal{N}(2.5, 2.0)$ | 2.5 | 2.0 |
| `ceo_loss_shock_strike` | $\text{LogN}(-0.92, 0.4)$ | — | — |
| `ceo_loss_shock_overwhelming` | $\text{LogN}(-0.69, 0.4)$ | — | — |
| `ceo_loss_shock_adverse` | $\text{LogN}(-0.69, 0.4)$ | — | — |

The CEO's beliefs about the Board mirror the ASA → Board prior structure. The lognormal shock attenuation priors have medians $e^{-0.92} \approx 0.40$ and $e^{-0.69} \approx 0.50$, consistent with the engine's shock relief mechanism.

---

## 8. Literature Sources

### 8.1 Prospect Theory and Loss Aversion

- **Tversky, A. & Kahneman, D. (1992).** Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5, 297–323.
  - Source of $\lambda = 2.25$ and the reference-dependent utility framework.
- **Malmendier, U. & Tate, G. (2005).** CEO overconfidence and corporate investment. *Journal of Finance*, 60(6), 2661–2700.
  - Documents that high-profile, overconfident CEOs exhibit stronger loss aversion in decisions.

### 8.2 CEO Turnover and Stigma

- **Gow, I., Larcker, D. & Tayan, B. (2017).** Retired or fired: How can investors tell if the CEO left voluntarily? Stanford Closer Look Series.
  - Push-out Score (0–10 scale); 48% of departures in ambiguous zone; mutual incentives to disguise forced exits.
- **Semadeni, M. et al. (2008).** Leaders from failed organisations face demotion and reduced board appointments via evaluative stigma.
- **Bilinski, P. & Novak, J.** Executives at stigmatised firms demand compensation premiums, move to less prestigious firms, accept lower pay.
- **Karpoff, J., Lee, D. & Martin, G. (2008).** 70–90% total CEO turnover within two years of enforcement actions.
  - Calibration anchor for $D_{\text{resign}} = 40$ (moderate stigma relative to enforcement-action base rate).

### 8.3 ESG and CEO Turnover

- **Çolak, G., Korkeamäki, T. & Meyer, N. (2023).** ESG and CEO turnover around the world. *Journal of Corporate Finance*, 84, 102523.
  - Extreme ESG risk doubles CEO turnover odds; non-pecuniary motives operate independently of CARs; stronger effects in stakeholder-oriented countries (Australia ranks high).
  - Source of the Board's sacking imperative calibration and the empirical basis for the Beta(12.5, 0.5) prior.

### 8.4 Integrity vs Competence Failures

- **Connelly, B. et al. (2016).** Investor perceptions of CEO successor selection in the wake of integrity and competence failures. *Strategic Management Journal*, 37, 2135–2151.
  - Investors strongly distinguish integrity from competence failures; integrity failures create strong pressure for visible leadership change.
- **Gentry, R. et al. (2021).** A database of CEO turnover and dismissal in S&P 1500 firms, 2000–2018. *Strategic Management Journal*.
  - Comprehensive turnover coding; misconduct-related dismissals carry strong negative signalling.

### 8.5 Joyce-Specific Sources

- **Qantas FY23 Remuneration Report.** Initial total remuneration A$21.4M; A$16.85M share sale June 2023.
- **Qantas Governance Review (August 2024).** Board clawed back A$9.26M (LTI forfeiture + STI reduction); found "no deliberate wrongdoing" but "considerable harm."
- **ACCC v Qantas Airways (filed July 2023).** Federal Court proceedings for selling tickets on cancelled flights; established legal exposure context.
- **Qantas 2023 AGM (3 November 2023).** Record vote against remuneration report; overwhelming strike.

---

## 9. Why No Statistical Estimation Pipeline

The Board and ASA utility functions are estimated via LLM-generated scenario ratings and Stan ordinal probit / softmax MLE models (`board_utility_quantification.py`, `asa_utility_quantification.py`). The CEO has no equivalent pipeline for three reasons:

1. **Single observation.** The CEO made one binary choice (resign on 5 September 2023). A softmax model with 15 parameters cannot be identified from a single data point.

2. **No natural rating instrument.** The Board and ASA pipelines use LLM-generated expert ratings across multiple hypothetical scenarios to create synthetic choice data. For the CEO, the utility function parameters (wealth values, non-monetary penalties) are directly observable from public records — there is no need for revealed-preference estimation.

3. **Strong theory.** The CEO utility function is grounded in established behavioural economics (CRRA, prospect theory, evaluative stigma) with parameter values calibrated to documented facts (contract terms, clawback amounts, remuneration disclosures). This produces more credible estimates than statistical estimation from limited data.

The opponent prior distributions (§7) provide the uncertainty quantification that a Bayesian estimation pipeline would otherwise deliver — they encode how much other actors are uncertain about the CEO's exact preferences.
