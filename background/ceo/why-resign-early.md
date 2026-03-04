# CEO Utility For Resigning Immediately

Alan Joyce’s bonus outcomes were not determined by a specific “AGM vote on his bonus” that he could avoid by resigning early, because:

- At the 2023 AGM, shareholders voted on the **remuneration report**, not directly on whether Joyce personally kept or lost his bonus. The chair explicitly acknowledged an “overwhelming vote against our remuneration report,” i.e., a record protest strike, but that vote is advisory under Australian law.investor.qantas+1
- Joyce’s FY23 remuneration was instead controlled by the **board’s discretion** and later by the governance review. The board initially reserved the right to withhold up to A$14.4m of his variable pay, and after the governance review it decided to cut his FY23 remuneration by A$9.26m (forfeiting long‑term incentives and reducing short‑term incentives), independent of any AGM [timing.abc](http://timing.abc)+5

So while resigning before the AGM may have been politically convenient, the loss of a large portion of his bonus ultimately came from the board’s post‑review decision, not from an AGM vote he sidestepped.

You can’t know his true utility function, but you can frame “leave now vs stay” using what’s on the record plus generic board‑behaviour economics.

## 1. Stated and widely inferred reasons

Publicly, Qantas and Joyce framed the early exit as about **accelerating renewal** and responding to pressure:

- Qantas said he would bring forward retirement “to help the company accelerate its renewal,” and Joyce said the focus on Qantas and events of the past made it clear the company needed to move ahead with renewal as a [priority.abc](http://priority.abc)+2
- Commentators described the move as bowing to intense public and political anger following the ACCC lawsuit over cancelled flights and broader controversies.aerotime+4

So at a minimum, an immediate resignation reduces ongoing reputational damage to the company and the board, and potentially protects his own public image relative to being forced out after a bruising AGM.

## 2. Bonus and pay dynamics

The bonus story cuts both ways:

- At the time of his September 2023 exit it was already unclear “what will happen with his entitlements now he has stepped down”; unions and others were calling for his bonuses to be stripped.wsws+1
- The later governance review found “considerable harm” from leadership decisions and led the board in August 2024 to cut more than A$9.2m from his final payout, including forfeited long‑term incentives and reduced short‑term incentives, even while finding no deliberate [wrongdoing.ch](http://wrongdoing.ch)-aviation+7

That outcome shows: resigning early did **not** protect him from significant clawback once the board had a governance review in hand. However, from Joyce’s perspective in early September 2023, stepping down might plausibly have increased the probability of retaining *some* bonus compared with waiting to be pushed out after further reputational damage and a hostile AGM.

In a utility story for your case study, you can reasonably posit:

- If he stayed through a disastrous AGM, the political pressure on the board to zero out his variable pay ex post would have been even stronger.
- By exiting “voluntarily” and giving the board a clean slate, he improves the optics of partial, rather than total, forfeiture.

You can’t prove that with data, but it is behaviourally consistent with how boards balance accountability and contractual expectations.

## 3. Other utilities from leaving immediately

Beyond bonus protection, there are several tangible and intangible benefits to leaving when he did:

- **Avoiding a public humiliation at the AGM:** Shareholder anger over pay and governance was already intense; the 2023 AGM subsequently delivered an overwhelming vote against the remuneration report. Exiting beforehand spared him direct confrontation and questioning in that forum.investor.qantas+1
- **Protecting the board and institution:** Early removal of a lightning‑rod CEO is a classic damage‑control tool: it gives the board something visible to offer angry stakeholders and regulators and supports the “new management, new culture” narrative.reuters+4
- **Legal and regulatory posture:** Stepping aside while ACCC and other proceedings were live separates ongoing defence and remediation from the individual most associated with the alleged conduct, which can be useful in settlement dynamics and in board‑commissioned reviews.flightglobal+2
- **Personal career/legacy:** He exits with a record profit and can frame his departure as putting the company first, rather than being pushed after a formal finding of wrongdoing. That has non‑trivial value for future roles, boards, speaking, and reputation.fortune+2

## 4. How to use this in your model

For your case study utility comparison:

- Model his choice as trading off:
  - foregone salary for two months,
  - a higher chance that *some* bonus / LTI is preserved,
  - reduced probability of more severe reputational and legal fallout attaching personally to him,
  - increased probability that the board and review frame problems as “systemic” rather than “this CEO refused to go.”
- Model the board’s side as gaining:
  - a salient accountability signal (CEO exit under pressure),
  - a stronger mandate to commission a hard governance review,
  - better odds of calming investors, regulators and politicians.

Your specific hypothesis—early resignation as a **signal** that makes it politically easier for the board to leave him with *some* bonus rather than none—is not directly documented in the reporting, but it is entirely consistent with the subsequent pattern: big cuts and forfeitures, but not a full wipe‑out, after a review that stressed “no deliberate wrongdoing” alongside “considerable harm.”abc+6

# Bayesian Assumptions

You can model this compactly with a two‑state world (resign now vs stay and be sacked), and then put a simple parametric utility over money plus a big non‑monetary “punishment” term.

## 1. Monetary part: CRRA utility over wealth

Let W*W* be Joyce’s eventual personal wealth from Qantas (salary over next months + expected bonus / LTI actually received, net of legal costs and clawbacks). Use constant‑relative‑risk‑aversion (CRRA):

Umoney(W)={W1−γ1−γγ≠1ln⁡Wγ=1*U*money(*W*)={1−*γW*1−*γ*ln*Wγ*=1*γ*=1

A reasonable **Bayesian prior** on the risk‑aversion parameter for a very wealthy executive could be:

- γ∼N(1.5, 0.52)*γ*∼N(1.5,0.52) truncated to [0.5,3][0.5,3].

That centres him as moderately risk‑averse but not extremely so.

### Loss aversion (Kahneman-Tversky reference dependence)

The standard CRRA model treats gains and losses symmetrically, but behavioural evidence strongly suggests executives are **loss-averse**: wealth losses below a reference point are felt disproportionately more than equivalent gains.

Following Tversky & Kahneman (1992), we apply cumulative prospect theory to the monetary component:

- **Reference point** W_ref ~ N(16, 3): pre-crisis expected total compensation (salary + STI + LTI). All game outcomes involve W < W_ref, so the CEO is always in the loss domain.
- **Loss aversion coefficient** lambda ~ N(2.25, 0.3): losses are felt 2.25x as strongly as equivalent gains. This is the canonical estimate from the cumulative prospect theory literature.

The reference-dependent CRRA utility is:

- W >= W_ref: U_money = CRRA(W)   [standard]
- W <  W_ref: U_money = lambda * CRRA(W) - (lambda - 1) * CRRA(W_ref)   [losses amplified]

This is continuous at W_ref and amplifies the utility drop below the reference by exactly lambda. In practice, it makes the sacked path (W=3, loss of 13 from reference) feel much worse than the resign path (W=8, loss of 8), beyond what CRRA curvature alone captures.

Loss aversion also applies to the **non-monetary penalties** (D terms) via a separate coefficient lambda_D ~ N(2.25, 0.3).  Joyce had an outsized ego and identity invested in his role as a powerful, high-profile CEO.  Being fired, publicly disgraced at an AGM, or having adverse review findings are evaluated relative to his expected status as a respected industry leader — not in absolute terms.  The total utility is therefore:

  U_total = U_money(W) - lambda_D * D_raw

With lambda_D = 2.25, the worst-case stay path (D_raw = 195) produces an effective penalty of 2.25 * 195 = 439, compared to the resign path (2.25 * 40 = 90).  The ratio between worst-case stay and resign penalties widens from 4x (raw) to the same 4x but at much higher absolute magnitude, making even moderate sacking probabilities devastating to EU(stay).

This is consistent with the psychology of powerful executives: Malmendier & Tate (2005) document that overconfident CEOs with high media profiles exhibit stronger loss aversion in their corporate decisions

## 2. Non‑monetary penalty: reputation / legal disutility

Let D*D* be an additive disutility term capturing reputation damage, public humiliation (e.g., at AGM), and being a named defendant in legal action:

Utotal=Umoney(W)−D*U*total=*U*money(*W*)−*D*

Model D*D* differently by scenario:

- If he **resigns now**: D = D_resign ~ N(40, 15) — moderate stigma, calibrated against empirical base rates: Karpoff, Lee & Martin (2008) and Desai, Hogan & Wilkins (2006) find 70-90% total CEO turnover within two years of enforcement actions. A lower D_resign (40 vs the original 50) reflects that Joyce framed his departure positively as "accelerating renewal," reducing stigma relative to a typical forced departure
- If he **stays**: D starts at a **baseline crisis cost** D_stay ~ N(25, 8), capturing the inherent cost of remaining through a governance crisis regardless of the final outcome:
  - Ongoing legal exposure (ACCC proceedings)
  - Hostile media scrutiny ("daily front-page headlines")
  - Shareholder activism campaigns
  - Board tension and months of uncertainty

  Conditional penalties then accumulate on top of D_stay:
  - D_agm ~ N(30, 10) if vote > 25% (AGM humiliation)
  - D_disgrace ~ N(30, 10) if overwhelming vote (public disgrace)
  - D_sacked ~ N(100, 30) if ultimately removed (career destruction)
  - D_adverse_review ~ N(10, 5) if review finds adverse results

  The worst-case stay path (sacked after overwhelming vote and adverse review) gives D = 25 + 100 + 30 + 30 + 10 = 195, roughly 4x the resign penalty.

These are in the same arbitrary "utils" scale as U_money; calibrate them so that the expected penalty is of similar magnitude to a meaningful fraction of his financial utility (e.g., equivalent to losing many millions).

## 3. Money outcomes by scenario (ACCC-era calibration)

The ACCC lawsuit (filed July 2023) fundamentally changed the CEO's wealth prospects.  The Board had already publicly flagged potential clawback of up to A$14.4M of variable pay, LTIs were frozen pending review, STIs were under scrutiny, and legal costs were mounting.  Wealth values are calibrated to this post-ACCC environment:

- W_resign ~ N(8, 2): partial bonus retained, controlled narrative, moderate clawback. Joyce exits with some pay but forfeits significant LTI.
- W_stay_kept ~ N(7, 2): even keeping the job, the CEO faces substantial pay erosion — frozen LTIs, reduced STI, legal costs.  This is markedly lower than the pre-crisis W_ref = 16 because the ACCC proceedings ensure ongoing financial drag regardless of outcome.
- W_stay_sacked ~ N(0.5, 0.3): full clawback after forced removal. Justified by the Board's demonstrated willingness to claw back A$9.26M in the actual (voluntary departure) scenario; a forced sacking after sustained defiance would trigger near-total forfeiture of all variable pay. Legal costs, forfeited LTIs, STI clawback, and reputational destruction of future earning capacity leave essentially nothing.

The key insight: with ACCC-era calibration, even the best stay outcome (W=7) is worse than the resign outcome (W=8).  Combined with loss aversion, the monetary component alone slightly favours resignation.  The D terms then amplify this — the stay path's conditional penalties (sacking, AGM humiliation, disgrace) are devastating under loss-averse evaluation.

## 4. Bayesian structure

You then have:

- Prior on γ*γ*, the risk‑aversion parameter.
- Priors on WR,WS*WR*,*WS* and on DR,DS*DR*,*DS*.

In Bayesian terms, you could write his *scenario‑specific* expected utilities as:

EU(resign)=E[Umoney(WR)−DR ∣ information at Sep 2023]*EU*(resign)=E[*U*money(*WR*)−*DR* information at Sep 2023]EU(stay)=E[Umoney(WS)−DS ∣ information at Sep 2023]*EU*(stay)=E[*U*money(*WS*)−*DS* information at Sep 2023]

Given your assumption of **certainty** of sacking and legal action if he stays, there’s no scenario probability on that branch: the uncertainty is only over the magnitudes.

### Level‑2 ARA treatment of D0_ceo

The CEO’s resign‑or‑stay decision (D0_ceo) is modelled as a pre‑game decision node evaluated via **Level‑2 Adversarial Risk Analysis**. This is critical because the CEO’s choice depends on what the Board is likely to do *after* the CEO decides to stay — and a naïve Level‑1 treatment gets this badly wrong.

**Why Level‑1 fails.** At Level‑1, the CEO’s rollouts use fixed policies for the Board’s subsequent D1 action. The default Board policy returns `D0_minimal` (do nothing). The CEO therefore "sees" a world where the Board takes no action, faces no risk from the AGM or governance review, and rationally prefers to stay. This produces an inverted prediction: ~35% resign, ~65% stay — the opposite of what actually occurred.

**Level‑2 fix.** At Level‑2, when the CEO evaluates "what happens if I stay?", the simulation triggers a *nested* ARA `predict()` call at the Board’s D1 decision node. This inner Level‑1 prediction:

1. Samples Board utility parameters from the **CEO’s priors about the Board** (a new prior block: `CEO → Board` in `opponent_priors.xlsx`)
2. For each sampled Board type, simulates the full game tree forward from D1 to find the Board’s best response
3. Aggregates across K parameter samples to produce a predictive distribution over Board actions (D0_minimal, D1_review, D3_ceo_transition)

The CEO now sees that the Board will very likely choose `D1_review`, which triggers:
- Higher expected vote percentages (governance scrutiny raises voter engagement)
- Possible adverse review findings (negative CAR, reputational damage)
- Risk of being sacked post‑review (Drev_sack_ceo)

These downstream consequences make EU(stay) substantially worse, pushing the predicted resignation probability above 70%.

**CEO → Board priors.** The CEO’s beliefs about Board utility parameters (added to `opponent_priors.xlsx`) mirror the ASA → Board priors with the same structure:

| Parameter | Distribution | Interpretation |
|-----------|-------------|----------------|
| vote_penalty_weight | N(2.0, 0.5) | Board dislikes high opposition votes |
| ceo_loss_cost | N(-1.5, 0.5) | Net benefit of removing tainted CEO (negative = reward) |
| spill_risk_weight | N(2.5, 0.5) | Board fears board spill motions |
| review_car_weight | N(15.0, 3.0) | Sensitivity to review market impact |
| review_direct_cost_weight | N(15.0, 3.0) | Sensitivity to review direct costs |
| implementation_cost_sack | N(0.3, 0.1) | Reduced CEO transition cost (stakeholder support) |
| early_ceo_departure_cost | N(0.5, 0.2) | Disruption from early resignation |

**Computational tractability.** Level‑2 is expensive: the outer loop samples K_ceo CEO types, each with R rollouts, and each rollout that reaches a Board decision node triggers an inner Level‑1 prediction with its own K × R rollouts. To keep wall‑clock time reasonable the engine uses reduced parameters for the D0_ceo prediction:

- `K_d0_ceo = 50` (vs K = 200 for the main solve) — CEO parameter samples
- `R_d0_ceo = 10` (vs R = 20 for the main solve) — rollouts per CEO action

The inner Board prediction reuses the same K and R from the D0_ceo PredictiveDistribution instance. With parallel workers, D0_ceo prediction completes in under 15 minutes.

## 5. Interpretation

With the parameterisations above:

- CRRA with γ*γ* around 1.5 makes him dislike the higher variance in WS*WS*.
- Lower mean WS*WS* and much larger DS*DS* make EU(stay)*EU*(stay) substantially lower than EU(resign)*EU*(resign) for most draws from the priors.

That gives you a clean, explicit Bayesian assumption set for your case write‑up, and you can tweak the distribution parameters to fit whatever numerical story you want to tell.

## 6. Bayesian prior for CEO departure: Beta(12, 1.5)

The D0_ceo ARA prediction is anchored by an empirical Bayesian prior derived from ASX 100 moral-reputational crisis events (2013-2023), conditioned on the absence of a visible contrition strategy by the incumbent CEO. See `ESG-and-CEO-turnover.md` for the full derivation.

**Prior specification:** p_departure ~ Beta(12, 1.5)

| Statistic             | Value        |
| --------------------- | ------------ |
| Prior mean            | 0.889        |
| Prior mode            | 0.917        |
| Prior median          | ~0.903       |
| 90% credible interval | [0.72, 0.99] |
| Effective sample size | 13.5         |

**Bayesian updating mechanism.** The engine combines the Beta prior with ARA-computed evidence via pseudo-count addition. The ARA prediction produces N belief draws, each contributing a soft vote for resign or stay. The prior adds 12 pseudo-counts for resignation and 1.5 for survival:

  alpha_post = 12 + sum(Pr(resign|draw_i) for i in 1..N)
  beta_post  = 1.5 + sum(Pr(stay|draw_i) for i in 1..N)
  P(resign)  = alpha_post / (alpha_post + beta_post)

With N=100 ARA draws, the prior has weight 13.5/(13.5+100) = 11.9%. This is appropriate: the prior provides a meaningful anchor from empirical data while allowing the game-theoretic analysis to dominate.

**Joyce as archetype.** Joyce maps to the "no contrition" archetype with an extreme version: combative public posture throughout COVID complaints and ghost flights, record bonus while workers stood down, lobbying against competition, and zero accountability signalling. In the filtered ASX 100 data, the conditional departure rate without contrition strategy is 12/12 (100%). The 1.5 pseudo-count for survival is Jeffreys-type regularisation preventing prior degeneracy.

## 7. Board sacking response: non-pecuniary CEO turnover in stakeholder-oriented countries

### Justification from Colak, Korkeamaki & Meyer (2024)

The Board's near-certain decision to sack a non-resigning CEO is justified by international evidence from Colak et al. (2024), "ESG and CEO Turnover Around the World," *Journal of Corporate Finance* 84: 102523.

**Key findings supporting the Board's sacking imperative:**

1. **Extreme ESG risk doubles CEO turnover odds (Table 3, Col 2).** The logit coefficient for Extreme RRI (>=60) is 0.691*** (z=3.38), yielding an odds ratio of ~2.0. CEOs of firms with extreme risk exposure have a 9.4 percentage point higher probability of losing their job (24.0% vs 14.6%). The effect is robust to CEO-specific, firm-level, and country-level controls, year/industry/country fixed effects, and alternative measures of risk exposure.

2. **Non-pecuniary motives operate independently (Table 6).** Even when ESG events produce *positive* CARs (no shareholder wealth destruction), the Extreme RRI x Positive CAR coefficient is 0.635** (z=2.01). Boards replace CEOs for non-pecuniary reasons — media shaming, public pressure, and reputational concerns — even when the firm's stock price is unharmed. This is the key result: boards act on reputation, not just financial performance.

3. **Stakeholder-oriented countries show stronger effects (Table 1D).** The non-pecuniary turnover channel is more pronounced in countries with high Environmental Performance Index, Employment Laws index, and Voice & Accountability index scores. Australia ranks high on all three dimensions (Employment Laws: 0.74, Voice & Accountability: 96.6), comparable to Nordic countries and substantially above the US (0.22, 85.6).

4. **Regression kink design confirms causality (Table 4).** A sharp RKD at the Extreme RRI threshold (RRI=60) shows significant slope changes in CEO turnover probability: robust bias-corrected estimates of 0.141-0.203*** across specifications. This is causal evidence, not merely associational.

5. **CEO turnover probability reaches ~50% at extreme RRI (Fig. 2).** The adjusted prediction graph shows CEO turnover probability jumping from ~15% at moderate RRI to ~50% at extreme levels (RRI 66-75). For the most extreme cases (RRI 76-100), the rate drops slightly due to very small sample sizes (n=2 observations above RRI 80).

### Application to Qantas (Joyce, September 2023)

The Qantas case sits at the extreme end of every dimension Colak et al. study:

- **RRI equivalent well above 60:** Concurrent ACCC lawsuit (filed July 2023), Senate inquiry into Qatar competition block, sustained front-page media coverage across all major Australian outlets for months. Multiple ESG dimensions (S: customer harm from ghost flights, G: governance failures, E: competition blocking).
- **Stakeholder-oriented governance environment:** Australia's high E/S/G norms create intense non-pecuniary pressure on boards. The post-Royal Commission "accountability norm" (see ESG-and-CEO-turnover.md, Section 4) amplifies the Colak et al. effect.
- **No contrition strategy:** Joyce's combative posture eliminates the only survival mechanism identified in Australian crisis data (see Section 6 above).
- **Pecuniary AND non-pecuniary channels active:** Unlike the "positive CAR" subsample in Colak et al., Qantas suffered both reputational damage and financial costs (ACCC penalties, customer compensation). Both channels independently predict CEO removal.

### Board utility calibration

These findings justify the following Board utility parameters:

- **ceo_loss_cost = -1.5** (previously 1.5): In the ACCC crisis context, retaining a tainted CEO is more costly than removing them. The Board actively benefits from CEO removal through: calmed regulators, restored investor confidence, credible "new management" narrative, reduced legal exposure. The sign flip reflects Colak et al.'s finding that boards in stakeholder-oriented countries face non-pecuniary pressure to act — the "cost of losing the CEO" becomes a "reward for decisive accountability."

- **implementation_cost_sack = 0.3** (previously 1.0): Stakeholder support for CEO removal dramatically reduces transition friction. When shareholders, regulators, media, and politicians are all demanding the CEO's departure, the Board faces minimal internal resistance. The governance review provides institutional cover for the decision.

### CEO full clawback calibration

- **W_stay_sacked = 0.5** (previously 1.5): A forced sacking after sustained defiance triggers full clawback. The Board demonstrated willingness to claw back A$9.26M even in the voluntary departure scenario (which was framed as cooperative). A forced removal — after the CEO refused to resign, endured a hostile AGM, and faced adverse governance review findings — would trigger near-total forfeiture: all LTIs cancelled, all STIs clawed back, legal costs mounting, and reputational destruction of future earning capacity. The 0.5 residual represents only base salary for months worked.