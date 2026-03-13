# Historical Data Tables

This document consolidates the empirical data tables underlying the prior distributions and stochastic decision models in the ARA engine. Each table cites its source file and identifies the model parameter or distribution it calibrates.

## 1. ASX CEO Departures After Moral-Reputational Crises

**Calibrates:** `Beta(12.5, 0.5)` prior on CEO departure probability at D0_ceo (engine/solver.py)

**Construction:** Jeffreys prior `Beta(0.5, 0.5)` updated with 12 Australian observations of no-remorse CEOs facing moral-reputational crises — all 12 departed. Posterior: `Beta(0.5 + 12, 0.5 + 0) = Beta(12.5, 0.5)`, mean = 0.962, mode = 0.958.

**Source:** `background/ceo/ESG-and-CEO-turnover.md`, `background/ceo/Australian-ESG-CEO-turnover.md`

### 1.1 Full ASX 100 Crisis Dataset (n = 26)

| # | Company | CEO | Crisis Event | Crisis Date | Status | Departure Date |
|---|---------|-----|-------------|-------------|--------|----------------|
| 1 | CBA | Ian Narev | AUSTRAC AML scandal | Aug 2017 | Departed | Apr 2018 |
| 2 | AMP | Craig Meller | Royal Commission (fees for no service) | Apr 2018 | Departed | Apr 2018 |
| 3 | NAB | Andrew Thorburn | Royal Commission (culture/remuneration) | Feb 2019 | Departed | Feb 2019 |
| 4 | Westpac | Brian Hartzer | AUSTRAC AML/child exploitation scandal | Nov 2019 | Departed | Dec 2019 |
| 5 | Rio Tinto | JS Jacques | Juukan Gorge heritage destruction | May 2020 | Departed | Jan 2021 |
| 6 | Crown Resorts | Ken Barton | Bergin Inquiry (money laundering) | Feb 2021 | Departed | Feb 2021 |
| 7 | Star Entertainment | Matt Bekier | Bell Inquiry (CUP card/junket scandal) | Mar 2022 | Departed | Mar 2022 |
| 8 | AMP | F. De Ferrari | Misconduct allegations/cultural crisis | Apr 2021 | Departed | Dec 2021 |
| 9 | James Hardie | Jack Truong | Workplace behaviour/misconduct | Jan 2022 | Departed | Jan 2022 |
| 10 | QBE Insurance | Pat Regan | Workplace communication breach | Sep 2020 | Departed | Sep 2020 |
| 11 | Tabcorp | Elmer F. Kupper | Regulatory investigation (Cambodia) | Mar 2016 | Departed | Mar 2016 |
| 12 | IOOF | Chris Kelaher | APRA legal action/Royal Commission | Dec 2018 | Departed | Apr 2019 |
| 13 | Bapcor | Darryl Abotomey | Board governance clash | Nov 2021 | Departed | Dec 2021 |
| 14 | ASX Ltd | Dominic Stevens | CHESS replacement project failure | Nov 2022 | Departed | Aug 2022 |
| 15 | Sigma Health | Mark Hooper | Major contract loss/underperformance | FY 2017 | Departed | FY 2017 |
| 16 | Bellamy's | Laura McBain | Chinese inventory/revenue collapse | Jan 2017 | Departed | Jan 2017 |
| 17 | Lendlease | Steve McCann | Systemic operational underperformance | May 2021 | Departed | May 2021 |
| 18 | Telstra | Andy Penn | Strategy stagnation/board pressure | May 2022 | Departed | Sep 2022 |
| 19 | Treasury Wine | Michael Clarke | China tariff crisis/inventory glut | Jan 2020 | Departed | Jun 2020 |
| 20 | BHP | Andrew Mackenzie | Samarco dam collapse (Brazil) | Nov 2015 | Stayed | Left Jan 2020 |
| 21 | ANZ | Shayne Elliott | Royal Commission fallout | Feb 2019 | Stayed | N/A |
| 22 | Woolworths | Brad Banducci | Systemic wage theft/underpayment | Oct 2019 | Stayed | N/A |
| 23 | Domino's | Don Meij | Franchisee wage underpayment scandal | Aug 2017 | Stayed | N/A |
| 24 | Medibank | David Koczkar | Massive customer data breach | Oct 2022 | Stayed | N/A |
| 25 | Link Group | Vivek Bhatia | UK FCA warning/Woodford redress | Sep 2022 | Stayed | N/A |
| 26 | Origin Energy | Frank Calabria | Takeover turmoil/transition strategy | 2022/23 | Stayed | N/A |

Overall departure rate: 19/26 = 73.1%.

### 1.2 Contrition Strategy Partition (n = 16, moral-reputational subset)

The prior is conditioned on the absence of a visible contrition strategy (voluntary incentive forfeiture + unreserved public apology + explicit personal ownership of remediation). This perfectly separates survivors from departures in the Australian sample:

|  | No Contrition | Contrition |
|--|---------------|------------|
| **Departed** | 12 | 0 |
| **Stayed** | 0 | 4 |

**No-contrition departures (12):** Narev, Meller, Thorburn, Hartzer, Jacques, Barton, Bekier, De Ferrari, Truong, Regan, Kupper, Kelaher.

**Contrition survivors (4):** Elliott (ANZ), Banducci (Woolworths), Meij (Domino's), Mackenzie (BHP).

The conditional departure rate given no contrition strategy is 12/12 = 100%. The Beta(12.5, 0.5) prior applies specifically to the no-contrition archetype, which matches Joyce's combative public posture throughout the Qantas crisis.

**Academic basis:** Colak, G., Korkeamaki, T. & Meyer, N. (2023). ESG and CEO turnover around the world. *Journal of Corporate Finance*, 84, 102523.

---

## 2. Qantas AGM Remuneration Vote History

**Calibrates:** Stan state-space belief model (`models/belief_model.stan`); vote escalation pattern informing `B_mkt` posterior draws in belief checkpoints.

**Source:** `data/agm-votes.csv`

| Year-Month | Vote Against Remuneration (%) | Vote Against Chairman (%) | Context |
|------------|-------------------------------|---------------------------|---------|
| 2020-10 | 8.93 | — | Post-COVID; moderate dissent |
| 2021-11 | 9.99 | — | Stable; no headline incident |
| 2022-11 | 9.38 | 2.04 | Stable; pre-crisis |
| **2023-11** | **82.93** | — | Crisis AGM: ACCC ghost flights, Senate inquiry, CEO resignation |

The 73.6 percentage point escalation from 2022 to 2023 (9.38% to 82.93%) is the largest year-on-year increase in the panel and one of the largest in ASX history. The 82.93% vote exceeded the 50% "overwhelming" threshold, constituting both a first strike and an overwhelming rejection.

---

## 3. Cross-Company ASX Voting Panel (n = 36)

**Calibrates:** VoteModel governance effects (`engine/chance_models.py`), ASA mobilisation estimation, structural floor calibration, and `gamma_D` posterior in the Stan belief model.

**Source:** `data/extended_voting_recommendations.csv`

### 3.1 Full Panel

| Company | ASX Code | Year | Month | Rem Against (%) | First Strike | ASA Against | Proxy Adv Against | Headline | Board Action | Voting Diff (pp) |
|---------|----------|------|-------|-----------------|--------------|-------------|-------------------|----------|--------------|------------------|
| Rio Tinto | RIO | 2021 | 5 | 61.0 | 1 | 1 | 1 | 1 | D1 | +49.0 |
| AGL Energy | AGL | 2021 | 9 | 31.0 | 1 | 1 | 1 | 1 | D0 | +16.0 |
| QBE Insurance | QBE | 2021 | 5 | 44.0 | 1 | 1 | 1 | 0 | D1 | +36.0 |
| Scentre Group | SCG | 2021 | 4 | 51.0 | 1 | 1 | 1 | 0 | D1 | +41.0 |
| Westpac | WBC | 2021 | 12 | 30.0 | 1 | 1 | 1 | 1 | D1 | +12.0 |
| IAG | IAG | 2021 | 10 | 57.3 | 1 | 1 | 1 | 1 | D1 | +35.3 |
| Dexus | DXS | 2021 | 10 | 66.0 | 1 | 1 | 1 | 0 | D0 | +61.0 |
| Link Administration | LNK | 2021 | 11 | 63.0 | 1 | 1 | 1 | 0 | D0 | +49.0 |
| Platinum Asset Mgmt | PTM | 2021 | 11 | 50.0 | 1 | 1 | 1 | 0 | D0 | +38.0 |
| Argo Investments | ARG | 2021 | 10 | 17.0 | 0 | 1 | 0 | 0 | D0 | 0.0 |
| Downer EDI | DOW | 2022 | 11 | 55.8 | 1 | 1 | 1 | 1 | D0 | +43.8 |
| Blackmores | BKL | 2022 | 10 | 43.4 | 1 | 1 | 1 | 0 | D0 | +28.4 |
| Newcrest Mining | NCM | 2022 | 11 | 37.0 | 1 | 1 | 1 | 0 | D0 | +28.0 |
| ASX Ltd | ASX | 2022 | 10 | 31.0 | 1 | 1 | 1 | 1 | D1 | +26.0 |
| Goodman Group | GMG | 2022 | 11 | 28.9 | 1 | 1 | 1 | 0 | D0 | +7.9 |
| Santos | STO | 2022 | 5 | 25.3 | 1 | 0 | 1 | 0 | D1 | +17.3 |
| Corporate Travel | CTD | 2022 | 10 | 33.0 | 1 | 1 | 1 | 0 | D0 | +22.0 |
| BHP Group | BHP | 2022 | 11 | 3.0 | 0 | 0 | 0 | 0 | D0 | +1.0 |
| Rio Tinto | RIO | 2022 | 5 | 15.7 | 0 | 0 | 0 | 0 | D0 | -45.3 |
| Westpac | WBC | 2022 | 12 | 12.0 | 0 | 0 | 0 | 0 | D1 | -18.0 |
| **Qantas** | **QAN** | **2023** | **11** | **82.9** | **1** | **1** | **1** | **1** | **D1** | **+75.9** |
| Harvey Norman | HVN | 2023 | 11 | 81.8 | 1 | 1 | 1 | 1 | D0 | +63.8 |
| Fortescue | FMG | 2023 | 11 | 52.0 | 1 | 0 | 1 | 1 | D0 | +37.0 |
| Treasury Wine | TWE | 2023 | 10 | 46.0 | 1 | 1 | 1 | 0 | D0 | +34.0 |
| Tabcorp | TAH | 2023 | 10 | 34.0 | 1 | 1 | 1 | 0 | D0 | +25.0 |
| Woolworths | WOW | 2023 | 10 | 28.0 | 1 | 1 | 1 | 1 | D0 | +23.5 |
| Elders | ELD | 2023 | 12 | 62.7 | 1 | 1 | 1 | 1 | D0 | +48.7 |
| Mineral Resources | MIN | 2023 | 11 | 74.6 | 1 | 1 | 1 | 1 | D0 | +56.6 |
| Computershare | CPU | 2023 | 11 | 27.5 | 1 | 1 | 1 | 0 | D0 | +23.5 |
| Perpetual | PPT | 2023 | 10 | 88.1 | 1 | 1 | 1 | 1 | D0 | +69.2 |
| Healius | HLS | 2023 | 11 | 55.0 | 1 | 1 | 1 | 0 | D0 | +43.0 |
| Sandfire Resources | SFR | 2023 | 11 | 56.1 | 1 | 1 | 1 | 0 | D0 | +41.1 |
| BrainChip | BRN | 2023 | 5 | 52.0 | 1 | 1 | 1 | 0 | D0 | +34.0 |
| Atlas Arteria | ALX | 2023 | 5 | 51.0 | 1 | 1 | 1 | 1 | D0 | +37.0 |
| Macquarie Group | MQG | 2023 | 7 | 25.4 | 1 | 1 | 1 | 1 | D0 | +14.4 |
| NRW Holdings | NWH | 2023 | 11 | 49.8 | 1 | 1 | 1 | 1 | D0 | +18.8 |

**Board action codes:** D0 = no action, D1 = review/governance reform, D3 = CEO transition.

### 3.2 Board Action Effects on Voting Dissent

| Board Action | Avg Voting Diff (pp) | n | Interpretation |
|--------------|---------------------|---|----------------|
| D0 — No action | +32.8 | 25 | Baseline escalation when board is passive |
| D1 — Review/governance reform | +14.9 | 9 | Most effective mitigation — dampens protest growth |
| D3 — CEO exit | +46.7 | 2 | Signals crisis severity; sacking itself amplifies protest |

The non-monotonic pattern (D1 mitigates, D3 escalates) is embedded in the `VoteModel._governance_effect()` method. D1 (review) has a positive governance effect that reduces B_agm and therefore vote percentages. D3 (CEO exit) has an ambiguous effect modelled as U(-1, 0.5) — amplification is ~3x more likely than mitigation.

---

## 4. ASA Headline Governance Incidents (n = 15)

**Calibrates:** Beta priors on P(ASA recommends strike | headline incident, board action) at the A2 node; ASA A2 calibration in `asa_utility_quantification.py`.

**Source:** `background/asa/voting-patterns.md`, `background/asa/asa-informative-prior.md`

The 14 eligible observations (excluding Qantas, the modelled entity) with `headline_incident = 1`:

| Company | Year | Board Action | ASA Against | Crisis Type |
|---------|------|-------------|-------------|-------------|
| Rio Tinto | 2021 | D3 (CEO exit) | Yes | Juukan Gorge cultural heritage destruction |
| AGL Energy | 2021 | D1 (review) | Yes | Climate/emissions governance |
| Westpac | 2021 | D1 (review) | Yes | AUSTRAC money-laundering scandal |
| IAG | 2021 | D1 (review) | Yes | Insurance pricing misconduct |
| Downer EDI | 2022 | D0 (no action) | Yes | Governance concerns |
| ASX Ltd | 2022 | D3 (CEO exit) | Yes | Technology governance failure (CHESS) |
| Harvey Norman | 2023 | D0 (no action) | Yes | Wage theft scandal |
| **Fortescue** | **2023** | **D0 (no action)** | **No** | **Headline incident; only non-recommendation** |
| Woolworths | 2023 | D0 (no action) | Yes | Cost-of-living pricing scandal |
| Elders | 2023 | D0 (no action) | Yes | Governance concerns |
| Mineral Resources | 2023 | D0 (no action) | Yes | Governance concerns |
| Perpetual | 2023 | D0 (no action) | Yes | Governance concerns |
| Atlas Arteria | 2023 | D0 (no action) | Yes | Governance concerns |
| Macquarie Group | 2023 | D0 (no action) | Yes | Governance concerns |

**Qantas (2023, D1, ASA against = Yes) excluded from all counts** — the modelled entity cannot inform its own prior.

### 4.1 Counts by Board Action

| Board Action | n | k (ASA against) | Rate |
|-------------|---|-----------------|------|
| D0 — No action | 9 | 8 | 88.9% |
| D1 — Review/governance reform | 3 | 3 | 100.0% |
| D3 — CEO exit | 2 | 2 | 100.0% |
| **Total** | **14** | **13** | **93.3%** |

Fortescue is the only headline-incident case where ASA did not recommend against (D0 bucket). The single non-recommendation likely reflects idiosyncratic circumstances rather than a systematic board-action effect.

### 4.2 Derived Beta Priors

Using a Jeffreys prior `Beta(0.5, 0.5)` and minimal monotonic separation:

| Board Action | Prior | Mean | Mode | 90% CI |
|-------------|-------|------|------|--------|
| D0 — No action | Beta(46, 4) | 0.920 | 0.938 | [0.850, 0.970] |
| D1 — Review | Beta(44, 4) | 0.917 | 0.935 | [0.846, 0.967] |
| D3 — CEO exit | Beta(43, 4) | 0.914 | 0.933 | [0.843, 0.966] |

The separations of 0.003 per step are deliberately small — the data provides no evidence for larger gaps. The critical constraint is that all three distributions remain concentrated above 0.90.

---

## 5. First-Strike and Second-Strike Rates Across the ASX

**Calibrates:** Board utility function (spill risk component), two-strikes rule modelling in `engine/utilities.py`.

**Source:** `background/ceo/2nd-strike.md`

### 5.1 First Strikes by Year

| Year | ASX 200 First Strikes | ASX 300 First Strikes | Avg Against Vote in Strike Cases | Context |
|------|----------------------|----------------------|----------------------------------|---------|
| 2020 | ~22 | ~40 | ~34% | Post-COVID; JobKeeper sensitivity |
| 2021 | ~28 | ~35 | ~35% | "War for talent" concerns |
| 2022 | 22 | ~35 | 34.2% | Return to normalcy; retention grant disputes |
| 2023 | 23 | **41** (record) | **45.7%** | Cost-of-living backlash; record severity |
| 2024 | Rising | Rising | N/A | Continued pressure |

The 2023 season recorded the highest number of strikes since the two-strikes rule was introduced in 2011, with average severity (45.7%) also at a record high.

### 5.2 Second Strikes (2020-2024)

| Year | Second Strikes (ASX 200) | Second Strikes (ASX 300) | Spill Resolution Passed? |
|------|-------------------------|-------------------------|--------------------------|
| 2020 | ~2-3 (est.) | ~5 (est.) | No |
| 2021 | ~2 (est.) | ~5 (est.) | No |
| 2022 | 4 (Goodman, Lovisa, Dicker Data, Link Administration) | ~5 | No — all defeated |
| 2023 | **0** | 5 (all outside ASX 200) | No — Lake Resources spill defeated ~70% against |
| 2024 | 6 (incl. Dicker Data 4th, Lovisa 4th consecutive) | 13 | No — spill resolutions averaged <7% support |

### 5.3 Second Strike Cases with Board Responses (2020-2023)

| Company | Year | Prior Year Against (%) | Latest Against (%) | Corrective Action | Root Cause |
|---------|------|----------------------|--------------------|--------------------|------------|
| Cromwell Property | 2020 | >25.0 | >25.0 | No (defensive) | Board disputes + strategy dissent |
| Crown Resorts | 2021 | >25.0 | >25.0 | Yes (board renewal) | Governance failures + regulatory breaches |
| Westpac | 2021 | >25.0 | >25.0 | Yes (risk overhaul) | Regulatory lawsuits + fraud allegations |
| Goodman Group | 2022 | >25.0 | >25.0 | Yes (10% LTI cut) | LTI quantum misalignment |
| Lovisa | 2022 | >25.0 | 33.0 | No (resistant) | CEO sign-on bonuses + insufficient hurdles |
| Dicker Data | 2022 | >25.0 | 33.0 | No (resistant) | Pay-performance misalignment |
| Link Administration | 2022 | >25.0 | 31.0 | Yes (improved disclosure) | Poor STI/LTI transparency |
| Lake Resources | 2023 | 34.8 | 50.0 | Yes (malus + clawback) | STI transparency + share price collapse |
| IDP Education | 2023 | >25.0 | >25.0 | No (insufficient) | High incentives despite weak share price |
| Reliance Worldwide | 2023 | >25.0 | 38.9 | No (resistant) | Pay vs performance alignment |
| Australian Clinical Labs | 2023 | >25.0 | >25.0 | Yes (cybersecurity uplift) | Safety governance + executive pay post-breach |

### 5.4 Board Spill Resolution Historical Summary

Since the two-strikes rule was introduced in July 2011:

| Statistic | Value | Source |
|-----------|-------|--------|
| Companies receiving two strikes (2011-2013) | 40 | Guerdon Associates |
| Spill resolutions passed (2011-2013) | 6 (~15% pass rate) | Guerdon Associates |
| Incumbent directors removed at spill meetings | **0** (since 2011) | Guerdon Associates |
| Avg spill resolution support (2024) | <7% | Glass Lewis |
| Avg spill resolution support (2025) | 4.5% (range: 1.4% ANZ to 13.4% ACL) | Georgeson |
| First Big Four bank second strike since Westpac 2019 | ANZ (Dec 2025): 97.73% voted against spill | The Nightly |

**Key finding:** No incumbent director has ever lost their board seat at a spill meeting since the rule was introduced. The two-strikes mechanism functions primarily as a communication tool and pressure valve, not as a genuine removal threat. Institutional investors deliberately restrain second-strike voting to avoid destabilising consequences.

---

## 6. External Governance Reviews — Market Impact (2014-2023)

**Calibrates:** ReviewModel CAR hierarchy (`engine/chance_models.py`): $\mu_f \sim t(4, -0.05, 0.03)$, $\sigma_f \sim \text{Half-Normal}(0.10)$, $\text{CAR} \sim t(3, \mu_f, \sigma_f)$.

**Source:** `background/board/governance-review-case-studies.md`

### 6.1 Event Study Results

| Company | Period | Announcement AR | Reviewer Revealed AR | Findings Released AR | Total CAR |
|---------|--------|-----------------|---------------------|---------------------|-----------|
| CBA | 2017-18 | -0.97% | +0.15% | **+1.75%** | +0.93% |
| Westpac | 2019-20 | -4.05% | -0.25% | **-3.00%** | -7.30% |
| Rio Tinto | 2020 | -1.95% | +0.55% | **-2.65%** | -4.05% |
| Star Entertainment | 2021-22 | -3.60% | -1.60% | **-13.95%** | -19.15% |
| BOQ | 2022-23 | — | — | **-5.70%** | N/A |
| Qantas | 2023-24 | -0.65% | 0.00% | **+0.85%** | +0.20% |

### 6.2 Findings Window Summary Statistics

| Statistic | Value |
|-----------|-------|
| n | 6 |
| Mean | -3.78% |
| Median | -2.83% |
| Min | -13.95% (Star) |
| Max | +1.75% (CBA) |
| Range | 15.70 pp |

The extreme heterogeneity (CBA relief rally vs. Star existential threat) motivates the heavy-tailed t(3) observation distribution. The -5% location parameter for $\mu_f$ reflects the central tendency, while the t(3) tails accommodate -14% outliers that a normal distribution would assign near-zero probability.

### 6.3 Broader ASX 100 Review Outcomes (2013-2023)

**Source:** `background/board/external-review-usa-vs-australia.md`

| Company | Review Period | Review Mechanism | Classification | Outcome |
|---------|--------------|-----------------|----------------|---------|
| CBA | 2017-18 | APRA Prudential Inquiry | Regulatory | Negative |
| Westpac | 2020-24 | Promontory Independent Review (CORE) | Regulatory | Negative (initial) |
| ANZ | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |
| NAB | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |
| Macquarie | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |
| AMP | 2019-21 | APRA/ASIC Compliance & Governance Review | Regulatory | Negative |
| Rio Tinto | 2020 | Board Review of Cultural Heritage | Non-Regulatory | Negative |
| Crown Resorts | 2020-21 | Bergin Inquiry (NSW Casino Control Act) | Regulatory | Negative |
| Star Entertainment | 2021-22 | Bell Inquiry / Gotterson Inquiry | Regulatory | Negative |
| ASX Ltd | 2018, 2023 | Technology Governance & CHESS Inquiries | Regulatory | Negative |
| BOQ | 2020 | Independent Board Performance Review | Non-Regulatory | Positive |
| BOQ | 2023 | Remediation EU (APRA & AUSTRAC) | Regulatory | Negative |
| Suncorp | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |
| IAG | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |
| QBE | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |
| Medibank | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |
| Mineral Resources | 2022 | Governance Framework Gap Analysis | Non-Regulatory | Neutral |
| Telstra | 2018-23 | Compliance & Sales Practice Review | Regulatory | Negative |
| Woolworths | 2020 | Underpayment Governance Review | Regulatory | Negative |
| Tabcorp | 2021-22 | Regulatory Compliance Review | Regulatory | Negative |
| Bendigo Bank | 2018-19 | APRA Governance Self-Assessment | Regulatory | Negative |

**Outcome distribution (regulatory reviews):** 19/20 = 95% negative. Positive outcomes are effectively absent in regulatory reviews and rare even in non-regulatory reviews.

---

## 7. Review Outcome Rating — Dirichlet Calibration

**Calibrates:** `Dirichlet(38, 160, 1)` prior on (negative, balanced, positive) outcome probabilities at the R chance node.

**Source:** `background/board/external-review-distributions.md`

### 7.1 Base Rate (Board-Initiated Reviews, All Contexts)

| Finding Category | Base Rate |
|-----------------|-----------|
| Positive | ~88% |
| Balanced/Neutral | ~7-10% |
| Negative | ~3-5% |

### 7.2 Posterior (Qantas September 2023 Context)

Updated using: (a) history of labour disputes and customer service failures, (b) 30% share price decline since March 2023, (c) ACCC ghost flight filing two weeks prior.

| Finding Category | Posterior | Dirichlet Pseudo-Count | Reasoning |
|-----------------|-----------|----------------------|-----------|
| Balanced/Neutral | 75-85% (E = 0.804) | 160 | "Mistakes were made" is dominant rational strategy — admits accountability without conceding liability |
| Negative | 15-20% (E = 0.191) | 38 | Updated from 3-5% due to ACCC severity; limited by board's control of review scope |
| Positive | <1% (E = 0.005) | 1 | Negligible — clean bill of health during active litigation would be non-credible ("gaslighting") |

**Actual outcome:** Balanced. The Saar review (August 2024) reported "mistakes were made" without finding "deliberate wrongdoing" — matching the modal prediction. The Board used the balanced finding to justify a $9.26 million clawback of the former CEO's payout.

---

## 8. Review Direct Costs — Gamma Calibration

**Calibrates:** `Gamma(4.55, 4741)` prior on direct review costs in `ReviewDirectCostModel`.

**Source:** `background/board/direct-costs-governance-review.md`

### 8.1 Cost Component Estimates (AUD, for M = AUD 10B market cap)

| Component | Low | Central | High | Confidence |
|-----------|-----|---------|------|------------|
| A: Reviewer fees | $2.0M (-0.20 bps) | $3.0M (-0.30 bps) | $5.0M (-0.50 bps) | High |
| B: Management distraction | $1.5M (-0.15 bps) | $4.0M (-0.40 bps) | $10.0M (-1.00 bps) | Medium |
| C: Internal resources | $1.2M (-0.12 bps) | $2.6M (-0.26 bps) | $4.5M (-0.45 bps) | Medium-High |
| **Total** | **$4.7M (-0.47 bps)** | **$9.6M (-0.96 bps)** | **$19.5M (-1.95 bps)** | |

### 8.2 Gamma Distribution Properties

| Property | Value |
|----------|-------|
| Shape (alpha) | 4.55 |
| Rate (beta) | 4741 |
| Mean | 0.00096 (9.6 bps) |
| SD | 0.00045 (4.5 bps) |
| Mode | 0.00075 (7.5 bps) |
| 5th percentile | 0.00031 (3.1 bps) |
| 95th percentile | 0.00185 (18.5 bps) |
| Skewness | 0.94 |

Management distraction dominates both the central estimate (~40%) and the uncertainty. Direct costs (~10 bps) are small relative to the CAR from findings release (E = -500 bps) and the market cap loss motivating the review (~3,000 bps for Qantas).
