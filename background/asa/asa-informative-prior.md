# Informative Prior: P(ASA Recommends Against | Headline Incident = 1, Board Action)

## Observed Data


| Board Action            | k / n     | Raw Rate  |
| ----------------------- | --------- | --------- |
| 0 Do nothing            | 9 / 10    | 0.900     |
| 1 Review or CEO resigns | 3 / 3     | 1.000     |
| 2 Sack CEO              | 2 / 2     | 1.000     |
| **Total**               | **14/15** | **0.933** |


## Domain Context

When a headline incident exists, the ASA's recommendation is near-automatic: they recommend a strike against the remuneration report in the vast majority of cases. Board action (review, CEO replacement) may influence the actual shareholder vote outcome, but the ASA's *recommendation* is largely invariant to board response. The 14/15 observed rate (93.3%) is consistent with this, and the single non-recommendation (1/15) likely reflects idiosyncratic circumstances rather than a systematic board-action effect.

## Prior Specification

Minimal monotonic separation with tight credible intervals, reflecting the domain conviction that all probabilities exceed 0.90 while preserving the structural requirement for distinct priors per board action level.


| Board Action            | Prior       | Mean  | Mode  | 90% CI         |
| ----------------------- | ----------- | ----- | ----- | -------------- |
| 0 Do nothing            | Beta(46, 4) | 0.920 | 0.938 | [0.850, 0.970] |
| 1 Review or CEO resigns | Beta(44, 4) | 0.917 | 0.935 | [0.846, 0.967] |
| 2 Sack CEO              | Beta(43, 4) | 0.914 | 0.933 | [0.843, 0.966] |


### Properties

- All means above 0.91, all modes above 0.93
- 90% credible intervals entirely above 0.84
- Monotonic decreasing: 0.920 > 0.917 > 0.914
- Near-identical distributions reflecting the data's inability to distinguish between board action levels
- Effective sample size of 48 to 50 per cell
- Separations of 0.003 per step are deliberately small; the data provides no evidence for larger gaps

### Construction Logic

The pooled observed rate is 14/15 = 0.933. A mildly informative Beta(1,1) prior gives a pooled posterior of approximately Beta(15,2), with mean 0.882 and mode 0.933. To achieve tighter intervals consistent with the domain conviction that all probabilities exceed 0.90, the effective sample size is scaled to approximately 48 to 50 per cell. The alpha parameters (46, 44, 43) are then set to maintain the shared mode near 0.933 while introducing minimal monotonic separation. The shared beta parameter of 4 controls the right-tail behaviour and keeps the 90% lower bound above 0.84.

### Sensitivity

The 0.003 per-step separation is a modelling choice, not a data-driven estimate. Doubling it to 0.006 (means of 0.920, 0.914, 0.908) remains fully consistent with the data and would be equally defensible. The critical constraint is that all three distributions remain concentrated above 0.90 with substantial overlap.

## Justification

The hierarchical model with monotonic decreasing constraint was rejected because:

1. **The constraint fought the data.** Observed rates were 0.90, 1.00, 1.00 (flat to increasing), but the model forced a decreasing pattern, creating artificial spread.
2. **Small samples amplified the distortion.** With n=3 and n=2, the constraint dominated the likelihood, pulling estimates well below observed values.
3. **Credible intervals failed to cover actuals.** The 90% interval for ba=2 ran approximately [0.43, 0.93], failing to cover the observed 2/2 = 1.000.
4. **The effect being estimated likely does not exist at this stage.** There is no statistical or contextual evidence that board action reduces the probability of an ASA "against" recommendation conditional on a headline incident.

## Modelling Note

The important distinction in any downstream model is between the ASA *recommendation* (near-certain given h=1, specified here) and the shareholder *vote outcome* (where board action plausibly matters). The board-action effect should be reserved for the vote stage, not imposed on the recommendation stage where the data does not support it.