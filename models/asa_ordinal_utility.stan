// Ordinal probit model for ASA utility weight estimation.
//
// Observations: Likert scores y[n] in {1,...,5} for each (scenario, action, draw)
// triple.  Multiple draws per (scenario, action) pair are repeated measurements
// of the same latent utility, modelled with a scenario-level random intercept.
//
// Latent utility (original scale):
//   mu[s] = phi[s] . w + anchored[s]
// where s indexes unique (scenario, action) pairs.
//
// Unlike the Board model, ASA has no vote penalty terms and no ordering
// constraints between weights.  All K=8 weights are independently positive.
// Each weight corresponds to a measurable game tree outcome or input.
//
// Parameters split into CONTEXT (fire equally for both actions, capture
// situation quality for Likert level fitting) and INTERACTION (action-varying,
// drive the strike/no-strike decision).  Context terms cancel in delta-EU
// so only interaction terms affect action probabilities.
//
// The probit link has its sigmoid transition over ~6 units.  Raw mu can span
// a wide range, placing observations in saturated Phi tails with zero gradient.
// We normalise:
//   eta_scaled = (mu[s] + RE) / mu_scale
// where mu_scale is pre-computed from the data so eta_scaled ~ [-3, 3].
// Cutpoints and sigma_scenario live on the normalised scale.
// Weights w retain their original (utility) scale.
//
// Weight parameterization (lognormal priors, directly interpretable):
//   All weights w[k] > 0 are given lognormal priors centred at spec defaults.
//   Prior locations and scales are passed as data for flexibility.
//
// Weight order (ESTIMABLE_PARAM_NAMES, K=8):
//   CONTEXT (same phi for both actions):
//     1: w_ctx_inaction          (Board passivity penalty)
//     2: w_ctx_departure         (CEO accountability credit)
//     3: w_ctx_review            (Governance review credit)
//   INTERACTION (fire only for rec_strike):
//     4: w_strike_cost           (Net mobilisation cost of striking)
//     5: w_strike_vs_passive     (Value of striking against passive board)
//     6: w_departure_dampens     (CEO departure reduces strike value)
//     7: w_sack_dampens          (Board-forced exit further reduces strike)
//     8: w_credibility_signal    (Repeat-game credibility value of striking)

data {
  int<lower=1> N;                          // total observations
  int<lower=1> S;                          // unique (scenario, action) pairs
  int<lower=1> K;                          // number of weight parameters (7)
  array[N] int<lower=1, upper=5> y;        // observed Likert scores
  array[N] int<lower=1, upper=S> sa_id;    // (scenario,action) pair index per obs
  matrix[S, K] phi;                        // basis function values
  vector[S] anchored;                      // fixed utility contributions

  int<lower=1> N_scenarios;                // number of unique scenarios
  array[S] int<lower=1, upper=N_scenarios> scenario_id;  // scenario index per (s,a) pair

  // Normalisation scale (pre-computed from init mu range)
  real<lower=0> mu_scale;                  // divides eta so probit sees ~[-3, 3]

  // Prior hyperparameters for weights (passed as data for flexibility)
  vector[K] prior_log_mean;               // log(spec_default) for each weight
  vector<lower=0>[K] prior_log_sd;        // prior SD on log scale
}

parameters {
  // Positive weights with upper bound to prevent exp() overflow during warmup
  vector<lower=0, upper=500>[K] w;

  // Cutpoints reparameterised to avoid degeneracy / overflow
  real cutpoint_base_raw;                  // unconstrained base location
  vector[3] cutpoint_gap_raw;              // unconstrained gaps (positive after transform)
  vector[N_scenarios] z_scenario;          // non-centered scenario RE
  real<lower=0> sigma_scenario;            // scenario RE SD (normalised scale)
}

transformed parameters {
  vector[S] mu;                            // latent utility per (scenario, action)
  ordered[4] cutpoints;                    // ordinal probit cutpoints (normalised scale)

  // Latent utility: linear basis + anchored
  mu = phi * w + anchored;

  // Robust cutpoint construction with bounded location and minimum gap.
  {
    real base = 3 * tanh(cutpoint_base_raw);          // keeps location in [-3, 3]
    cutpoints[1] = base;
    for (g in 1:3) {
      real gap = 0.25 + 2.0 * inv_logit(cutpoint_gap_raw[g]); // gap in (0.25, 2.25)
      cutpoints[g + 1] = cutpoints[g] + gap;
    }
  }
}

model {
  // ── Priors: lognormal centred at spec defaults ──
  for (k in 1:K) {
    w[k] ~ lognormal(prior_log_mean[k], prior_log_sd[k]);
  }

  // Cutpoint priors (robust, on unconstrained base/gaps)
  cutpoint_base_raw ~ normal(0, 1.5);
  cutpoint_gap_raw ~ normal(0, 1);

  // Scenario random effects (non-centered parameterization, normalised scale)
  sigma_scenario ~ student_t(4, 0, 1);
  z_scenario ~ std_normal();

  // ── Likelihood: ordinal probit with normalised eta ──
  for (n in 1:N) {
    int sa = sa_id[n];
    int sc = scenario_id[sa];
    real eta = (mu[sa] + sigma_scenario * z_scenario[sc]) / mu_scale;
    // Clamp to prevent NaN from extreme proposals during warmup
    eta = fmin(fmax(eta, -20), 20);
    y[n] ~ ordered_probit(eta, cutpoints);
  }
}

generated quantities {
  // Posterior predictive checks
  array[N] int<lower=1, upper=5> y_rep;
  for (n in 1:N) {
    int sa = sa_id[n];
    int sc = scenario_id[sa];
    real eta = (mu[sa] + sigma_scenario * z_scenario[sc]) / mu_scale;
    eta = fmin(fmax(eta, -20), 20);
    y_rep[n] = ordered_probit_rng(eta, cutpoints);
  }
}
