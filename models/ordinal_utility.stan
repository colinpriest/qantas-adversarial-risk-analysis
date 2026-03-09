// Ordinal probit model for Board utility weight estimation.
//
// Observations: Likert scores y[n] in {1,...,5} for each (scenario, action, draw)
// triple.  Multiple draws per (scenario, action) pair are repeated measurements
// of the same latent utility, modelled with a scenario-level random intercept.
//
// Latent utility (original scale):
//   mu[s] = phi[s] . w + anchored[s]
//           - has_strike[s] * w_strike * vote_x_strike[s]
//           - has_overwh[s] * w_overwh * vote_x_overwh[s]
// where s indexes unique (scenario, action) pairs.
//
// The probit link has its sigmoid transition over ~6 units.  Raw mu can span
// 20+ units (dominated by vote penalties), placing most observations in
// saturated Phi tails with zero gradient.  We normalise:
//   eta_scaled = (mu[s] + RE) / mu_scale
// where mu_scale is pre-computed from the data so eta_scaled ~ [-3, 3].
// Cutpoints and sigma_scenario live on the normalised scale.
// Weights w retain their original (utility) scale.
//
// Weight parameterization (lognormal priors, directly interpretable):
//   All weights w[k] > 0 are declared with <lower=0> and given lognormal priors
//   centred at spec defaults: w[k] ~ lognormal(log(default_k), sigma_prior).
//   No hidden transforms — displayed posterior values ARE the utility weights.
//
// Ordering constraint: w_removal > w_remove_ceo_overwhelming > 0.
//   delta_removal ~ lognormal(...) > 0
//   w[5] = w[6] + delta_removal  (computed in transformed parameters)
//
// Vote penalty parameters (w_strike, w_overwhelming):
//   w_strike, w_overwh > 0, lognormal priors.
//   Linear penalty: -w * x where x = normalised vote excess.
//
// Weight order (ESTIMABLE_PARAM_NAMES):
//   1: w_inaction_base
//   2: w_inaction_no_review
//   3: w_inaction_delay  (reactive governance penalty)
//   4: w1  (early CEO departure cost)
//   5: w_removal  (CEO involuntary removal cost)  [derived: w[6] + delta]
//   6: w_remove_ceo_overwhelming  (CEO removal shock relief, overwhelming vote)
//   7: w15  (adverse review + CEO present penalty)

data {
  int<lower=1> N;                          // total observations
  int<lower=1> S;                          // unique (scenario, action) pairs
  int<lower=1> K;                          // number of linear weight parameters (7)
  array[N] int<lower=1, upper=5> y;        // observed Likert scores
  array[N] int<lower=1, upper=S> sa_id;    // (scenario,action) pair index per obs
  matrix[S, K] phi;                        // basis function values
  vector[S] anchored;                      // fixed utility contributions

  int<lower=1> N_scenarios;                // number of unique scenarios
  array[S] int<lower=1, upper=N_scenarios> scenario_id;  // scenario index per (s,a) pair

  // Vote penalty data (linear in vote excess)
  vector[S] vote_x_strike;                 // max(0, (V-0.25)/0.75) per (s,a) pair
  vector[S] vote_x_overwh;                 // max(0, (V-0.50)/0.50) per (s,a) pair
  array[S] int<lower=0, upper=1> has_strike;   // 1 if vote > 25%
  array[S] int<lower=0, upper=1> has_overwh;   // 1 if vote > 50%

  // Normalisation scale (pre-computed from init mu range)
  real<lower=0> mu_scale;                  // divides eta so probit sees ~[-3, 3]
}

parameters {
  // Direct positive weights — no hidden transforms.
  // Stan internally log-transforms <lower=0> parameters for HMC.
  real<lower=0> w_raw_1;                   // w_inaction_base
  real<lower=0> w_raw_2;                   // w_inaction_no_review
  real<lower=0> w_raw_3;                   // w_inaction_delay
  real<lower=0> w_raw_4;                   // w1
  real<lower=0> w_raw_6;                   // w_remove_ceo_overwhelming
  real<lower=0> delta_removal;             // w_removal - w_remove_ceo_overwhelming > 0
  real<lower=0> w_raw_7;                   // w15

  real<lower=0> w_strike;                  // vote strike penalty
  real<lower=0> w_overwh;                  // vote overwhelming penalty

  // Cutpoints reparameterised to avoid degeneracy / overflow
  real cutpoint_base_raw;                  // unconstrained base location
  vector[3] cutpoint_gap_raw;              // unconstrained gaps (positive after transform)
  vector[N_scenarios] z_scenario;          // non-centered scenario RE
  real<lower=0> sigma_scenario;            // scenario RE SD (normalised scale)
}

transformed parameters {
  vector[K] w;                             // positive linear weights (original scale)
  vector[S] mu;                            // latent utility per (scenario, action)
  ordered[4] cutpoints;                    // ordinal probit cutpoints (normalised scale)

  // Direct assignment — w values ARE the parameters, no transform needed
  w[1] = w_raw_1;                          // w_inaction_base
  w[2] = w_raw_2;                          // w_inaction_no_review
  w[3] = w_raw_3;                          // w_inaction_delay
  w[4] = w_raw_4;                          // w1
  // w_removal = w_remove_ceo_overwhelming + delta, guarantees w_removal > w_remove_ceo_overwhelming > 0
  w[6] = w_raw_6;                          // w_remove_ceo_overwhelming
  w[5] = w_raw_6 + delta_removal;          // w_removal
  w[7] = w_raw_7;                          // w15

  // Latent utility: linear basis + anchored - linear vote penalties
  mu = phi * w + anchored;
  for (s in 1:S) {
    if (has_strike[s])
      mu[s] -= w_strike * vote_x_strike[s];
    if (has_overwh[s])
      mu[s] -= w_overwh * vote_x_overwh[s];
  }

  // Robust cutpoint construction with bounded location and minimum gap.
  // This prevents exp overflow/underflow that previously produced invalid ordered vectors.
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
  // w ~ lognormal(log(default), sigma) has median = default.
  // SD = 1.0 gives ~2.7x range per SD on the ratio scale.
  w_raw_1 ~ lognormal(1.10, 1.0);         // log(3.0) ≈ 1.10   w_inaction_base, median 3.0
  w_raw_2 ~ lognormal(0.69, 1.0);         // log(2.0) ≈ 0.69   w_inaction_no_review, median 2.0
  w_raw_3 ~ lognormal(0.41, 1.0);         // log(1.5) ≈ 0.41   w_inaction_delay, median 1.5
  w_raw_4 ~ lognormal(-0.69, 1.0);        // log(0.5) ≈ -0.69  w1, median 0.5
  w_raw_6 ~ lognormal(-0.69, 1.0);        // log(0.5) ≈ -0.69  w_remove_ceo_overwhelming, median 0.5
  delta_removal ~ lognormal(0.26, 1.0);   // log(1.3) ≈ 0.26   delta (w_removal - w_rceo), median 1.3
  w_raw_7 ~ lognormal(1.61, 1.0);         // log(5.0) ≈ 1.61   w15, median 5.0

  // Vote penalty priors (lognormal, same as before but now explicit)
  w_strike ~ lognormal(0.69, 1.0);        // log(2.0) ≈ 0.69   w_strike, median 2.0
  w_overwh ~ lognormal(1.10, 1.0);        // log(3.0) ≈ 1.10   w_overwhelming, median 3.0

  // Cutpoint priors (robust, on unconstrained base/gaps)
  cutpoint_base_raw ~ normal(0, 1.5);      // keeps base near 0 after tanh scaling
  cutpoint_gap_raw ~ normal(0, 1);         // softplus via inv_logit; gap prior ~0.25–2.25

  // Scenario random effects (non-centered parameterization, normalised scale)
  sigma_scenario ~ student_t(4, 0, 1);
  z_scenario ~ std_normal();

  // ── Likelihood: ordinal probit with normalised eta ──
  for (n in 1:N) {
    int sa = sa_id[n];
    int sc = scenario_id[sa];
    real eta = (mu[sa] + sigma_scenario * z_scenario[sc]) / mu_scale;
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
    y_rep[n] = ordered_probit_rng(eta, cutpoints);
  }
}
