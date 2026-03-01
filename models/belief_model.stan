data {
  int<lower=1> T;
  vector[T] shock;          // z-scored
  vector[T] abret;          // z-scored

  // scaling constants for bookkeeping (not used)
  real shock_mean;
  real shock_sd;
  real abret_mean;
  real abret_sd;

  int<lower=0> N_rem;
  array[N_rem] int<lower=1, upper=T> rem_idx;
  vector<lower=0, upper=1>[N_rem] rem_obs;

  int<lower=0> N_chair;
  array[N_chair] int<lower=1, upper=T> chair_idx;
  vector<lower=0, upper=1>[N_chair] chair_obs;
}

parameters {
  // State dynamics
  real rho_raw;                     // mapped to (-0.95, 0.95)
  real beta;
  real<lower=1e-6> sigma_B;
  real<lower=1e-6> sigma_B0;
  vector[T] zB;

  // Market observation (z-scale): only slope is learned
  real lambda_r;

  // Votes (logit-normal)
  real alpha_rem;
  real<lower=1e-3> sigma_rem;

  real alpha_chair;
  real lambda_chair;
  real<lower=1e-3> sigma_chair;
}

transformed parameters {
  real rho;
  vector[T] B;

  rho = 0.95 * tanh(rho_raw);

  // Belief state evolution (non-centered innovations zB)
  B[1] = sigma_B0 * zB[1] + beta * shock[1];
  for (t in 2:T) {
    B[t] = rho * B[t-1] + beta * shock[t] + sigma_B * zB[t];
  }
}

model {
  // ---- Priors (state) ----
  rho_raw ~ normal(0, 0.4);
  beta ~ normal(0, 0.7);

  // Scales: half-Student-t via lower bound
  sigma_B  ~ student_t(4, 0, 0.25);
  sigma_B0 ~ student_t(4, 0, 0.35);

  zB ~ normal(0, 1);

  // Market slope (abret is z-scored)
  lambda_r ~ normal(0, 0.5);

  // ---- Vote priors ----
  alpha_rem ~ normal(logit(0.10), 1.0);
  sigma_rem ~ student_t(4, 0, 0.6);

  alpha_chair ~ normal(logit(0.02), 1.5);
  lambda_chair ~ normal(0, 0.7);
  sigma_chair ~ student_t(4, 0, 0.8);

  // ---- Likelihoods ----
  abret ~ normal(lambda_r * B, 1.0);

  // Anchor: lambda_rem = 1, so B is in rem-vote logit units
  for (i in 1:N_rem) {
    int t = rem_idx[i];
    real y = fmin(1 - 1e-6, fmax(1e-6, rem_obs[i]));
    logit(y) ~ normal(alpha_rem + B[t], sigma_rem);
  }

  for (i in 1:N_chair) {
    int t = chair_idx[i];
    real y = fmin(1 - 1e-6, fmax(1e-6, chair_obs[i]));
    logit(y) ~ normal(alpha_chair + lambda_chair * B[t], sigma_chair);
  }
}