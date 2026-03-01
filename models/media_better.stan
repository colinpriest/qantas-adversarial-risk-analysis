// models/media_better.stan  (STAN "Better" - LOG-SCALE OBSERVATION)
// Coverage: logistic trend with floor
// True intensity: AR(1) on log M_t
// Observation: log(y + eps) ~ Normal(logC + logM, sigmaY)  (no exp overflow)

data {
  int<lower=1> T;
  vector[T] t_scaled;

  int<lower=1> N_y;
  array[N_y] int<lower=1, upper=T> y_idx;
  vector[N_y] y_obs;              // observed intensity, must be >= 0
  real<lower=0> eps;              // small constant for log(y+eps)
}

transformed data {
  vector[N_y] logy = log(y_obs + eps);
  real logy_bar = mean(logy);
}

parameters {
  // Coverage trend
  real aC;
  real bC;

  // AR(1) on log intensity
  real mu_logM;
  real<lower=0> sigma_logM;
  real phi_raw;
  vector[T] z;

  // Observation noise on log scale
  real<lower=0> sigmaY;
}

transformed parameters {
  vector[T] C;
  vector[T] logC;
  real phi;
  vector[T] logM;

  phi = 0.98 * tanh(phi_raw);

  C = 0.02 + 0.96 * inv_logit(aC + bC * t_scaled);
  logC = log(C);

  logM[1] = mu_logM + (sigma_logM / sqrt(1 - phi * phi)) * z[1];
  for (t in 2:T) {
    logM[t] = mu_logM + phi * (logM[t-1] - mu_logM) + sigma_logM * z[t];
  }
}

model {
  // Priors
  aC ~ normal(0, 1);
  bC ~ normal(0, 1);

  // Anchor to observed log-scale
  mu_logM ~ normal(logy_bar, 0.8);      // a bit wider than before
  sigma_logM ~ normal(0, 0.35);         // half-normal
  phi_raw ~ normal(0, 0.8);
  z ~ normal(0, 1);

  sigmaY ~ normal(0, 0.8);              // log-scale noise; adjust if needed

  // Likelihood (log scale)
  for (i in 1:N_y) {
    int t = y_idx[i];
    logy[i] ~ normal(logC[t] + logM[t], sigmaY);
  }
}

generated quantities {
  vector[N_y] logy_rep;
  for (i in 1:N_y) {
    int t = y_idx[i];
    logy_rep[i] = normal_rng(logC[t] + logM[t], sigmaY);
  }
}