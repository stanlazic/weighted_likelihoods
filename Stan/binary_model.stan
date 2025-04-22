data {
  int N;
  array[N] int y;
  vector[N] x1;
  vector[N] x2;
  vector[N] w;
}

parameters {
  real b0;
  real b1;
  real b2;
}

transformed parameters {
  vector[N] eta = b0 + b1*x1 + b2*x2;
}

model {
  b0 ~ normal(0, 10);
  b1 ~ normal(0, 10);
  b2 ~ normal(0, 10);

  for (i in 1:N) {
  target += bernoulli_logit_lpmf(y[i] | eta[i]) * w[i];
  }
}

generated quantities {
  vector[N] y_pred;
  for (i in 1:N) {
    y_pred[i] = bernoulli_logit_rng(eta[i]);
  }
}
