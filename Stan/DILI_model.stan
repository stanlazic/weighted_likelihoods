/*
Shrinkage parameter (sigma) is learned from the data, but with a range
of priors for sigma
*/

data{
  int<lower=1> N;         // number of compounds
  int<lower=1> P;         // number of coefs
  array[N] int y;         // outcome
  matrix[N,P] X;          // assay and other data
  real sigma_prior;       // prior for sigma
  array[N] real w;        // weights for class imbalance

  int<lower=0,upper=1> make_pred;  // predict new compounds (1=yes, 0=no)
  int<lower=1> N_pred;             // number of new compounds
  matrix[N_pred,P] X_pred;         // data for new compounds
}

parameters{
  ordered[2] cutpoints;      // two thresholds
  vector[P] beta;            // parameters
  real<lower=0> sigma;       // tuning param for prior over coefs
  real mu;                   // mean for coefs... should be close to zero
}

transformed parameters{
  vector[N] eta;             // linear predictor
  eta = X * beta;
}

model{
  cutpoints ~ normal(0, 20); // prior for cutpoints

  // Laplace prior on coefs (similar to L1 reg)
  sigma ~ normal(0, sigma_prior);
  mu ~ normal(0, 2);
  beta ~ double_exponential(mu, sigma);

  // likelihood
  for (i in 1:N){
    target += ordered_logistic_lpmf(y[i] | eta[i], cutpoints) * w[i]; 
  }
}

generated quantities{
  vector[N] log_lik;        // Log likelihood for WAIC calc
  vector[N] train_pred;     // predicted category for training data
  vector[make_pred == 1 ? N_pred : 0] ypred;     // predict category
  vector[make_pred == 1 ? N_pred : 0] eta_pred;  // eta for new data

  for (i in 1:N){
    log_lik[i] =  w[i] * ordered_logistic_lpmf(y[i] | eta[i], cutpoints);
    train_pred[i] = ordered_logistic_rng(eta[i], cutpoints);
  }

  // new compounds
  if (make_pred == 1){
    eta_pred = X_pred * beta;
    for (j in 1:N_pred) {
      ypred[j] = ordered_logistic_rng(eta_pred[j], cutpoints);
    }
  }

}
