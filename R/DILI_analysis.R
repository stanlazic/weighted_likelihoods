library(tidyverse)
library(cmdstanr)
library(beeswarm)
library(ModelMetrics)
library(CalibrationCurves)

source("functions.R")

## read in data
d <- read.delim("../data/DILI_raw_data.txt")

## center and scale continuous variables
d2 <- data.frame(apply(dplyr::select(d, log10.cmax:ClogP), 2, scale))

## create model matrix
X <- model.matrix(~ 0 + (Spher + BSEP + THP1 + Glu + Gal + ClogP + BA)^2 +
  log10.cmax, data = data.frame(d2, BA = d$BA))

compiled_model <- cmdstan_model("../Stan/DILI_model.stan")

## fit model on all data to get cutpoints for plots
m1_unweighted <- compiled_model$sample(
  data = list(
    N = nrow(X),
    P = ncol(X),
    y = d$dili.sev,
    X = X,
    w = rep(1, nrow(X)),
    sigma_prior = 0.2,
    make_pred = 1,
    N_pred = nrow(X),
    X_pred = X
  ),
  seed = 123,
  chains = 5,
  parallel_chains = 5,
  iter_warmup = 1000,
  iter_sampling = 2000,
  adapt_delta = 0.95
)


m1_weighted <- compiled_model$sample(
  data = list(
    N = nrow(X),
    P = ncol(X),
    y = d$dili.sev,
    X = X,
    w = calc_weights(d$dili.sev),
    sigma_prior = 0.2,
    make_pred = 1,
    N_pred = nrow(X),
    X_pred = X
  ),
  seed = 123,
  chains = 5,
  parallel_chains = 5,
  iter_warmup = 1000,
  iter_sampling = 2000,
  adapt_delta = 0.95
)


## mean cutpoints
cutpoints_unweighted <- m1_unweighted$draws(format = "df") %>%
  select_at(vars(starts_with("cutpoints"))) %>%
  apply(., 2, plogis) %>%
  colMeans()

cutpoints_weighted <- m1_weighted$draws(format = "df") %>%
  select_at(vars(starts_with("cutpoints"))) %>%
  apply(., 2, plogis) %>%
  colMeans()


## ============================================================
## Leave-one-out cross-validation
## ============================================================

## preallocate vectors and matrices to store results
eta_median_unweighted <- rep(NA, nrow(d))
eta_sd_unweighted <- rep(NA, nrow(d))
ypred_unweighted <- matrix(NA, nrow(d), ncol = 3)

eta_median_weighted <- rep(NA, nrow(d))
eta_sd_weighted <- rep(NA, nrow(d))
ypred_weighted <- matrix(NA, nrow(d), ncol = 3)

## Do LOO
for (i in 1:nrow(d)) {
  ## fit model with no weights
  no_wts <- compiled_model$sample(
    data = list(
      N = nrow(X) - 1,
      P = ncol(X),
      y = d$dili.sev[-i],
      X = X[-i, ],
      w = rep(1, nrow(X) - 1),
      sigma_prior = 0.2,
      make_pred = 1,
      N_pred = 1,
      X_pred = matrix(X[i, ], nrow = 1)
    ),
    seed = 123,
    chains = 5,
    parallel_chains = 5,
    iter_warmup = 1000,
    iter_sampling = 2000,
    adapt_delta = 0.95
  )

  ## extract posterior samples
  post_unweighted <- no_wts$draws(format = "df")

  ## median predictions on probability scale
  eta_median_unweighted[i] <- post_unweighted %>%
    select_at(vars(starts_with("eta_pred"))) %>%
    apply(., 1, plogis) %>%
    median()

  ## SD of predictions
  eta_sd_unweighted[i] <- post_unweighted %>%
    select_at(vars(starts_with("eta_pred"))) %>%
    apply(., 1, plogis) %>%
    sd()

  ## predicted categories
  ypred_unweighted[i, ] <- post_unweighted %>%
    select_at(vars(starts_with("ypred"))) %>%
    pull(1) %>%
    table() %>%
    prop.table() %>%
    t()



  ## fit model with weights
  with_wts <- compiled_model$sample(
    data = list(
      N = nrow(X) - 1,
      P = ncol(X),
      y = d$dili.sev[-i],
      X = X[-i, ],
      w = calc_weights(d$dili.sev[-i]),
      sigma_prior = 0.2,
      make_pred = 1,
      N_pred = 1,
      X_pred = matrix(X[i, ], nrow = 1)
    ),
    seed = 123,
    chains = 5,
    parallel_chains = 5,
    iter_warmup = 1000,
    iter_sampling = 2000,
    adapt_delta = 0.95
  )

  ## extract posterior samples
  post_weighted <- with_wts$draws(format = "df")

  ## median predictions on probability scale
  eta_median_weighted[i] <- post_weighted %>%
    select_at(vars(starts_with("eta_pred"))) %>%
    apply(., 1, plogis) %>%
    median()

  ## SD of predictions
  eta_sd_weighted[i] <- post_weighted %>%
    select_at(vars(starts_with("eta_pred"))) %>%
    apply(., 1, plogis) %>%
    sd()

  ## predicted categories
  ypred_weighted[i, ] <- post_weighted %>%
    select_at(vars(starts_with("ypred"))) %>%
    pull(1) %>%
    table() %>%
    prop.table() %>%
    t()
}



## calculate ranked probability score for each sample
rps_unweighted <- rps(d$dili.sev, ypred_unweighted, return_mean = FALSE)
rps_weighted <- rps(d$dili.sev, ypred_weighted, return_mean = FALSE)

## mass of posterior in correct category as a measure of prediction accuracy
true_category_diff <- rep(NA, nrow(d))
for (i in 1:nrow(d)) {
  true_category_diff[i] <- ypred_weighted[i, d$dili.sev[i]] - ypred_unweighted[i, d$dili.sev[i]]
}



pdf("../../LaTeX_template_files/Fig2.pdf", height = 12, width = 8)
par(
  las = 1,
  mfrow = c(3, 2),
  mar = c(5, 5.1, 3.5, 0.5),
  cex = 0.9,
  cex.axis = 1, cex.lab = 1,
  cex.main = 1
)
beeswarm::beeswarm(eta_median_unweighted ~ d$dili.sev,
  pch = 16, ylim = c(0, 1), method = "center", main = "Unweighted",
  xlab = "True DILI Category", ylab = "Predicted DILI Severity",
  pwcol = d$cols, cex = 0.8
)
abline(h = c(cutpoints_unweighted), lty = 2)
mtext("A", adj = 0, cex = 1.5, font = 2, line = 1.5)

beeswarm::beeswarm(eta_median_weighted ~ d$dili.sev,
  pch = 16, ylim = c(0, 1), method = "center", main = "Weighted",
  xlab = "True DILI Category", ylab = "Predicted DILI Severity",
  pwcol = d$cols, cex = 0.8
)
abline(h = c(cutpoints_weighted), lty = 2)
mtext("B", adj = 0, cex = 1.5, font = 2, line = 1.5)

plot(eta_median_unweighted ~ eta_median_weighted,
  col = d$cols, xlab = "Weighted", ylab = "Unweighted",
  main = "Posterior Median"
)
mtext("C", adj = 0, cex = 1.5, font = 2, line = 1.5)
abline(0, 1)

plot(eta_sd_unweighted ~ eta_sd_weighted,
  col = d$cols, xlab = "Weighted", ylab = "Unweighted",
  xlim = c(0, 0.3), ylim = c(0, 0.3), main = "Posterior SD"
)
mtext("D", adj = 0, cex = 1.5, font = 2, line = 1.5)
abline(0, 1)

beeswarm(true_category_diff ~ factor(d$dili.sev),
  method = "center", pwcol = d$cols, pch = 16, ylim = c(-0.15, 0.15),
  cex = 0.8, xlab = "True DILI Category",
  ylab = "Weighted - Unweighted", main = "Difference in Posterior Mass\nin the True DILI Category",
)
abline(h = 0, lty = 2)
mtext("E", adj = 0, cex = 1.5, font = 2, line = 1.5)
text(3.5, -0.1, "Worse", cex = 1, font = 2, srt = -90)
text(3.5, 0.1, "Better", cex = 1, font = 2, srt = -90)

beeswarm(I(rps_weighted - rps_unweighted) ~ factor(d$dili.sev),
  method = "center", pwcol = d$cols, pch = 16, ylim = c(-0.2, 0.2),
  cex = 0.8, xlab = "True DILI Category",
  ylab = "Weighted - Unweighted", main = "Difference in Ranked\nProbability Score",
)
abline(h = 0, lty = 2)
mtext("F", adj = 0, cex = 1.5, font = 2, line = 1.5)
text(3.5, -0.1, "Better", cex = 1, font = 2, srt = -90)
text(3.5, 0.1, "Worse", cex = 1, font = 2, srt = -90)

dev.off()



## ============================================================
## metrics
## ============================================================


## calculate confusion matrix
cm_unweighted <- table(d$dili.sev, apply(ypred_unweighted, 1, which.max))
cm_weighted <- table(d$dili.sev, apply(ypred_weighted, 1, which.max))

## overall accuracy
cm_unweighted %>%
  diag() %>%
  sum() / nrow(d)

cm_unweighted %>%
  diag() %>%
  sum() / nrow(d)

## overall balanced accuracy
(cm_unweighted[1, 1] / rowSums(cm_unweighted)[1] +
  cm_unweighted[2, 2] / rowSums(cm_unweighted)[2] +
  cm_unweighted[3, 3] / rowSums(cm_unweighted)[3]) / 3

(cm_weighted[1, 1] / rowSums(cm_weighted)[1] +
  cm_weighted[2, 2] / rowSums(cm_weighted)[2] +
  cm_weighted[3, 3] / rowSums(cm_weighted)[3]) / 3


## overall mean calibration (calculated as MSE between actual and predicted proportions)
sum((prop.table(table(d$dili.sev)) - prop.table(table(apply(ypred_unweighted, 1, which.max))))^2)
sum((prop.table(table(d$dili.sev)) - prop.table(table(apply(ypred_weighted, 1, which.max))))^2)

## 1 vs. {2+3} unweighted 
auc(d$dili.sev >= 2, eta_median_unweighted)
1 - ce(d$dili.sev >= 2, as.numeric(eta_median_unweighted > cutpoints_unweighted[1]))
balanced_accuracy(d$dili.sev >= 2, eta_median_unweighted, cutpoints_unweighted[1])
sensitivity(d$dili.sev >= 2, eta_median_unweighted, cutpoints_unweighted[1])
specificity(d$dili.sev >= 2, eta_median_unweighted, cutpoints_unweighted[1])
ppv(d$dili.sev >= 2, eta_median_unweighted, cutpoints_unweighted[1])
npv(d$dili.sev >= 2, eta_median_unweighted, cutpoints_unweighted[1])
f1Score(d$dili.sev >= 2, eta_median_unweighted, cutpoints_unweighted[1])
p4(d$dili.sev >= 2, eta_median_unweighted, cutpoints_unweighted[1])


## 1 vs. {2+3} weighted
auc(d$dili.sev >= 2, eta_median_weighted)
1 - ce(d$dili.sev >= 2, as.numeric(eta_median_weighted > cutpoints_weighted[1]))
balanced_accuracy(d$dili.sev >= 2, eta_median_weighted, cutpoints_weighted[1])
sensitivity(d$dili.sev >= 2, eta_median_weighted, cutpoints_weighted[1])
specificity(d$dili.sev >= 2, eta_median_weighted, cutpoints_weighted[1])
ppv(d$dili.sev >= 2, eta_median_weighted, cutpoints_weighted[1])
npv(d$dili.sev >= 2, eta_median_weighted, cutpoints_weighted[1])
f1Score(d$dili.sev >= 2, eta_median_weighted, cutpoints_weighted[1])
p4(d$dili.sev >= 2, eta_median_weighted, cutpoints_weighted[1])


## {1+2} vs. 3 unweighted
auc(d$dili.sev == 3, eta_median_unweighted)
1 - ce(d$dili.sev == 3, as.numeric(eta_median_unweighted > cutpoints_unweighted[2]))
balanced_accuracy(d$dili.sev == 3, eta_median_unweighted, cutpoints_unweighted[2])
sensitivity(d$dili.sev == 3, eta_median_unweighted, cutpoints_unweighted[2])
specificity(d$dili.sev == 3, eta_median_unweighted, cutpoints_unweighted[2])
ppv(d$dili.sev == 3, eta_median_unweighted, cutpoints_unweighted[2])
npv(d$dili.sev == 3, eta_median_unweighted, cutpoints_unweighted[2])
f1Score(d$dili.sev == 3, eta_median_unweighted, cutpoints_unweighted[2])
p4(d$dili.sev == 3, eta_median_unweighted, cutpoints_unweighted[2])


## {1+2} vs. 3 weighted
auc(d$dili.sev == 3, eta_median_weighted)
1 - ce(d$dili.sev == 3, as.numeric(eta_median_weighted > cutpoints_unweighted[2]))
balanced_accuracy(d$dili.sev == 3, eta_median_weighted, cutpoints_unweighted[2])
sensitivity(d$dili.sev == 3, eta_median_weighted, cutpoints_weighted[2])
specificity(d$dili.sev == 3, eta_median_weighted, cutpoints_weighted[2])
ppv(d$dili.sev == 3, eta_median_weighted, cutpoints_weighted[2])
npv(d$dili.sev == 3, eta_median_weighted, cutpoints_weighted[2])
f1Score(d$dili.sev == 3, eta_median_weighted, cutpoints_weighted[2])
p4(d$dili.sev == 3, eta_median_weighted, cutpoints_weighted[2])

