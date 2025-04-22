library(tidyverse)
library(cmdstanr)
library(beeswarm)
library(CalibrationCurves)
library(ModelMetrics)

source("functions.R")

## read in data
d <- read.csv("../data/simulated_data.csv")

## compile model
compiled_model <- cmdstan_model("../Stan/binary_model.stan")

## fit model with no weighting
m1_unweighted <- compiled_model$sample(
  data = list(
    N = nrow(d),
    y = d$y,
    x1 = d$x1,
    x2 = d$x2,
    w = rep(1, nrow(d))
  ),
  seed = 123,
  chains = 5,
  parallel_chains = 5,
  iter_warmup = 1000,
  iter_sampling = 2000,
  adapt_delta = 0.95
)


## fit model with weights
m1_weighted <- compiled_model$sample(
  data = list(
    N = nrow(d),
    y = d$y,
    x1 = d$x1,
    x2 = d$x2,
    w = calc_weights(d$y)
  ),
  seed = 123,
  chains = 5,
  parallel_chains = 5,
  iter_warmup = 1000,
  iter_sampling = 2000,
  adapt_delta = 0.95
)

## compare parameter estimates
m1_unweighted$summary(c("b0", "b1", "b2"))
m1_weighted$summary(c("b0", "b1", "b2"))


## extract posterior samples
post_unweighted <- m1_unweighted$draws(format = "df")
post_weighted <- m1_weighted$draws(format = "df")


## extract predicted values
eta_unweighted <- post_unweighted %>%
  select_at(vars(starts_with("eta"))) %>%
  apply(., 2, plogis)

eta_weighted <- post_weighted %>%
  select_at(vars(starts_with("eta"))) %>%
  apply(., 2, plogis)


## calculate median predicted values
eta_unweighted_median <- apply(eta_unweighted, 2, median)
eta_weighted_median <- apply(eta_weighted, 2, median)

## extract parameters for plotting
params_unweighted <- post_unweighted %>%
  select_at(vars(starts_with("b"))) %>%
  apply(., 2, quantile, probs = c(0.025, 0.5, 0.975))

params_weighted <- post_weighted %>%
  select_at(vars(starts_with("b"))) %>%
  apply(., 2, quantile, probs = c(0.025, 0.5, 0.975))

## extract predicted values
ypred_unweighted <- post_unweighted %>%
  select_at(vars(starts_with("y_pred"))) %>%
  apply(., 2, function(x) prop.table(table(x))) %>%
  t()

ypred_weighted <- post_weighted %>%
  select_at(vars(starts_with("y_pred"))) %>%
  apply(., 2, function(x) prop.table(table(x))) %>%
  t()


## calculate decision boundaries
x1_seq <- seq(-2, 0.7, length.out = 100)
x2_boundary_unweighted <- (-mean(post_unweighted$b0) - mean(post_unweighted$b1) * x1_seq) /
  mean(post_unweighted$b2)
x2_boundary_weighted <- (-mean(post_weighted$b0) - mean(post_weighted$b1) * x1_seq) / mean(post_weighted$b2)


pdf("../../LaTeX_template_files/Fig1.pdf", height = 10.2, width = 7.8)
par(
  las = 1,
  mfrow = c(3, 2),
  mar = c(5, 5.2, 3.5, 1),
  cex.axis = 1.5, cex.lab = 1.5,
  cex.main = 1.5,
  mgp = c(3.75, 1, 0)
)

beeswarm(x1 ~ y,
  data = d, corral = "wrap",
  pwcol = ifelse(d$y == 1, "firebrick", "steelblue"),
  pwpch = ifelse(d$y == 1, 16, 1),
  method = "center", main = "Predictor X1",
  xlab = "True Class", ylab = "Value"
)
mtext("A", adj = 0, cex = 1.75, font = 2, line = 1.5)

beeswarm(x2 ~ y,
  data = d, corral = "wrap",
  pwcol = ifelse(d$y == 1, "firebrick", "steelblue"),
  pwpch = ifelse(d$y == 1, 16, 1),
  method = "center", main = "Predictor X2",
  xlab = "True Class", ylab = "Value",
)
mtext("B", adj = 0, cex = 1.75, font = 2, line = 1.5)

plot(params_unweighted[2, ] ~ I(c(1, 2, 3) - 0.05),
  xlab = "Parameter",
  ylab = "Value",
  main = "Parameter estimates",
  xaxt = "n", ylim = c(-4, 4), xlim = c(0.75, 3.25), pch = 21, bg = "lightgrey"
)
segments(
  x0 = I(c(1, 2, 3) - 0.05),
  y0 = params_unweighted[1, ],
  x1 = I(c(1, 2, 3) - 0.05),
  y1 = params_unweighted[3, ], lty = 2, lwd = 1.5
)
points(params_unweighted[2, ] ~ I(c(1, 2, 3) - 0.05), pch = 21, bg = "lightgrey")

points(params_weighted[2, ] ~ I(c(1, 2, 3) + 0.05),
  pch = 18, cex = 1.4
)
segments(
  x0 = I(c(1, 2, 3) + 0.05),
  y0 = params_weighted[1, ],
  x1 = I(c(1, 2, 3) + 0.05),
  y1 = params_weighted[3, ], lwd = 1.5
)
axis(1,
  at = c(1, 2, 3),
  labels = c(expression(beta[0]), expression(beta[1]), expression(beta[2]))
)
abline(h = 0, lty = 3)
legend("topright",
  legend = c("Unweighted", "Weighted"), lwd = 1.5,
  pch = c(21, 18), pt.bg = c("lightgrey", "black"), pt.cex = 1.2
)
mtext("C", adj = 0, cex = 1.75, font = 2, line = 1.5)


plot(x2 ~ x1,
  data = d, col = ifelse(d$y == 1, "firebrick", "steelblue"),
  pch = ifelse(d$y == 1, 16, 1), xlim = c(-2, 1.2), ylim = c(-0.6, 2),
  xlab = "X1", ylab = "X2", main = "Decision boundary"
)

lines(x2_boundary_unweighted ~ x1_seq,
  col = "black", lwd = 2, lty = 3
)

lines(x2_boundary_weighted ~ x1_seq,
  col = "black", lwd = 2, lty = 2
)

text(x1_seq[100] - 0.1, x2_boundary_unweighted[100], "Unweighted", pos = 4, cex = 0.9, font = 2)
text(x1_seq[100], x2_boundary_weighted[100], "Weighted", pos = 4, cex = 0.9, font = 2)
legend("topright",
  legend = c("Class 0", "Class 1"),
  col = c("steelblue", "firebrick"), pch = c(1, 16)
)
mtext("D", adj = 0, cex = 1.75, font = 2, line = 1.5)


beeswarm(eta_unweighted_median ~ d$y,
  corral = "wrap",
  pwcol = ifelse(d$y == 1, "firebrick", "steelblue"),
  pwpch = ifelse(d$y == 1, 16, 1),
  ylim = c(0, 1), method = "center", main = "Unweighted analysis",
  xlab = "True Class", ylab = "Predicted probability",
)
abline(h = 0.5, lty = 3)
mtext("E", adj = 0, cex = 1.75, font = 2, line = 1.5)

beeswarm(eta_weighted_median ~ d$y,
  corral = "wrap",
  pwcol = ifelse(d$y == 1, "firebrick", "steelblue"),
  pwpch = ifelse(d$y == 1, 16, 1),
  ylim = c(0, 1), method = "center", main = "Weighted analysis",
  xlab = "True Class", ylab = "Predicted probability",
)
abline(h = 0.5, lty = 3)
mtext("F", adj = 0, cex = 1.75, font = 2, line = 1.5)

dev.off()




## ============================================================
## metrics
## ============================================================

## threshold for binary classification
threshold <- 0.5

## unweighted model metrics
auc(d$y, eta_unweighted_median)
brier(d$y, eta_unweighted_median)
balanced_brier(d$y, eta_unweighted_median)
1 - ce(d$y, as.numeric(eta_unweighted_median > threshold)) # accuracy
balanced_accuracy(d$y, eta_unweighted_median)
sensitivity(d$y, eta_unweighted_median)
specificity(d$y, eta_unweighted_median)
ppv(d$y, eta_unweighted_median)
npv(d$y, eta_unweighted_median)
f1Score(d$y, eta_unweighted_median)
p4(d$y, eta_unweighted_median)
## mean calibration (calculated as MSE between actual and predicted proportions)
sum((prop.table(table(d$y)) - prop.table(table(apply(ypred_unweighted, 1, which.max))))^2)
## weak calibration
val.prob.ci.2(eta_unweighted_median, d$y)


## weighted model metrics
auc(d$y, eta_weighted_median)
brier(d$y, eta_weighted_median)
balanced_brier(d$y, eta_weighted_median)
1 - ce(d$y, as.numeric(eta_weighted_median > threshold)) # accuracy
balanced_accuracy(d$y, eta_weighted_median)
sensitivity(d$y, eta_weighted_median)
specificity(d$y, eta_weighted_median)
ppv(d$y, eta_weighted_median)
npv(d$y, eta_weighted_median)
f1Score(d$y, eta_weighted_median)
p4(d$y, eta_weighted_median)
## mean calibration (calculated as MSE between actual and predicted proportions)
sum((prop.table(table(d$y)) - prop.table(table(apply(ypred_weighted, 1, which.max))))^2)
## weak calibration
val.prob.ci.2(eta_weighted_median, d$y)
