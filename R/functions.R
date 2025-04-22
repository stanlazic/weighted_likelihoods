## Function definitions

calc_weights <- function(x) {
  ## Calculate weights to account for unbalanced groups. The weights need to sum across the samples to N.
  ## x: vector of group labels (e.g. 0/1, healthy/cancer)
  N <- length(x)
  wts <- 1 / prop.table(table(x)) # wts = inverse frequency

  ## create vector of weights of length N
  group_names <- names(wts)
  weights <- rep(NA, N)

  for (i in 1:N) {
    weights[i] <- wts[match(x[i], group_names)]
  }

  weights <- weights / sum(weights) * N # normalize
  return(weights)
}


p4 <- function(actual, predicted, threshold = 0.5) {
  ## actual: vector of 0/1 outcomes (e.g. 1 = cancer, 0 = healthy)
  ## predicted: vector of predicted probabilities
  ## threshold: threshold for classification (e.g. 0.5)

  4 / (1 / ModelMetrics::sensitivity(actual, predicted, threshold) +
    1 / ModelMetrics::specificity(actual, predicted, threshold) +
    1 / ModelMetrics::ppv(actual, predicted, threshold) +
    1 / ModelMetrics::npv(actual, predicted, threshold))
}


balanced_accuracy <- function(actual, predicted, threshold) {
  ## actual: vector of 0/1 outcomes (e.g. 1 = cancer, 0 = healthy)
  ## predicted: vector of predicted probabilities
  ## threshold: threshold for classification (e.g. 0.5)

  (ModelMetrics::sensitivity(actual, predicted, threshold) +
    ModelMetrics::specificity(actual, predicted, threshold)) / 2
}


rps <- function(actual, predicted, return_mean = TRUE) {
  ## actual: vector of 0/1 outcomes (e.g. 1 = cancer, 0 = healthy)
  ## predicted: vector of predicted probabilities
  ## return_mean: logical, if TRUE return mean RPS, otherwise return individual RPS values

  ## Ensure true values are a factor and ordered
  actual <- as.ordered(actual)

  ## Convert true values to integer indices
  true_indices <- as.integer(actual)

  ## Ensure predicted probabilities are in matrix form
  predicted <- as.matrix(predicted)

  ## Check if the dimensions of the inputs match
  if (length(true_indices) != nrow(predicted) || length(levels(actual)) != ncol(predicted)) {
    stop("The dimensions of the true values and predicted probabilities do not match.")
  }

  ## Initialize RPS
  rps <- numeric(length(true_indices))

  ## Calculate RPS for each observation
  for (i in 1:length(true_indices)) {
    cumulative_probs <- cumsum(predicted[i, ])
    true_cumulative <- rep(0, length(cumulative_probs))
    true_cumulative[true_indices[i]:length(cumulative_probs)] <- 1
    rps[i] <- sum((cumulative_probs - true_cumulative)^2)
  }

  ## Calculate mean RPS
  if (return_mean) {
    mean_rps <- mean(rps)
    return(mean_rps)
  } else {
    ## return individual RPS values
    return(rps)
  }
}



balanced_brier <- function(actual, predicted) {
  ## actual: vector of 0/1 outcomes (e.g. 1 = cancer, 0 = healthy)
  ## predicted: vector of predicted probabilities

  ## Brier score for Class 0 samples
  brier_0 <- mean((predicted[actual == 0] - actual[actual == 0])^2, na.rm = TRUE)

  ## Brier score for Class 1 samples
  brier_1 <- mean((predicted[actual == 1] - actual[actual == 1])^2, na.rm = TRUE)

  ## average Brier score
  (brier_0 + brier_1) / 2
}
