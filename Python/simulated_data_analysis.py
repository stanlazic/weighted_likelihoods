import os
import numpy as np
import pymc as pm
import pandas as pd

d = pd.read_csv(os.path.join("..", "data", "simulated_data.csv"))

with pm.Model() as unweighted_model:
    # Priors
    b0 = pm.Normal("b0", mu=0, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    b2 = pm.Normal("b2", mu=0, sigma=10)
    
    # Likelihood
    eta = pm.math.sigmoid(b0 + b1 * d["x1"] + b2 * d["x2"])
    y = pm.Bernoulli("y", p=eta,  observed=d["y"])
    
    trace_unweighted = pm.sample(2000, tune=1000)



with pm.Model() as weighted_model:
    # convert the data to numpy arrays
    x1 = d["x1"].to_numpy()
    x2 = d["x2"].to_numpy()
    w = d["w"].to_numpy()
    y = d["y"].to_numpy()
    
    # Priors
    b0 = pm.Normal("b0", mu=0, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    b2 = pm.Normal("b2", mu=0, sigma=10)
    
    # Likelihood
    eta = pm.math.sigmoid(b0 + b1 * x1 + b2 * x2)
    logprob = pm.Bernoulli.logp(y, p=eta)
    pm.Potential("weights", (w * logprob).sum())
        
    trace_weighted = pm.sample(2000, tune=1000)

# Extract the posterior samples
trace_unweighted.posterior.mean()
trace_weighted.posterior.mean()
