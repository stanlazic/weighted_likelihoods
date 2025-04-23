using Pkg
Pkg.activate(".")

# run once to install packages
# Pkg.instantiate()

using CSV
using DataFrames
using Turing
using Turing: @addlogprob!


d = CSV.read(joinpath("..", "data", "simulated_data.csv"), DataFrame)


# define model
@model function model(y, x1, x2, w)
    # Priors
    b0 ~ Normal(0, 10)
    b1 ~ Normal(0, 10)
    b2 ~ Normal(0, 10)

    # Likelihood
    eta = b0 .+ b1 .* x1 .+ b2 .* x2
    @addlogprob! sum(loglikelihood.(BernoulliLogit.(eta), y) .* w)
end

# unweighted model, pass in 1.0 for weights
post_unweighted = sample(model(d.y, d.x1, d.x2, repeat([1.0], nrow(d))),
                         NUTS(),
                         MCMCThreads(),
                         3_000,
                         3)

# weighted model
post_weighted = sample(model(d.y, d.x1, d.x2, d.w),
                       NUTS(),
                       MCMCThreads(),
                       3_000, 3)
