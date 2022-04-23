using OptimalTransport
using Distances
using Plots
using PythonOT: PythonOT
using Tulip

using LinearAlgebra
using Random

using ExactOptimalTransport

using Tulip




# uniform histograms
μ = fill(1/250, 250)
ν = fill(1/200, 200)

# random cost matrix
C = pairwise(SqEuclidean(), rand(1, 250), rand(1, 200); dims=2)

# regularization parameter
ε = 0.01

# solve entropically regularized optimal transport problem
sinkhorn(μ, ν, C, ε)
sinkhorn_stabilized(μ, ν, C, ε, return_duals = true)




# compute optimal transport map with Tulip
lp = Tulip.Optimizer()
    P = ExactOptimalTransport.emd(μ, ν, C, lp)

# compute optimal transport cost without recomputing the plan
ExactOptimalTransport.emd2(μ, ν, C, lp; plan=P)


using StochasticOptimalTransport
wasserstein(C, μ, ν)

ot_entropic_dual(u, v, eps, K)
