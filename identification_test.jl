
using Optim
using LinearAlgebra
using Random
using Distributions
using BlackBoxOptim
using MLDataUtils
using Plots
include("data_sim_seed.jl")
include("data_sim_like.jl")
n_firms=500 #divide sample in two parts for two fold
n_rep=1


function check_id(b)
    up_data, down_data, up_profit_data_cf, down_profit_data_cf, price_data_cf, A_mat, β_diff, β_up, β_down, Σ_up, Σ_down, t =
     sim_data([b[1],b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5])], n_firms, 25+n_rep)
    Random.seed!(25+n_rep)
    price_data_cf = price_data_cf + 2*rand(Normal(0.,1.0), n_firms)
# Matrix of data: x, y, p; each row is an observation
    data = zeros(n_firms, 3)
    for i = 1:n_firms
        data[i,1] = up_data[1,i]
        data[i,2] = down_data[1,i]
        data[i,3] = price_data_cf[i]
    end
    return β_diff*t
end


fun = x-> norm(check_id([2., 5.5, 1.5, 2., 3.])- check_id(vcat(2.5,x)))

res = Optim.optimize(fun, rand(4))
@show Optim.minimizer(res)
