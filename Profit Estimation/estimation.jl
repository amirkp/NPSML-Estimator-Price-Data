# Estimation with profit data


using Distributed
# using BSON
# addprocs(4)

# @everywhere begin
    using Optim
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
# end
using KernelDensity
@everywhere include("data_sim_seed_scalar_unobs.jl")
@everywhere include("data_sim_like.jl")
@everywhere n_firms=1500

using Plots
n_rep=1
b_up = [2 0.5; 1. 0.2]
b_down = [1 0.2; 2. 0.3]

sig_up = [2. 0; 0 1.]
sig_down = [3. 0; 0 1.]
up_data, down_data, up_profit_data_cf, down_profit_data_cf, price_data_cf, A_mat, β_diff, β_up, β_down, Σ_up, Σ_down =
 sim_data(b_up, b_down, sig_up, sig_down, n_firms, 25+n_rep)
Random.seed!(25+n_rep)
# price_data_cf = price_data_cf + rand(Normal(0.,4.), n_firms)

data = zeros(n_firms, 4)
for i = 1:n_firms
    data[i,1] = up_data[1,i]
    data[i,2] = down_data[1,i]
    data[i,3] = up_profit_data_cf[i]
    data[i,4] = down_profit_data_cf[i]
end

scatter(data[:,1], data[:,3], markersize=1)

function est_cdf(y_data, x_data,  y, x, h)
    num= 0.0
    den= 0.0
    for i =1:size(y_data)[1]
        num+= cdf(Normal(0,1), (y-y_data[i])/h[1])* pdf(Normal(0,1), (x-x_data[i])/h[2])
        den+= (pdf(Normal(0,1), (x-x_data[i])/h[2]))
        # println("x is: ", x_data[i], " y is: ", y_data[i], " value is: ", (1/(h[1]*h[2])) * cdf(Normal(0,1), (y-y_data[i])/h[1])* pdf(Normal(0,1), (x-x_data[i])/h[2]))
    end
    return (num/den)
end
scatter(data[:,1], data[:,3], markersize=2)
h= [0.04344535653362956, 0.29067309145466425]
est_cdf(data[:,3], data[:,1], 35, 5. ,h)

plot(x->est_cdf(data[:,3], data[:,1], x,0. ,[.5, .17]), -2,10  , legends=false)
plot!(x->est_cdf(data[:,3], data[:,1], x,-1. ,[.5, .17]), -2,10  , legends=false)
sum(5 .> data[:,3])



x_den = kde(data[:,1])

plot(x->pdf(x_den,x), -10,10.)
function cv(y_data, x_data,  y, h)
    cv=0.0
    for i = 1:n_firms
        cv += ((y_data[i] < y) - est_cdf(y_data[1:end .!= i], x_data[1:end .!= i],  y, x_data[i], h) )^2 * pdf(x_den, x_data[i])
    end
    println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return n_firms^(-1)*cv
end

Optim.optimize(x->cv(data[:,3], data[:,1], 10 ,x), [1.,1.])




function cv_int(h)
    y_grid = 1:100:1500
    y_vec  = data[y_grid[1:end],3]
    cv_vals = pmap(x->cv(data[:,3], data[:,1], x ,h), y_vec)
    println("h: ", h, " value: ", sum(cv_vals)  )
    return sum(cv_vals)
end

cv_int([1.2,0.2])

Optim.optimize(cv_int, [1.,1.])
