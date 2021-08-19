# Estimation with profit data



using Distributed
using BSON
addprocs(4)

@everywhere begin
    using Optim
    using LinearAlgebra
    using Random
    using Distributions
    using JuMP
    using Gurobi
    using KernelDensity
end


@everywhere include("LP_DGP-mc.jl")
@everywhere include("data_sim_seed.jl")
@everywhere include("sinkhorn_DGP-1.jl")
@everywhere n_firms=500


i=1

n_rep=i
b_up = [4 0.5; 1. 0.2]
b_down = [-3 0.2; 2. 0.3]

sig_up = [0.2 .2; 0 .2]
sig_down = [0.5 .25; 0. .5]
# ud, dd, up, dp,down =  sim_data_LP(b_up,b_down,sig_up,sig_down,1500,2)
up_data, down_data, up_profit_data, down_profit_data, dd =
 sim_data_LP(b_up, b_down, sig_up, sig_down, n_firms, 25+n_rep)

data = zeros(n_firms, 6)
for i = 1:n_firms
    data[i,1] = up_data[1,i]
    data[i,2] = down_data[1,i]
    data[i,3] = up_profit_data[i]
    data[i,4] = down_profit_data[i]
    data[i,5] = up_data[2,i]
    data[i,6] = down_data[2,i]
end

up_data1, down_data1, up_profit_data1, down_profit_data1 =
 sim_data_sinkhorn(b_up, b_down, sig_up, sig_down, n_firms, 25+n_rep)
# scatter(data[:,1],data[:,2])


scatter(data[:,1], data[:,2], markersize=3, legends=false,
   xlabel="upstream x", ylabel="downstream y", color=:red)
#

scatter(data[:,1], data[:,3], markersize=3, legends=false,
   xlabel="upstream x", ylabel="downstream y", color=:red)
# scatter(up_data1[1,:], data[:,1], markersize=3, legends=false,
#  xlabel="upstream x", ylabel="downstream y", color=:yellow)



scatter!(up_data1[1,:], down_data1[1,:], markersize=3, legends=false,
 xlabel="upstream x", ylabel="downstream y", color=:yellow)

scatter!(up_data1[1,:], up_profit_data1, markersize=3, legends=false,
 xlabel="upstream x", ylabel="downstream y", color=:yellow)



scatter(data[:,2], data[:,4], markersize=3, legends=false,
 xlabel="upstream x", ylabel="downstream y", color=:red)
scatter!(down_data1[1,:], down_profit_data1, markersize=3, legends=false,
 xlabel="upstream x", ylabel="downstream y", color=:yellow)















function est_cdf_step(y_data, x_data,  y, x, h)
    num= 0.0
    den= 0.0
    for i =1:size(y_data)[1]
        num+= (y_data[i] < y) * pdf(Normal(0,1), (x-x_data[i])/h)
        den+= (pdf(Normal(0,1), (x-x_data[i])/h))
        # println("x is: ", x_data[i], " y is: ", y_data[i], " value is: ", (1/(h[1]*h[2])) * cdf(Normal(0,1), (y-y_data[i])/h[1])* pdf(Normal(0,1), (x-x_data[i])/h[2]))
    end
    return (num/den)
end
#
x_den = kde(data[:,1])
# y_den = kde(data[:,2])
# plot(x->pdf(x_den,x), -10,10.)
function cv_fun(y_data, x_data,  y, h)
    cv=0.0
    for i = 1:n_firms
        cv += ((y_data[i] < y) - est_cdf_step(y_data[1:end .!= i], x_data[1:end .!= i],  y, x_data[i], h) )^2 * pdf(x_den, x_data[i])
    end
    # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return n_firms^(-1)*cv
end

# Optim.optimize(x->cv(data[:,3], data[:,1], 2. ,x), 0.0001, 1.)
#
#
#
#
function cv_int(h1)
    step_size = floor(Int, n_firms/10)
    y_grid = 1:step_size:n_firms
    y_vec  = data[y_grid[1:end],3]
    # println(cv.(data[:,3], data[:,1], y_vec ,h))
    # @show cv(data[:,3], data[:,1], y_vec[1] ,h)
    tmp_f = x->cv_fun(data[:,3], data[:,1], x ,h1)
    cv_vals = map(tmp_f , y_vec)
    println("h: ", h1, " value: ", sum(cv_vals))
    return sum(cv_vals)
end
#
# @show cv_int(1.2)
# resh = Optim.optimize(cv_int, 0.0001, 1.0)
# @show h = Optim.minimizer(resh)
h= 0.05

data=hcat(data, zeros(n_firms, 2))
for i = 1:n_firms
    data[i,7] = est_cdf_step(data[:,3], data[:,1], data[i,3], data[i,1] ,h)
    data[i,8] = est_cdf_step(data[:,4], data[:,2], data[i,4], data[i,2] , h)
end



# Objective function: Argument is the vector of parameters beta
# It solves for the market with the realized characteristics,
# This means using the quantiles that are implied from the profits.
function md_objective(b)
    # b_up = [b[1] 0.5; 1. 0.2]
    # b_down = [b[2] 0.2; 2. 0.3]
    A_mat = [b[1] b[2]; b[3] 0.5]

    sig_up = [1. .2; 0 .2]
    sig_down = [0.5 .25; -1 .5]
    # ud, dd, up, dp,down =  sim_data_LP(b_up,b_down,sig_up,sig_down,1500,2)
    eps_vec = quantile.(LogNormal(0.,abs(b[4])),data[:,7])
    eta_vec = quantile.(LogNormal(0., abs(b[5])),data[:,8])
    surplus_vec = diag(hcat(data[:,1], eps_vec) * A_mat * Transpose(hcat(data[:,2], eta_vec)))
    total_profit = data[:,3]+ data[:,4]
    return sum((total_profit - surplus_vec).^2)
end

# res= Optim.optimize(md_objective, rand(5))
res= Optim.optimize(md_objective, [-3., 0.7, 3., .2, .001])
println("res")

#
#
# scatter(data[:,1], data[:,2], markersize=2, legends=false,
#    xlabel="upstream x", ylabel="downstream y")
# savefig("/Users/amir/Downloads/figs/matching")
#
#
# scatter(data[:,1], data[:,3], markersize=2, legends=false,
#    xlabel="upstream x", ylabel="upstream profit")
# savefig("/Users/amir/Downloads/figs/up_profit")
#
#
#
# scatter(data[:,5], data[:,6], markersize=2, legends=false,
#    xlabel="upstream eps", ylabel="downstream eta",
#     xlims=(0,2), ylims=(0,2))
# savefig("/Users/amir/Downloads/figs/unobs_real")
#
# scatter(data[:,7], data[:,8], markersize=2, legends=false,
#    xlabel="normalized upstream eps", ylabel="normalized downstream eta ")
# savefig("/Users/amir/Downloads/figs/unobs_est")
#
# eps_vec = quantile.(LogNormal(0.,abs(0.2)),data[:,7])
# eta_vec = quantile.(LogNormal(-1, abs(0.5)),data[:,8])
#
#
# scatter(eps_vec, eta_vec, markersize=2, legends=false,
#    xlabel="estimated upstream eps", ylabel="estimated downstream eta "
#    , xlims=(0,2), ylims=(0,2))
# savefig("/Users/amir/Downloads/figs/unobs_est_inverted")
#

reps=1:20
est_pars = pmap(rep_fun,reps)
est_pars

mean(est_pars)




# -3.0  0.7
#  3.0  0.5

estimation_result = Dict()
push!(estimation_result, "beta_hat" => est_pars)
# push!(estimation_result, "beta" => A_mat)


bson("/Users/amir/Downloads/mc_profs_01_tst.bson", estimation_result)




est_bcv_500 = BSON.load("/Users/amir/Downloads/mc_profs_03.bson")
beta500 = est_bcv_500["beta_hat"]
true_pars=[-3., .7, 3, .2, .5]
errs = zeros(80, 5)
for i =1:80
        beta500[i]=(beta500[i])
        println(round.(beta500[i]; digits=2))
        beta500[i][4] = (beta500[i][4])
        beta500[i][5] = abs(beta500[i][5])
        println(round.(beta500[i]; digits=2))
        errs[i,:]=beta500[i] - true_pars
end
mse_500 =((mean(errs.^2,dims=1)))


bias_500 = mean((beta500))-true_pars

println("nfirms: ", 1500)

println("bias: ",round.(bias_500; digits=4))
println("rmse: ",round.(sqrt.(mse_500); digits=2))
# println("mse: ",round.((mse_500); digits=4))
