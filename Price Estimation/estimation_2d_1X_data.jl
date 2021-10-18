# Estimation with profit data

using Statistics
using BSON
using Plots
using Optim
using LinearAlgebra
using Random
using Distributions
using JuMP
using Gurobi
using KernelDensity
using DelimitedFiles
using CSV
using DataFrames
include("LP_DGP-mc-21d.jl")



data = readdlm("/Users/amir/github/NPSML-Estimator-Price-Data/Profit Estimation/data.csv", ',', Float64, '\n', header=true)


labels= data[2]




data = data[1]
minimum(data[:,1])
#
# data= copy(dat)
# data[:,1]= dat[:,5]
# data[:,2]= dat[:,2]
# data[:,5]= dat[:,1]
# scatter(log.data[:,5])

data_HHI1 = zeros(0,6)
for i=1:n_firms
    if data[i,3] == 1
        data_HHI1 = [data_HHI1; data[i,:]']
    end
end


# data with HHI value different from one
data_HHIn1=zeros(0, 6)
for i=1:n_firms
    if data[i,3] != 1
        data_HHIn1 = [data_HHIn1; data[i,:]']
    end
end



scatter(data_HHIn1[:,3],data_HHIn1[:,6], markersize=2)

cor(data_HHI1[:,1], data_HHI1[:,6])
#
# scatter(data_HHI1[:,2], data_HHI1[:,7], markersize=2)
# scatter(data_HHIn1[:,2], data_HHIn1[:,7], markersize=2)

# scatter(data[:,2], data[:,7], markersize=2)
# cor(data_HHIn1[:,2], data_HHIn1[:,7])


# scatter(data_HHI1[:,2], markersize=3)




function est_cdf_step_1d(y_data, x1_data,  y, x, h)
    num= 0.0
    den= 0.0
    for i =1:size(y_data)[1]
        num+= (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1])
        den+= (pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]))
        # println("x is: ", x1_data[i], " y is: ", y_data[i], " value is: ", (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2]))
    end
    return (num/den)
end

# Bandwidth selection for the sub-sample of firms with HHI == 1
x1_HHI1_den = kde(data_HHI1[:,1])
y1_HHI1_den = kde(data_HHI1[:,2])
# one dimensional cv function
function cv_fun_1d(y_data, x1_data,  y, h, den)
    cv=0.0
    for i = 1:size(y_data)[1]
        cv += ((y_data[i] < y) - est_cdf_step_1d(y_data[1:end .!= i], x1_data[1:end .!= i],  y, [x1_data[i]], h) )^2 * pdf(den, x1_data[i])
    end
    # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return n_firms^(-1)*cv
end



function cv_int_1d(h1,pi_data, x_data, den)
    step_size = floor(Int, size(pi_data)[1]/20)
    y_grid = 1:step_size:size(pi_data)[1]
    y_vec  = pi_data[y_grid[1:end]]  #choose the column corrsponding to  profit
    # println(cv.(data[:,3], data[:,1], y_vec ,h))
    # @show cv(data[:,3], data[:,1], y_vec[1] ,h)
    tmp_f = x->cv_fun_1d(pi_data, x_data, x ,h1)
    cv_vals = map(tmp_f , y_vec)
    println("h: ", h1, " value: ", sum(cv_vals))
    return sum(cv_vals)
end


cv_fun_firm_y1_HHI1 = x->cv_int_1d(x, data_HHI1[:,5], data_HHI1[:,2], y1_HHI1_den)
res_HHI1_y1 = Optim.optimize(cv_fun_firm_y1_HHI1, .001,2.9)
@show h_HHI1_y1 = Optim.minimizer(res_HHI1_y1)


cv_fun_ceo_x_HHI1 = x->cv_int_1d(x, data_HHI1[:,4], data_HHI1[:,1], x1_HHI1_den)
res_HHI1_x = Optim.optimize(cv_fun_ceo_x_HHI1, .001,2.9)
@show h_HHI1_x = Optim.minimizer(res_HHI1_x)



data_HHI1=hcat(data_HHI1, zeros(size(data_HHI1)[1], 2))

for i = 1:size(data_HHI1)[1]
    data_HHI1[i,7] = est_cdf_step_1d(data_HHI1[:,4], data_HHI1[:,1], data_HHI1[i,4], [data_HHI1[i,1]] ,h_HHI1_x)
    data_HHI1[i,8] = est_cdf_step_1d(data_HHI1[:,5], data_HHI1[:,2], data_HHI1[i,5], [data_HHI1[i,2]] , h_HHI1_y1)
end



#################
##################



function est_cdf_step_2d(y_data, x1_data, x2_data,  y, x, h)
    num= 0.0
    den= 0.0
    for i =1:size(y_data)[1]
        num+= (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2])
        den+= (pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]))* pdf(Normal(0,1), (x[2]-x2_data[i])/h[2])
        # println("x is: ", x1_data[i], " y is: ", y_data[i], " value is: ", (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2]))
    end
    return (num/den)
end



# Estimating the marginal densities for x, y1, y2
# This is used for bandwidth selection

x1_HHIn1_den = kde(data_HHIn1[:,1])
y1y2_HHIn1_den = kde((data_HHIn1[:,2],data_HHIn1[:,3]))
# two dimensional cross-validation to use with firm side.
function cv_fun_2d(y_data, x1_data, x2_data,  y, h, den)
    cv=0.0
    for i = 1:size(y_data)[1]
        cv += ((y_data[i] < y) - est_cdf_step_2d(y_data[1:end .!= i], x1_data[1:end .!= i], x2_data[1:end .!= i],  y, [x1_data[i] x2_data[i]], h) )^2 * pdf(den, x1_data[i], x2_data[i])
    end
    # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return (size(y_data)[1])^(-1)*cv
end


# res = Optim.optimize(x->cv_fun_1d(data[:,4], data[:,1], 20. ,x), 0.001, 0.9)
# @show h1d=Optim.minimizer(res)
#
# two dimensional cross validation integrated function for the firm side
function cv_int_2d(h1,y_data,x1_data, x2_data, den )
    step_size = floor(Int, size(y_data)[1]/10)
    y_grid = 1:step_size:size(y_data)[1]
    y_vec  = y_data[y_grid[1:end]]
    # println(cv.(data[:,3], data[:,1], y_vec ,h))
    # @show cv(data[:,3], data[:,1], y_vec[1] ,h)
    tmp_f = x->cv_fun_2d(y_data, x1_data, x2_data, x , h1, den)
    cv_vals = map(tmp_f , y_vec)
    println("h: ", h1, " value: ", sum(cv_vals))
    return sum(cv_vals)
end




cv_fun_firm_y1y2_HHIn1 = x->cv_int_2d(x, data_HHIn1[:,5], data_HHIn1[:,2], data_HHIn1[:,3], y1y2_HHIn1_den)
res_HHIn1_y1y2 = Optim.optimize(cv_fun_firm_y1y2_HHIn1, [1.,1.])
@show h_HHIn1_y1y2 = Optim.minimizer(res_HHIn1_y1y2)



cv_fun_ceo_x_HHIn1 = x->cv_int_1d(x, data_HHIn1[:,4], data_HHIn1[:,1], x1_HHIn1_den)
res_HHI1_x = Optim.optimize(cv_fun_ceo_x_HHI1, .001,2.9)
@show h_HHI1_x = Optim.minimizer(res_HHI1_x)


data_HHIn1=hcat(data_HHIn1, zeros(size(data_HHIn1)[1], 2))

for i = 1:size(data_HHIn1)[1]
    data_HHIn1[i,7] = est_cdf_step_1d(data_HHIn1[:,4], data_HHIn1[:,1], data_HHIn1[i,4], [data_HHIn1[i,1]] ,h_HHI1_x)
    data_HHIn1[i,8] = est_cdf_step_2d(data_HHIn1[:,5], data_HHIn1[:,2], data_HHIn1[:,3], data_HHIn1[i,5], [data_HHIn1[i,2] data_HHIn1[i,3]] , h_HHIn1_y1y2)
end


data_out = vcat(data_HHI1, data_HHIn1)

writedlm("/Users/amir/github/NPSML-Estimator-Price-Data/Profit Estimation/data_out.csv", data_out,  ',')


data = readdlm("/Users/amir/github/NPSML-Estimator-Price-Data/Profit Estimation/data.csv", ',', Float64, '\n', header=true)


labels= data[2]




data = data[1]
minimum(data[:,1])
#
# data= copy(dat)
# data[:,1]= dat[:,5]
# data[:,2]= dat[:,2]
# data[:,5]= dat[:,1]
# scatter(log.data[:,5])

data_HHI1 = zeros(0,6)
for i=1:n_firms
    if data[i,3] == 1
        data_HHI1 = [data_HHI1; data[i,:]']
    end
end


# data with HHI value different from one
data_HHIn1=zeros(0, 6)
for i=1:n_firms
    if data[i,3] != 1
        data_HHIn1 = [data_HHIn1; data[i,:]']
    end
end



scatter(data_HHIn1[:,3],data_HHIn1[:,6], markersize=2)

cor(data_HHI1[:,1], data_HHI1[:,6])
#
# scatter(data_HHI1[:,2], data_HHI1[:,7], markersize=2)
# scatter(data_HHIn1[:,2], data_HHIn1[:,7], markersize=2)

# scatter(data[:,2], data[:,7], markersize=2)
# cor(data_HHIn1[:,2], data_HHIn1[:,7])


# scatter(data_HHI1[:,2], markersize=3)




function est_cdf_step_1d(y_data, x1_data,  y, x, h)
    num= 0.0
    den= 0.0
    for i =1:size(y_data)[1]
        num+= (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1])
        den+= (pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]))
        # println("x is: ", x1_data[i], " y is: ", y_data[i], " value is: ", (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2]))
    end
    return (num/den)
end

# Bandwidth selection for the sub-sample of firms with HHI == 1
x1_HHI1_den = kde(data_HHI1[:,1])
y1_HHI1_den = kde(data_HHI1[:,2])
# one dimensional cv function
function cv_fun_1d(y_data, x1_data,  y, h, den)
    cv=0.0
    for i = 1:size(y_data)[1]
        cv += ((y_data[i] < y) - est_cdf_step_1d(y_data[1:end .!= i], x1_data[1:end .!= i],  y, [x1_data[i]], h) )^2 * pdf(den, x1_data[i])
    end
    # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return n_firms^(-1)*cv
end



function cv_int_1d(h1,pi_data, x_data, den)
    step_size = floor(Int, size(pi_data)[1]/20)
    y_grid = 1:step_size:size(pi_data)[1]
    y_vec  = pi_data[y_grid[1:end]]  #choose the column corrsponding to  profit
    # println(cv.(data[:,3], data[:,1], y_vec ,h))
    # @show cv(data[:,3], data[:,1], y_vec[1] ,h)
    tmp_f = x->cv_fun_1d(pi_data, x_data, x ,h1)
    cv_vals = map(tmp_f , y_vec)
    println("h: ", h1, " value: ", sum(cv_vals))
    return sum(cv_vals)
end


cv_fun_firm_y1_HHI1 = x->cv_int_1d(x, data_HHI1[:,5], data_HHI1[:,2], y1_HHI1_den)
res_HHI1_y1 = Optim.optimize(cv_fun_firm_y1_HHI1, .001,2.9)
@show h_HHI1_y1 = Optim.minimizer(res_HHI1_y1)


cv_fun_ceo_x_HHI1 = x->cv_int_1d(x, data_HHI1[:,4], data_HHI1[:,1], x1_HHI1_den)
res_HHI1_x = Optim.optimize(cv_fun_ceo_x_HHI1, .001,2.9)
@show h_HHI1_x = Optim.minimizer(res_HHI1_x)



data_HHI1=hcat(data_HHI1, zeros(size(data_HHI1)[1], 2))

for i = 1:size(data_HHI1)[1]
    data_HHI1[i,7] = est_cdf_step_1d(data_HHI1[:,4], data_HHI1[:,1], data_HHI1[i,4], [data_HHI1[i,1]] ,h_HHI1_x)
    data_HHI1[i,8] = est_cdf_step_1d(data_HHI1[:,5], data_HHI1[:,2], data_HHI1[i,5], [data_HHI1[i,2]] , h_HHI1_y1)
end



#################
##################



function est_cdf_step_2d(y_data, x1_data, x2_data,  y, x, h)
    num= 0.0
    den= 0.0
    for i =1:size(y_data)[1]
        num+= (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2])
        den+= (pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]))* pdf(Normal(0,1), (x[2]-x2_data[i])/h[2])
        # println("x is: ", x1_data[i], " y is: ", y_data[i], " value is: ", (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2]))
    end
    return (num/den)
end



# Estimating the marginal densities for x, y1, y2
# This is used for bandwidth selection

x1_HHIn1_den = kde(data_HHIn1[:,1])
y1y2_HHIn1_den = kde((data_HHIn1[:,2],data_HHIn1[:,3]))
# two dimensional cross-validation to use with firm side.
function cv_fun_2d(y_data, x1_data, x2_data,  y, h, den)
    cv=0.0
    for i = 1:size(y_data)[1]
        cv += ((y_data[i] < y) - est_cdf_step_2d(y_data[1:end .!= i], x1_data[1:end .!= i], x2_data[1:end .!= i],  y, [x1_data[i] x2_data[i]], h) )^2 * pdf(den, x1_data[i], x2_data[i])
    end
    # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return (size(y_data)[1])^(-1)*cv
end


# res = Optim.optimize(x->cv_fun_1d(data[:,4], data[:,1], 20. ,x), 0.001, 0.9)
# @show h1d=Optim.minimizer(res)
#
# two dimensional cross validation integrated function for the firm side
function cv_int_2d(h1,y_data,x1_data, x2_data, den )
    step_size = floor(Int, size(y_data)[1]/10)
    y_grid = 1:step_size:size(y_data)[1]
    y_vec  = y_data[y_grid[1:end]]
    # println(cv.(data[:,3], data[:,1], y_vec ,h))
    # @show cv(data[:,3], data[:,1], y_vec[1] ,h)
    tmp_f = x->cv_fun_2d(y_data, x1_data, x2_data, x , h1, den)
    cv_vals = map(tmp_f , y_vec)
    println("h: ", h1, " value: ", sum(cv_vals))
    return sum(cv_vals)
end




cv_fun_firm_y1y2_HHIn1 = x->cv_int_2d(x, data_HHIn1[:,5], data_HHIn1[:,2], data_HHIn1[:,3], y1y2_HHIn1_den)
res_HHIn1_y1y2 = Optim.optimize(cv_fun_firm_y1y2_HHIn1, [1.,1.])
@show h_HHIn1_y1y2 = Optim.minimizer(res_HHIn1_y1y2)



cv_fun_ceo_x_HHIn1 = x->cv_int_1d(x, data_HHIn1[:,4], data_HHIn1[:,1], x1_HHIn1_den)
res_HHI1_x = Optim.optimize(cv_fun_ceo_x_HHI1, .001,2.9)
@show h_HHI1_x = Optim.minimizer(res_HHI1_x)


data_HHIn1=hcat(data_HHIn1, zeros(size(data_HHIn1)[1], 2))

for i = 1:size(data_HHIn1)[1]
    data_HHIn1[i,7] = est_cdf_step_1d(data_HHIn1[:,4], data_HHIn1[:,1], data_HHIn1[i,4], [data_HHIn1[i,1]] ,h_HHI1_x)
    data_HHIn1[i,8] = est_cdf_step_2d(data_HHIn1[:,5], data_HHIn1[:,2], data_HHIn1[:,3], data_HHIn1[i,5], [data_HHIn1[i,2] data_HHIn1[i,3]] , h_HHIn1_y1y2)
end


data_out = vcat(data_HHI1, data_HHIn1)

writedlm("/Users/amir/github/NPSML-Estimator-Price-Data/Profit Estimation/data_out.csv", data_out,  ',')


cor(data_out[:,2], data_out[:,7])





# Objective function: Argument is the vector of parameters beta
# It solves for the market with the realized characteristics,
# This means using the quantiles that are implied from the profits.
function md_objective(b)
    # A_mat = [
    #  3. 2. 0.5;
    #  1. 0. 1.;
    #
    # ]
    # sig_up = [
    #  0.2 .2;
    #  0.0 .4]
    # sig_down = [
    #  0.5 .25;
    #  0.1 .5;
    #  0.0 0.3]
    A_mat = [b[1] b[2] 1.0;
            1.0 0. 1.]
    # [3., 2., 1., 1., 1, 1., .4 .3]
    # sig_up = [1. .2; 0 .2]
    # sig_down = [0.5 .25; 0. .5]
    # ud, dd, up, dp,down =  sim_data_LP(b_up,b_down,sig_up,sig_down,1500,2)
    eps_vec = quantile.(LogNormal(0.,abs(b[3])),data[:,6])
    eta_vec = quantile.(LogNormal(0., abs(b[4])) ,data[:,7])
    # eta_vec = quantile.(LogNormal(0., abs(b[10])),data[:,8])
    surplus_vec = diag(hcat(data[:,1], eps_vec) * A_mat * Transpose(hcat(data[:,2:3], eta_vec)))
    total_profit = data[:,4]+ data[:,5]
    # println("par is ", b,  " value is: ", sum((total_profit - surplus_vec).^2))
    return sum((total_profit - surplus_vec).^2)
end

# res= Optim.optimize(md_objective, rand(10))
res= Optim.optimize(md_objective, [3., 2., 2.5, 1.])
res= Optim.optimize(md_objective, rand(4))

@show Optim.minimizer(res)

scatter(data[:,6], data[:,7], markersize=2)
cor(data[:,1], data[:,6])

scatter(data_HHI1[:,6], data_HHI1[:,7], markersize=2)
scatter(data_HHI1[:,1], data_HHI1[:,6], markersize=2)
cor(data_HHI1[:,7], data_HHI1[:,6])




Optim.minimizer(res)


# res= Optim.optimize(md_objective, rand(5))

# res= Optim.optimize(md_objective, [-3, 2, 1., 1., -1, 0.5, 1. , 1. , .4, .3], iterations=1000000)
# res= Optim.optimize(md_objective, [-3, 2, 1., -1., .4, .3], iterations=1000000)

# res= Optim.optimize(md_objective, rand(6), iterations=1000000)

# Optim.minimizer(res)
#
# return vcat(Optim.minimizer(res),Optim.minimum(res))
# end
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

reps=1:4
est_pars = pmap(rep_fun,reps)
# @show est_pars

# mean(abs(est_pars))





estimation_result = Dict()
push!(estimation_result, "beta_hat" => est_pars)
# push!(estimation_result, "beta" => A_mat)


bson("/Users/amir/Downloads/mc_profs_01_tst.bson", estimation_result)


# A_mat = [
#  3. 2. 1.;
#  1. 1. 0.5;
#  1. 1. 1.0
# ]
# sig_up = [
#  0.2 .2;
#  0.3 .2;
#  0.0 .4]
# sig_down = [
#  0.5 .25;
#  0.1 .5;
#  0.0 0.3]
#


est_bcv_500 = BSON.load("/Users/amir/Downloads/mc_profs_01_tst.bson")
beta500 = est_bcv_500["beta_hat"]
# true_pars=[-3., .7, 3, .2, .5]
true_pars = [3., 2., 2.5, 1., .4 ,.3, 0. ]
# true_pars=[3., 2., 1., 1., -1, 1., .4, .3, 0]
# true_pars=[3., 2., 1., 1., 1, 1., .5, 1., 1. , .4, 0]
errs = zeros(4, 7)
for i =1:4
        beta500[i]=(beta500[i])
        println(round.(beta500[i]; digits=2))
        beta500[i][4] = abs(beta500[i][4])
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
# bias: [0.6671, -0.2114, -0.5944, 0.0756, 0.0918, 151.8326]
# rmse: [0.87 0.4 0.77 0.12 0.16 170.25]


scatter(data[:,2], data[:,3],markersize=2)
