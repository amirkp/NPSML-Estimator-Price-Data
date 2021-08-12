# Estimation with profit data



using Distributed
# using BSON
# addprocs(4)

@everywhere begin
    using Optim
    using LinearAlgebra
    using Random
    using Distributions
    using JuMP
    using Gurobi
    using KernelDensity
end

# @everywhere include("data_sim_seed_scalar_unobs.jl")
@everywhere include("LP_DGP.jl")
# @everywhere include("data_sim_like.jl")
@everywhere n_firms=500

using Plots
n_rep=1
b_up = [-4 0.5; 1. 0.2]
b_down = [1 0.2; 2. 0.3]

sig_up = [1. .2; 0 .2]
sig_down = [0.5 .25; -1 .5]
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



plot(x-> pdf(LogNormal(1, .2), x),0 ,10)
plot(x-> pdf(LogNormal(.5, .25), x),0 ,10)

scatter(data[:,5], data[:,6], markersize=1)
scatter(data[:,5], dd[2,:], markersize=1)


scatter(data[:,7], data[:,8], markersize=2)
scatter(data[:,5], dd[2,:], markersize=1)


mean(dd[2,:])
dd[2,:] == data[:,6]
mean(data[:,6])
scatter(dd[2,:])

scatter(data[:,6])

dd


# estimate the conditional cdf of y
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


scatter(data[:,1], data[:,3], markersize=2)
h= [0.4344535653362956, 0.29067309145466425]
est_cdf(data[:,3], data[:,1], 35, 5. ,h)

plot(x->est_cdf(data[:,3], data[:,1], x,1.5,h), -2,10  , legends=false)
plot!(x->est_cdf_step(data[:,3], data[:,1], x,2.5,[1.7, .4]), -2,5  , legends=false)
plot!(x->est_cdf_step(data[:,3], data[:,1], 5,1.5,[.5, .7]), -2,30  , legends=false)
plot(x->est_cdf_step(data[:,3], data[:,1], 10,x ,[.5, .5]), -5,5 , legends=false)



est_cdf(data[:,3], data[:,1], x,1.5,[.5, .07])









x_den = kde(data[:,1])
y_den = kde(data[:,2])
plot(x->pdf(x_den,x), -10,10.)
function cv(y_data, x_data,  y, h)
    cv=0.0
    for i = 1:n_firms
        cv += ((y_data[i] < y) - est_cdf_step(y_data[1:end .!= i], x_data[1:end .!= i],  y, x_data[i], h[1]) )^2 * pdf(x_den, x_data[i])
    end
    # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return n_firms^(-1)*cv
end

Optim.optimize(x->cv(data[:,3], data[:,1], 2. ,x), 0.0001, 1.)




function cv_int(h)
    y_grid = 1:10:1500
    y_vec  = data[y_grid[1:end],3]
    cv_vals = pmap(x->cv(data[:,3], data[:,1], x ,h), y_vec)
    println("h: ", h, " value: ", sum(cv_vals)  )
    return sum(cv_vals)
end

cv_int([1.2])
Optim.optimize(cv_int, 0.0001, 1.0)


function cv_y(y_data, x_data,  y, h)
    cv=0.0
    for i = 1:n_firms
        cv += ((y_data[i] < y) - est_cdf(y_data[1:end .!= i], x_data[1:end .!= i],  y, x_data[i], h) )^2 * pdf(y_den, x_data[i])
    end
    # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    return n_firms^(-1)*cv
end

function cv_int_y(h)
    y_grid = 1:10:1500
    y_vec  = data[y_grid[1:end],4]
    cv_vals = pmap(x->cv_y(data[:,4], data[:,2], x ,h), y_vec)
    println("h: ", h, " value: ", sum(cv_vals)  )
    return sum(cv_vals)
end



# cv_int_y([1.2,0.2])
# res_y = Optim.optimize(cv_int_y, [1.,1.])

# h_down = Optim.minimizer(res_y)
pdf(x_den, 0.5)
data
data=hcat(data, zeros(n_firms, 2))
for i = 1:n_firms
    data[i,7] = est_cdf_step(data[:,3], data[:,1], data[i,3], data[i,1] ,0.02)
    data[i,8] = est_cdf_step(data[:,4], data[:,2], data[i,4], data[i,2] , .02)
end


# scatter(data[:,7], data[:,8], markersize = 1.5 )
# scatter(data[:,5], data[:,6], markersize = 1.5 )

# sig_up = [1. .2; 0 .2]
# sig_down = [0.5 .25; -1 .5]


# scatter((data[:,7]), data[:,8], markersize = 1.5 )

scatter((data[:,5]), (data[:,6]), markersize = 4., xlims=(.5, 2.), color=:red)
scatter!((quantile.(LogNormal(0.,0.2),data[:,7])),(quantile.(LogNormal(-1, 0.5),data[:,8])), markersize=3.,xlims=(.5, 2.), color=:yellow)
eps_vec = (quantile.(LogNormal(0.,0.2),data[:,7]))
eta_vec = (quantile.(LogNormal(-1, 0.5),data[:,8]))
A_mat = b_down + b_up
profs=diag(hcat(data[:,1], eps_vec) * A_mat * Transpose(hcat(data[:,2], eta_vec)))
scatter(data[:,3]+data[:,4], profs)







hcat(data[1,1], eps_vec[1]) * A_mat * Transpose(hcat(data[:,2], eta_vec))

# Objective function: Argument is the vector of parameters beta
# It solves for the market with the realized characteristics,
# This means using the quantiles that are implied from the profits.
function md_objective(b)
    # b_up = [b[1] 0.5; 1. 0.2]
    # b_down = [b[2] 0.2; 2. 0.3]
    A_mat = [b[1] b[2]; b[3] b[6]]

    sig_up = [1. .2; 0 .2]
    sig_down = [0.5 .25; -1 .5]
    # ud, dd, up, dp,down =  sim_data_LP(b_up,b_down,sig_up,sig_down,1500,2)
    eps_vec = quantile.(LogNormal(0.,abs(b[4])),data[:,7])
    eta_vec = quantile.(LogNormal(-1, abs(b[5])),data[:,8])
    surplus_vec = diag(hcat(data[:,1], eps_vec) * A_mat * Transpose(hcat(data[:,2], eta_vec)))
    total_profit = data[:,3]+ data[:,4]
    return sum((total_profit - surplus_vec).^2)
end



res= Optim.optimize(md_objective, rand(6))

Optim.minimizer(res)







A_mat
