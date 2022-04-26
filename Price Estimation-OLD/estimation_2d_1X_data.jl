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

n_firms = 1054

# data with HHI =1
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


# x, y_1, y_2 , pi_u,pi_d,price
scatter(data_HHIn1[:,3],data_HHIn1[:,4], markersize=2)
scatter(data_HHI1[:,3],data_HHIn1[:,], markersize=2)

cor(data_HHIn1[:,3], data_HHIn1[:,4])
cor(data[:,3], data[:,4])
A_mat = rand(2,3)
bx = 1.
by = rand(2)
Σ_up = rand(3,3)
Σ_down = rand(3,3)

i =10
up_ = data_HHIn1[:,1]
down_ = data_HHIn1[:,2:3]'
n_firms = size(data_HHIn1)[1]

simtst = x ->sim_data_LP(up_, down_, A_mat,bx, by, Σ_up, Σ_down, n_firms,x)
res_ = map(simtst, 1:10)
res_[1][2][1,2]

t = (1
    + 1)

t

sim_data_LP(up_, down_, A_mat,bx, by, Σ_up, Σ_down, n_firms,i)

# Let's first get the estimator to work and then try some bw selection
function loglikep(b)
    n_sim=5
    b_up = zeros(2,3)
    b_up[1,:]= [b[1] b[2] b[3]]
    b_up[2,3] = b[4]

    b_down = zeros(2,3)
    b_down[1,:] = [b[5] b[6] 0]
    b_down[2,:] = [0 b[7] b[8]]
    A_mat = b_up + b_down
    Σ_up = [0, abs(b[9])]; Σ_down = [0,abs(b[10])]
    solve_draw = x->sim_data_LP(up_, down_, b_up, b_down,A_mat , Σ_up, Σ_down , n_firms,  1234+x)
    sim_dat = map(solve_draw, 1:n_sim)

    # return sim_dat
    ll=0.0
    mu_data = mean(data_HHIn1)
    for i =1:n_firms
        h=[0.5, 0.5, 0.5]
        like =0.
        for j =1:n_sim
            #normalize prices, by normalizing location of prices
            sim_dat[j][4][:] = sim_dat[j][4][:] .+ (mu_data - mean(sim_dat[j][4]))
            like+=(
            pdf(Normal(),((down_[1,i] - sim_dat[j][2][i])/h[1]))
            *pdf(Normal(),((down_[2,i] - sim_dat[j][3][i])/h[2]))
            *pdf(Normal(),((data_HHIn1[i, 4] - sim_dat[j][4][i])/h[3]))
            )
        end
        ll+=log(like/(n_sim*h[1]*h[2])*h[3])
    end

    println("parameter: ", b, " function value: ", -ll/n_firms)
    return -ll/n_firms
end

res = loglikep(ones(10))

opres = Optim.optimize(loglikep, ones(10))

105.68 without normalization
10.15












b=ones(10)
b[1] =-1.0
b[5] = 1.05

b[2] =0.05
b[6] = 0.05
b_up = zeros(2,3)
b_up[1,:]= [b[1] b[2] b[3]]
b_up[2,3] = b[4]

b_down = zeros(2,3)
b_down[1,:] = [b[5] b[6] 0]
b_down[2,:] = [0 b[7] b[8]]
A_mat = b_up + b_down
Σ_up = [0, abs(b[9])]; Σ_down = [0,abs(b[10])]
res =sim_data_LP(up_, down_, b_up, b_down,A_mat , Σ_up, Σ_down , n_firms,  1234)

scatter(res[1][1,:], res[2])
scatter(data_HHIn1[:,1], data_HHIn1[:,2])

scatter(res[1][1,:], res[3])
scatter(data_HHIn1[:,1], data_HHIn1[:,3])
pr = res[4] .+ (mean(data_HHIn1[:,4]) - mean(res[4]))

scatter(res[1][1,:], pr)

scatter(data_HHIn1[:,1], data_HHIn1[:,4])
data_HHIn1[:,1]data_HHIn1[:,1]
mean(data_HHIn1[:,4])

mean(res[4])


mean(pr)

m
res
