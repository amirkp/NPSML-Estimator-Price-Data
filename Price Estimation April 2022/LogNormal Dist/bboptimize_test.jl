# The goal in here is to verify whether bboptimize converges to the same global optimizer
# and how often 

using Distributed
using PrettyTables
using BSON

addprocs(2)
@everywhere using Optim
@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
    using Plots
    using Assignment
    using Gurobi
    using CMAEvolutionStrategy
    using JuMP
    using BenchmarkTools
    include("JV_DGP-LogNormal.jl")
    include("LP_DGP.jl")
end
# @benchmark sim_data_JV_LogNormal(bup, bdown, sig_up, sig_down, n_firms, 36, false, 0, 0)


#ceo disu for labor 

@everywhere begin
        n_firms=50

        bup = [-2.5 1.5 -3;
               -1.5 -.5 0;
              0 0  0 ]
        bdown = [3.5 2.5 0;
                1.5  0 0;
                0 0 3.]
        B= bup+bdown

    

        sig_up = [0 .1;
                    0 .2;
                    0 .1]
        sig_down = [0 .3;
                    0 .4;
                    0 .1]
    up_data, down_data, price_data, upr, dpr= sim_data_JV_LogNormal(bup, bdown, sig_up, sig_down, n_firms, 28, false, 0, 0,2.)
    # up1, down1, price1 =sim_data_LP(bup, bdown, sig_up, sig_down, n_firms,36)
    mu_price = mean(price_data)
    # mu_price1 = mean(price1)
    tpar = [-2.5, 1.5, -1.5, -.5, 3.5, 2.5, 1.5, -3, 3,2]
end

########################
########################
###### BANDWIDTH ######
########################
########################
########################

@everywhere function bcv2_fun(h)
    h=abs.(h)
    ll = 0.0
    for i = 1:n_firms
        for j=1:n_firms
            if (j!=i)
                expr_1 = ((down_data[1,i]-down_data[1,j])/h[1])^2 + ((down_data[2,i]-down_data[2,j])/h[2])^2 + ((price_data[i]-price_data[j])/h[3])^2
                expr_2 = pdf(Normal(),(down_data[1,i]-down_data[1,j])/h[1]) * pdf(Normal(),((down_data[2,i]-down_data[2,j])/h[2])) * pdf(Normal(),((price_data[i]-price_data[j])/h[3]))
                ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
            end
        end
    end
    val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1) +
                            ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
    # println("band: ",h," val: ", val)
    return val
end

# bcv2_fun([-.1, 1.0, 1.])


# Optimize over choice of h
@everywhere res_bcv = Optim.optimize(bcv2_fun, [0.1,0.1,0.1])
# res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)

@everywhere @show h = abs.(Optim.minimizer(res_bcv))



#################################
#################################
######### Silverman #############
#################################

# n_sim =50
# m=3
# S=cov(hcat(down_data[1,:], down_data[2,:], price_data))
# H_Silverman = (4/(n_sim*(m+2)))^(2/(m+4)) * S

# @show h= sqrt.(diag(H_Silverman))
# h = [.2,.2,.2]
# h = h/5
########################
##################
#### MAIN LIKELIHOOD FUNCTION
#############
######################
###################

@everywhere function loglike(b)
    n_sim=25


    bup = [
        vcat(b[1:2], (b[8]))';
        vcat(b[3:4], 0.)';
        vcat(0 , 0, 0)'
    ]

    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., (b[9]) )'
     ]

    solve_draw =  x->sim_data_JV_up_obs(bup, bdown , sig_up, sig_down, n_firms, 360+x, true, up_data[1:2,:],b[10])

    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0

    n_zeros = 0
    for i =1:n_firms
        like =0.
        for j =1:n_sim
            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                *pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h[3]))
                )
        end
        # println("like is: ", like, " log of which is: ", log(like/(n_sim*h[1]*h[2]*h[3])))
        if like == 0
        #     # println("Like is zero!!!")
            ll+= -n_firms
            n_zeros += 1
        else
            ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
            # ll+=like
        end

        
    end
    if mod(time(),10)<.1
        println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    end
    Random.seed!()
    # sleep(1)
    return -ll/n_firms
end

res=Optim.optimize(x->loglike(vcat(tpar[1:8], x)),rand(2))


Optim.minimizer(res)

Optim.optimize(x->loglike(vcat(tpar[1:9], x)), -10, 10)



@everywhere function opt_test(i)
    opt_res = zeros(11)
    bbo_search_range = (-10,10)
    bbo_population_size =50
    bbo_ndim = 3
    bbo_max_time=15*2^bbo_ndim
    fun = x->loglike(vcat(tpar[1:7],x))
    opt = bbsetup(fun; SearchRange = bbo_search_range, NumDimensions =bbo_ndim, PopulationSize = 50 ,
     Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = bbo_max_time, TraceInterval=10);
    #large bandwidth 
    sol = bboptimize(opt)
    # opt_res[1:10] = best_candidate(sol)
    # opt_res[1] = best_candidate(sol)
    # opt_res[11] = best_fitness(sol)
    opt_res = vcat(best_candidate(sol), best_fitness(sol))
    return opt_res
end

# opt_results = pmap(opt_test, 1:24)
# interrupt()
# res_mat = reduce(vcat, opt_results')
# pretty_table(res_mat)




opt_results_med = pmap(opt_test, 1:4)
res_mat_med = reduce(vcat, opt_results_med')
pretty_table(res_mat_med)

# estimation_result = Dict()
# push!(estimation_result, "beta_hat" => res_mat)
# push!(estimation_result, "beta_hat2" => res_mat_med)
# bson("/Users/akp/github/NPSML-Estimator-Price-Data/Price Estimation April 2022/LogNormal Dist/bbotest/opt_results24.bson", estimation_result)
