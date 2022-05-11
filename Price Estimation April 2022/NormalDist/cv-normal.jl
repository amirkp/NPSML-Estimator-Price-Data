############## Cross validation for the normal case
using Distributed
using FLoops
addprocs(23)    # Cores -1 (This is for #444 Mac Pro)

@everywhere using Optim    # Nelder-Mead Local Optimizer
@everywhere using CMAEvolutionStrategy # Global Optimizer

@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
    using Plots
    using Assignment
    using BenchmarkTools
    using InvertedIndices
    include("JV_DGP-Normal.jl")
end




######################################
######################################
######################################
#### GENERATING THE FAKE DATASET #####
######################################
######################################
######################################


## FAKE DATA SET IS DEFINED ON ALL WORKERS
# IN CASE NEEDED IN LATER VERSIONS OF THE ESTIMATOR
@everywhere begin
        n_firms=200

        # Upstream coefficients
         bup = [
                     -1.     1.5    -1;
                     .5     2.5     0;
                      0      0      0
                            ]
        # Downstream coefficients
        bdown = [
                    2.5      -2      0;
                    1         0      0;
                    0         0     -1.5
                            ]

        # Matrix of total production coefficients
        B = bup+bdown


        # Parameters of the distribution of observed and unobserved types
        # First column: μ     Second column: σ^2
        sigup =     [
                        0 2.;
                        0 1.;
                        0 1.
                    ]


        sigdown = [
                        0 2.5;
                        0 3.;
                        0 1.
                    ]


    # The last three args are used when simulating markets with pre-determined observed types
    # i.e. The observed types are passed on as arguments, only the unobservables are drawn from the distribution
    up_data, down_data, price_data =
        sim_data_JV_Normal(bup, bdown, sigup, sigdown, n_firms, 23, false, 0, 0)

    # mean of transfers in the data
    mu_price = mean(price_data)
end

###########################################################
###########################################################
####### BANDWIDTH/TUNING PARAMETERS SELECTION #############
###########################################################
###########################################################

# This part of the code deals with bandwidth selection
# At the moment it uses bcv2 criterion function
# Reference: Sain, S. R., Baggerly, K. A., & Scott, D. W. (1994). Cross-validation of multivariate densities. Journal of the American Statistical Association,
##################################
##################################


# h: vector of bandwidths
# function to be minimized over the choice of h
# function uses the fake data above
function bcv2_fun(h)
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
res_bcv = Optim.optimize(bcv2_fun, rand(3))
# res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)
h = abs.(Optim.minimizer(res_bcv))


### alternative candidate for bandwidth 
### we just choose it as twice res_bcv
h_alt = 2*h
# #################################
# #################################
# ######### Silverman #############
# #################################

# n_sim =50
# m=3
# S=cov(hcat(down_data[1,:], down_data[2,:], price_data))
# H_Silverman = (4/(n_sim*(m+2)))^(2/(m+4)) * S
# @show h= sqrt.(diag(H_Silverman))




# generating hold out samples 
k_fold = 10

hold_id = sample(1:n_firms, n_firms, replace = false)

# Each row represents the id of the firms that are held out 
hold_id = reshape(hold_id, (k_fold,Int(n_firms/k_fold))  )


################################################
################################################
################################################
################################################
#### MAIN LIKELIHOOD FUNCTION ##################
################################################
################################################
################################################

function loglike(b,hold_vec,h)
    n_sim=25
    bup = [
        vcat(b[1:2],b[8])';
        vcat(b[3:4], 0.)';
        vcat(0 , 0, 0)'
    ]


    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., b[9] )'
     ]
     sig_up = [0 2.;
                 0 1.;
                 0 1.]


     sig_down = [0 2.5;
                 0 3.;
                 0 1.]


    solve_draw =  x->sim_data_JV_Normal(bup, bdown , sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:], down_data[1:2,:])

    sim_dat = pmap(solve_draw, 1:n_sim)
    ll=0.0


    for j=1:n_sim
        pconst = mean(sim_dat[j][3][Not(hold_vec)])-mu_price
        sim_dat[j][3][:] = sim_dat[j][3] .+ pconst
    end
    n_zeros = 0
    for i =1:n_firms
        if !in(hold_vec).(i) 
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
                # println("Like is zero!!!")
                ll+= -n_firms
                n_zeros += 1
            else
                ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
            end
        end
    end
    println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    return -ll/(n_firms-length(hold_vec))
end


# Vector of true parameters

tpar = [-1, 1.5, .5, 2.5, 2.5, -2, 1, -1, .5 ]

# Log-likelihood value at the truth
loglike(tmp,[],h)
(loglike(tpar,hold_id[1,:]) + loglike(tpar,hold_id[2,:]))/2
loglike(tpar,hold_id[2,:])

loglike_nohold = x->loglike(x, hold_id[2,:],h)
loglike_nohold = x->loglike(x, [],h)
# loglike_nohold = x->loglike(x, hold_id[2,:],h)
# res_CMAE = CMAEvolutionStrategy.minimize(loglike_nohold,rand(9), 1.,
#         lower = nothing,
#          upper = nothing,
#          noise_handling = 1.,
#          callback = (object, inputs, function_values, ranks) -> nothing,
#          parallel_evaluation = false,
#          multi_threading = false,
#          verbosity = 1,
#          seed = rand(UInt),
#          maxtime = 10000,
#          maxiter = nothing,
#          maxfevals = nothing,
#          ftarget = nothing,
#          xtol = nothing,
#          ftol = 1e-3)



# # Estimated parameters: 
# est_pars = xbest(res_CMAE)


# res_NM = Optim.optimize(loglike, est_pars)

bbo_search_range = (-5,5)
bbo_population_size =50
SMM_session =1
bbo_max_time=22000
bbo_ndim = 9
opt6 = bbsetup(loglike_nohold; 
    SearchRange = bbo_search_range, NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
    Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = bbo_max_time,
    TraceMode=:verbose)


bbsolution1 = bboptimize(opt4)

#holdout1 
bbsolution5 = bboptimize(opt5)

#holdout 2
bbsolution6 = bboptimize(opt6)

bbsolution2 = bboptimize(opt2)

bbsolution3 = bboptimize(opt3)
pars1 = best_candidate(bbsolution5)
pars2 = best_candidate(bbsolution6)

Optim.optimize(loglike_nohold, best_candidate(bbsolution1))




function fc_err(b,hold_vec,h)
    n_sim=25
    bup = [
        vcat(b[1:2],b[8])';
        vcat(b[3:4], 0.)';
        vcat(0 , 0, 0)'
    ]


    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., b[9] )'
     ]
     sig_up = [0 2.;
                 0 1.;
                 0 1.]


     sig_down = [0 2.5;
                 0 3.;
                 0 1.]


    solve_draw =  x->sim_data_JV_Normal(bup, bdown , sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:], down_data[1:2,:])

    sim_dat = pmap(solve_draw, 1:n_sim)
    ll=0.0


    for j=1:n_sim
        pconst = mean(sim_dat[j][3][Not(hold_vec)])-mu_price
        sim_dat[j][3][:] = sim_dat[j][3] .+ pconst
    end
    for i =1:n_firms
        if in(hold_vec).(i) 
            like =0.
            for j =1:n_sim
                like+=(
                    (down_data[1,i] - sim_dat[j][2][1,i])^2 
                    +(down_data[2,i] - sim_dat[j][2][2,i])^2
                    +(price_data[i] - sim_dat[j][3][i])^2
                    )
            end
            # println("like is: ", like, " log of which is: ", log(like/(n_sim*h[1]*h[2]*h[3])))
            ll+=like/n_sim
        end
    end
    println("parameter: ", round.(b, digits=3), " function value: ", -ll)
    return ll
end



fc_err(tpar, hold_id[1,:],h)

fc_err(pars1, hold_id[1,:],h)
fc_err(pars2, hold_id[2,:],h)

