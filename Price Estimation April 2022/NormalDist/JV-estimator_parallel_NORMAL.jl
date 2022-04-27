#################### ESTIMATING MATCHING MODEL WITH PRICE DATA ######################
#####################################################################################
## AMIR KAZEMPOUR ############################ APRIL 2022 ###########################
#####################################################################################
# THE CODE GENERATES A FAKE DATASET OF {MATCHES,PRICES} ACCORDING TO THE NORMAL DISTRIBUTION ASSUMPTION
# THEN, USING THE FAKE DATA SET OF OBSERVED MATCH CHARACTRISTICS AND TRANSFERS, THE ESTIMATOR SHOULD RECOVER
# THE TRUE PARAMETERS OF THE DGP
############################################################################################################
############################################################################################################
############################################################################################################
# package for parallel computation
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
                     1.     1.5    -1;
                     .5     2.5     0;
                      0      0      0
                            ]
        # Downstream coefficients
        bdown = [
                    2.5      -2      0;
                    1         0      0;
                    0         0     .5
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
    println("band: ",h," val: ", val)
    return val
end

# bcv2_fun([-.1, 1.0, 1.])


# Optimize over choice of h
res_bcv = Optim.optimize(bcv2_fun, rand(3))
# res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)
h = 2*abs.(Optim.minimizer(res_bcv))





################################################
################################################
################################################
################################################
#### MAIN LIKELIHOOD FUNCTION ##################
################################################
################################################
################################################

function loglike(b)
    n_sim=50
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
     # x->sim_data_like(up_data[1:2,:],bup, bdown , [2, 1., 1], [2.5, 3, 1], n_firms, 1234+x, 2.5)
    sim_dat = pmap(solve_draw, 1:n_sim)
    ll=0.0
    # mu_price = b[10]

    for j=1:n_sim
        pconst = mean(sim_dat[j][3])-mu_price
        sim_dat[j][3][:] = sim_dat[j][3] .+ pconst
    end
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
            # println("Like is zero!!!")
            ll+= -n_firms
            n_zeros += 1
        else
            ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
        end


    end
    println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    return -ll/n_firms
end


# Vector of true parameters
tpar = [1, 1.5, .5, 2.5, 2.5, -2, 1, -1, .5 ]

# Log-likelihood value at the truth
loglike(tpar)



res_CMAE = CMAEvolutionStrategy.minimize(loglike, rand(9), 1.,
        lower = nothing,
         upper = nothing,
         noise_handling = nothing,
         callback = (object, inputs, function_values, ranks) -> nothing,
         parallel_evaluation = false,
         multi_threading = false,
         verbosity = 1,
         seed = rand(UInt),
         maxtime = 1000,
         maxiter = nothing,
         maxfevals = nothing,
         ftarget = nothing,
         xtol = nothing,
         ftol = 1e-3)



# Estimated parameters: 
est_pars = xbest(res_CMAE)


res_NM = Optim.optimize(loglike, est_pars)













##################################################
##################################################
##################################################
############ Estimator Second Approach ###########
##################################################
##################################################
##################################################



@everywhere function bcv2_fun(down_data, price_data, h)
    h=abs.(h)
    ll = 0.0
    n_firms = length(price_data)
    for i = 1:n_firms
        for j=1:n_firms
            if (j!=i)
                expr_1 = ((down_data[1,i]-down_data[1,j])/h[1])^2 + ((down_data[2,i]-down_data[2,j])/h[2])^2 + ((price_data[i]-price_data[j])/h[3])^2
                expr_2 = pdf(Normal(),(down_data[1,i]-down_data[1,j])/h[1]) * pdf(Normal(),((down_data[2,i]-down_data[2,j])/h[2])) * pdf(Normal(),((price_data[i]-price_data[j])/h[3]))
                ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
            end
        end
    end
    val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
    # println("band: ",h," val: ", val)
    return val
end

function loglike_varh(b)
    n_sim=50

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

    sig_up = [  0 2.;
                0 1.;
                0 1.
                ]


    sig_down = [0 0.5;
                0 3.;
                0 1.
            ]


    solve_draw= x->sim_data_JV_Normal(bup, bdown , sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:], down_data[1:2,:])


  

    sim_dat = pmap(solve_draw, 1:n_sim)

    # function bcv2_fun(k,  h)
    #     h=abs.(h)
    #     ll = 0.0
    #     n_firms = n_sim
    #     for i = 1:n_firms
    #         for j=1:n_firms
    #             if (j!=i)
    #                 expr_1 = ((sim_dat[k][2][1,i]-sim_dat[k][2][1,j])/h[1])^2 + ((sim_dat[k][2][2,i]-sim_dat[k][2][2,j])/h[2])^2 + ((sim_dat[k][3][i]-sim_dat[k][3][j])/h[3])^2
    #                 expr_2 = pdf(Normal(),(sim_dat[k][2][1,i]-sim_dat[k][2][1,j])/h[1]) * pdf(Normal(),((sim_dat[k][2][2,i]-sim_dat[k][2][2,j])/h[2])) * pdf(Normal(),((sim_dat[k][3][i]-sim_dat[k][3][j])/h[3]))
    #                 ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
    #             end
    #         end
    #     end
    #     val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
    #     # println("band: ",h," val: ", val)
    #     return val
    # end
    h_vec = Array{Float64, 2}(undef, 3, n_firms)

    for i = 1:n_firms
        res_bcv = Optim.optimize(h -> bcv2_fun(hcat([sim_dat[j][2][1,i] for j=1:n_sim], [sim_dat[j][2][2,i]  for j=1:n_sim])', [sim_dat[j][3][i] for j=1:n_sim], h),
                 .01*ones(3), g_tol=1e-4)
        h_vec[:,i] = abs.(Optim.minimizer(res_bcv))
        if mod(i,50) == 0
            println("i=",i, " band: ", h_vec[:,i])
        end

        # h_vec[:,i]=[0.01,0.01,0.5]
    end





    ll=0.0

    for j=1:n_sim
        pconst = mean(sim_dat[j][3])-mu_price
        sim_dat[j][3][:] = sim_dat[j][3] .+ pconst
    end
    
    n_zeros = 0
    for i =1:n_firms
        like =0.
        for j =1:n_sim
            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h_vec[1,i]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h_vec[2,i]))
                *pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h_vec[3,i]))
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
    println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    return -ll/n_firms
end





loglike_varh(tpar)
loglike_varh(rand(9))
loglike_varh([-12.82, -4.914, 29.151, 18.431, 17.161, -13.677, 4.35, 29.476, -50.488])



res_est=Optim.optimize(loglike_varh, tpar)
rest_est2 = Optim.optimize(loglike_varh, rand(9))



res_CMAE_vh = CMAEvolutionStrategy.minimize(loglike_varh, rand(9), 1.,
        lower = nothing,
         upper = nothing,
         noise_handling = nothing,
         callback = (object, inputs, function_values, ranks) -> nothing,
         parallel_evaluation = false,
         multi_threading = false,
         verbosity = 1,
         seed = rand(UInt),
         maxtime = 20000,
         maxiter = nothing,
         maxfevals = nothing,
         ftarget = nothing,
         xtol = nothing,
         ftol = 1e-3)



# Estimated parameters: 
est_pars = xbest(res_CMAE_vh)




@benchmark begin 
    i=rand(1:20,1)[1]
    tfun = h ->bcv2_fun(hcat([sim_dat[j][2][1,i] for j=1:n_sim], [sim_dat[j][2][2,i]  for j=1:n_sim])', [sim_dat[j][3][i] for j=1:n_sim], h)
    tmres = Optim.optimize(tfun, [.01,.01,.01])
    tres = Optim.minimizer(tmres)
end
@benchmark tmres = Optim.optimize(tfun, [.01,.01,.01])
tmres = Optim.optimize(tfun,  0.001* ones(3), g_tol=1e-4)
tres = Optim.minimizer(tmres)







bcv2_fun(hcat([sim_dat[j][2][1,1] for j=1:n_sim], [sim_dat[j][2][2,1]  for j=1:n_sim])', [sim_dat[j][3][1] for j=1:n_sim], .5*[.2,.2,.2])

scatter([sim_dat[j][2][1,1] for j=1:n_sim],[sim_dat[j][2][2,1]  for j=1:n_sim])


scatter([sim_dat[j][2][1,1] for j=1:n_sim],[sim_dat[j][3][1] for j=1:n_sim])









n_firms=2000

# Upstream coefficients
bup = [
        1.     1.5    -1;
        .5     2.5     0;
        0      0      0
            ]
# Downstream coefficients
bdown = [
    2.5      -2      0;
    1         0      0;
    0         0     .5
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
    sim_data_JV_Normal(bup, bdown, sigup, sigdown, n_firms, 200, false, 0, 0)



include("data_sim_like_2d_2d_diff.jl")
up_data1, down_data1, price_data1 =
    sim_data_like(1, bup, bdown, [2., 1., 1.], [2.5, 3, 1.], n_firms, 200)



p1= scatter(up_data[1,:], down_data[1,:])
p2 = scatter!(up_data1[1,:], down_data1[1,:], markersize = 3, color=:red)
plot(p1,p2, markersize=1, legends = false)




p1= scatter(up_data[1,:], price_data)
p2 = scatter!(up_data1[1,:], price_data1, markersize = 3, color=:red)