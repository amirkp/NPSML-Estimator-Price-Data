

using Distributed
addprocs(24)
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
        n_firms=200

        bup = [-2.5 1.5 1;
               -.5 -.5 0;
              0 0  0 ]
        bdown = [3.5 1.5 0;
                1.5  0 0;
                0 0 1.]
        B= bup+bdown

        sig_up = [0 .1;
                    0 .2;
                    0 .1]
        sig_down = [0 .3;
                    0 .4;
                    0 .1]
    up_data, down_data, price_data, upr, dpr= sim_data_JV_LogNormal(bup, bdown, sig_up, sig_down, n_firms, 38, false, 0, 0,3.)
    # up1, down1, price1 =sim_data_LP(bup, bdown, sig_up, sig_down, n_firms,36)
    mu_price = mean(price_data)
    # mu_price1 = mean(price1)
    tpar = [-2.5, 1.5, -.5, -.5, 3.5, 1.5, 1.5, 1, 1]
end



# scatter(up_data[1,:], down_data[3,:],markersize =3)
# scatter(up_data[1,:], down_data[3,:])
# scatter(up_data[1,:], price_data)
# scatter(up_data[1,:], up1[1,:])
scatter(up_data[2,:], price_data)
scatter(down_data[2,:], dpl)
# scatter(up1[1,:], down1[1,:])
# scatter(up_data[2,:], up1[2,:])
# scatter(down_data[1,:], down1[1,:])
# scatter(price_data, price1)

# scatter(up_data[1,:], down_data[2,:])
# scatter(up1[1,:], down1[1,:],color=:red)
scatter(up_data[1,:], dpr)
scatter(up_data[1,:], upl)

########################
########################
###### BANDWIDTH ######
########################
########################
########################

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
res_bcv = Optim.optimize(bcv2_fun, [0.1,0.1,0.1])
# res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)

h = abs.(Optim.minimizer(res_bcv))



#################################
#################################
######### Silverman #############
#################################

n_sim =50
m=3
S=cov(hcat(down_data[1,:], down_data[2,:], price_data))
H_Silverman = (4/(n_sim*(m+2)))^(2/(m+4)) * S

@show h= sqrt.(diag(H_Silverman))
h = [.2,.2,.2]

# h = h/5
########################
##################
#### MAIN LIKELIHOOD FUNCTION
#############
######################
###################
hx= [.01, .01]
function loglike(b)
    n_sim=20


    bup = [
        vcat(b[1:2],abs(b[8]))';
        vcat(b[3:4], 0.)';
        vcat(0 , 0, 0)'
    ]

    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., abs(b[9]) )'
     ]

    # bup = [
    #     vcat(b[1:2], 1.)';
    #     vcat(b[3:4], 0.)';
    #     vcat(0 , 0, 0)'
    # ]

    # bdown = [
    #     vcat(b[5], b[6],0)';
    #     vcat(b[7], 0, 0)';
    #     vcat(0 ,0., 1. )'
    #  ]

    solve_draw =  x->sim_data_JV_up_obs(bup, bdown , sig_up, sig_down, n_firms, 360+x, true, up_data[1:2,:],b[10])

    sim_dat = pmap(solve_draw, 1:n_sim)
    ll=0.0
    
    
    # h=0.01*[0.2, 0.4, 1.]


    n_zeros = 0
            like+=(
    # for i =1:n_firms
    #     like =0.
    #     for j =1:n_sim
    #         like+=(
    #             pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
    #             *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
    #             *pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h[3]))
    #             )
    #         # like+=(
    #         #     (down_data[1,i] - sim_dat[j][2][1,i])^2
    #         #     +(down_data[2,i] - sim_dat[j][2][2,i])^2
    #         #     +(price_data[i] - sim_dat[j][3][i])^2
    #         #     )
    #     end
        @floop for i =1:n_firms
            like =0.
            for k = 1:n_firms
                for j =1:n_sim
                    like+=(
                        pdf(Normal(),((up_data[1,i] - sim_dat[j][1][1,k])/hx[1]))
                        *pdf(Normal(),((up_data[1,i] - sim_dat[j][1][2,k])/hx[2]))
                        *pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,k])/h[1]))
                        *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,k])/h[2]))
                        *pdf(Normal(),((price_data[i] - sim_dat[j][3][k])/h[3]))
                        )

                end
            end
        

        # println("like is: ", like, " log of which is: ", log(like/(n_sim*h[1]*h[2]*h[3])))
        if like == 0
        #     # println("Like is zero!!!")
            ll+= -n_firms
            n_zeros += 1
        else
            ll+=log(like/(n_sim*h[1]*h[2]*h[3]*hx[1]*hx[2]))

            @tullio ll = (
                pdf(Normal(),((up_data[1,i] - sim_dat[j][1][1,k])/hx[1]))
                *pdf(Normal(),((up_data[1,i] - sim_dat[j][1][2,k])/hx[2]))
                *pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,k])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,k])/h[2]))
                *pdf(Normal(),((price_data[i] - sim_dat[j][3][k])/h[3]))
                )
            # ll+=like
        end


    end
    println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    return -ll/n_firms
end


@benchmark loglike(vcat(tpar,1.))
# tpar
# loglike(tpar[1:4])

# @benchmark loglike(rand(3))
xrange = 0.:0.05:9
# scatter(xrange, [loglike(vcat(tpar[1:9], xrange[i])) for i in 1:length(xrange)])
scatter(xrange, [loglike(vcat(tpar[1:7], xrange[i], tpar[9], 3)) for i in 1:length(xrange)])

scatter(xrange, [loglike(vcat(tpar[1:9], xrange[i])) for i in 1:length(xrange)])


stop
# plot(x->loglike(vcat(-2.5, x, -.5)), -2,2 )

Optim.optimize(x->loglike(vcat(x, tpar[2:9],3.)),-5,2)


@everywhere n_firms =200

bbo_search_range = (-8,10)
bbo_population_size =20
SMM_session =1
bbo_max_time=22000
bbo_ndim = 10

opt_20 = bbsetup(loglike;  SearchRange = bbo_search_range, NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = bbo_max_time)

#large bandwidth 

sol20 = bboptimize(opt_20)
best_candidate(sol20)
best_fitness(sol20)
# altpar = best_candidate(smaller_bw_sol1)

# [-3.016, 0.882, -0.67, -0.111, 3.151, -0.121, 0.803, 0.042, -0.216, 1.357] function value: -2.6738246697409935

# [-3.033628667531997, 0.8226415225489178, -0.6353453347920984, -0.0817466294926248, 3.2314522185925916, -0.33718104808090443, 0.8151541380850537, 0.05546259388310969, -0.04389629889560204, 1.159638109739515]




res_CMAE = CMAEvolutionStrategy.minimize(loglike, vcat(tpar,1.), 1.,
    lower = nothing,
    upper = nothing,
    noise_handling = nothing,
    callback = (object, inputs, function_values, ranks) -> nothing,
    parallel_evaluation = false,
    multi_threading = false,
    verbosity = 1,
    seed = rand(UInt),
    maxtime = (n_firms/100)*2000,
    maxiter = nothing,
    maxfevals = nothing,
    ftarget = nothing,
    xtol = nothing,
    ftol = 1e-6)

# bbsolution1 = bboptimize(opt1)
# best_candidate(bbsolution1)














M = rand(1:20, 3, 7)

@tullio S := log(M[r,c]*M[r,1])  # sum over r ∈ 1:3, for each c ∈ 1:7
sum(log.(M))