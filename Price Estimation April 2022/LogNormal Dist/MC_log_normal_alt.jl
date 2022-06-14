### Monte-Carlo -- NPSML estimator -- Normally Distributed
### Simulation using finite market approximation. 
# package for parallel computation
using Distributed
using BSON
# using FLoops
addprocs(24)    # Cores  (This is for #444 Mac Pro)
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
    include("JV_DGP-LogNormal.jl")
    using Evolutionary
    # include("LP_DGP.jl")
end

@everywhere begin 
    n_reps = 24 # Number of replications (fake datasets)
    # n_sim =25
    true_pars = [-2.5, 1.5, -.5, -.5, 3.5, 1.5, 1.5, 1, 1, 3.]
end




@everywhere function replicate_byseed(n_rep, n_firms, n_sim)

    Σ_up = [0 .1;
            0 .2;
            0 .1]


    Σ_down =  [0 .3;
               0 .4;
               0 .1]

    #      [β11u, β12u, β21u, β11u, β11d, β12d, β21u, β13u, β33d]


    function par_gen(b)
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

        return bup, bdown
    end


    bup, bdown = par_gen(true_pars)
    up_data, down_data, price_data =
        sim_data_JV_LogNormal(bup, bdown, Σ_up, Σ_down, n_firms, 38+n_rep, false, 0, 0, 3.)
    # println("hi after fake data")
    # mean of transfers in the data
    # mu_price = mean(price_data)



    # # h: vector of bandwidths
    # # function to be minimized over the choice of h
    # # function uses the fake data above
    function bcv2_fun(h, down_data, price_data)
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
        val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1) +
                                ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
        # println("band: ",h," val: ", val)
        return val
    end

    # # only use a sample of size of the nsims not the total observed sample 
    # inds = rand(1:n_firms, n_sim)
    inds = 1:n_firms
    # # Optimize over choice of h
    res_bcv = Optim.optimize(x->bcv2_fun(x,down_data[1:2,inds],price_data[inds]), [0.1,.1,.1])
    # # res_bcv = Optim.optimize(bcv2_fun, [0.1,.1,.1])
    @show h = abs.(Optim.minimizer(res_bcv))
    # h =[0.01, 0.2, 0.07]
    # # Silverman
    # m=3
    # S=cov(hcat(down_data[1,:], down_data[2,:], price_data))
    # H_Silverman = (4/(n_sim*n_firms*(m+2)))^(2/(m+4)) * S
    # @show h= 0.25 .* sqrt.(diag(H_Silverman))

    # println("hi before function")
    println("hi", n_sim)
    function loglike(b)
        # n_sim=50
        
    
        bup = [
            vcat(b[1:2], b[8])';
            vcat(b[3:4], 0.)';
            vcat(0 , 0, 0)'
        ]
    
        bdown = [
            vcat(b[5], b[6],0)';
            vcat(b[7], 0, 0)';
            vcat(0 ,0., b[9] )'
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
    
        solve_draw =  x->sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360+x, true, up_data[1:2,:],3.)
    
        sim_dat = pmap(solve_draw, 1:n_sim)
        # sim_dat = solve_draw.(1:n_sim)
        # sim_dat = solve_draw.(1:n_sim)
        # sim_dat = []
        # for i = 1:n_sim
        #     push!(sim_dat, solve_draw(i))
        # end



        

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
        if mod(time(),10)<0.05
            println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        end

        return -ll/n_firms
    end

    # # return loglike(vcat(true_pars, 1))
    # res_CMAE = CMAEvolutionStrategy.minimize(loglike, rand(9), 1.,
    #     lower = nothing,
    #     upper = nothing,
    #     noise_handling = nothing,
    #     callback = (object, inputs, function_values, ranks) -> nothing,
    #     parallel_evaluation = false,
    #     multi_threading = false,
    #     verbosity = 1,
    #     seed = rand(UInt),
    #     maxtime = (n_firms/100)*1000,
    #     maxiter = nothing,
    #     maxfevals = 20000,
    #     ftarget = nothing,
    #     xtol = nothing,
    #     ftol = 1e-4)
    # res = Evolutionary.optimize(loglike, rand(9), CMAES())
    # return Evolutionary.minimizer(res), Evolutionary.minimum(res)



    # # # Estimated parameters: 
    # println("Best Cand:  " ,xbest(res_CMAE))
    # return xbest(res_CMAE), fbest(res_CMAE), h 
    # opt = bbsetup(loglike; SearchRange = bbo_search_range, NumDimensions =bbo_ndim,  Method = :simultaneous_perturbation_stochastic_approximation, MaxTime = bbo_max_time)

    # bbsolution = bboptimize(opt)


    bbo_search_range = (-5,5)
    bbo_population_size =50
    bbo_max_time=100
    bbo_ndim = 10
    bbo_feval = 10000

    opt = bbsetup(loglike; SearchRange = bbo_search_range, 
      NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
      Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
      TraceInterval=10.0, TraceMode=:compact)


    bbsolution1 = bboptimize(opt) 
    return bbsolution1
end

replicate_byseed(2, 100,25) 

# Parameter estimates 
for n_sim =25:25:25
    for n_firms = 100:100:100
        est_pars = pmap(x->replicate_byseed(x, n_firms, n_sim),1:n_reps)
        estimation_result = Dict()
        push!(estimation_result, "beta_hat" => est_pars)
        bson("LogNormal Dist/MC/02/MC_nf_$(n_firms)_sim_$(n_sim).bson", estimation_result)
    end
end

 /Users/akp/github/NPSML-Estimator-Price-Data/Price Estimation April 2022/LogNormal Dist/MC/01/MC_nf_100_sim_25.bson

interrupt()

########################################
########################################
##### Comparing the results ############
########################################
########################################


# est_matrix = Array{Float64, 3}(undef, 5, n_reps, 10)



MC_out = BSON.load("LogNormal Dist/MC/01/MC_nf_$((1)*100)_sim_25.bson")
MC_out = MC_out["beta_hat"]
MC_est= [MC_out[i][1] for i = 1:n_reps ]
MC_fit= [MC_out[i][2] for i = 1:n_reps ]

MC_est= reduce(vcat, MC_est')


MC_out2 = BSON.load("LogNormal Dist/MC/02/MC_nf_$((1)*100)_sim_25.bson")
MC_out2 = MC_out2["beta_hat"]
MC_est2= [MC_out2[i][1] for i = 1:n_reps ]
MC_fit2= [MC_out2[i][2] for i = 1:n_reps ]

MC_est2= reduce(vcat, MC_est2')

hcat(MC_est2, MC_fit2 )



sum(MC_fit2.<0)


println(mean(estimates, dims =2))
best_candidate(tst["beta_hat"][1])
for j = 1:2
    for i = 1:n_reps
        tmp_est = BSON.load("LogNormal Dist/MC/01/MC_nf_$((j)*100)_sim_25.bson")
        est_matrix[j,i,:] = vcat(tmp_est["beta_hat"][i][1][:], tmp_est["beta_hat"][i][2])
        est_matrix[j,i,8:9]=abs.(est_matrix[j,i,8:9])
    end
end


mse_vec = zeros(5, 10)
bias_vec = zeros(5,10)
true_pars[8:9] = abs.(true_pars[8:9])
true_pars= vcat(true_pars, 0 )

for j = 1:5 
    for i = 1: n_reps
        mse_vec[j,:] += (est_matrix[j,i,:] - true_pars).^2/n_reps
        bias_vec[j,:] += (est_matrix[j,i,:] - true_pars)/n_reps
        est_matrix[j,i,8:9]=abs.(est_matrix[j,i,8:9])
    end
    
end

sqrt.(mse_vec)
bias_vec



############# Spec 2

est_matrix2 = Array{Float64, 3}(undef, 3, n_reps, 9)


for j = 1:3
    for i = 1: n_reps
        tmp_est, tpar = BSON.load("NormalDist/MC/MC_half_$(j*100)_sim_$(25 +0*(j-1)*25).bson")
        est_matrix2[j,i,:] = tmp_est[2][i][:]
        est_matrix2[j,i,8:9]=abs.(est_matrix[j,i,8:9])
    end
end


mse_vec2 = zeros(5, 9)
bias_vec2 = zeros(5,9)
true_pars[8:9] = abs.(true_pars[8:9])

for j = 1:3
    for i = 1: n_reps
        mse_vec2[j,:] += (est_matrix2[j,i,:] - true_pars).^2/n_reps
        bias_vec2[j,:] += (est_matrix2[j,i,:] - true_pars)/n_reps
    end
    
end

sqrt.(mse_vec2)
bias_vec2















function est_performance(est_matrix, true_pars)
    rmse = zeros(9)
    bias = zeros(9)
    est_matrix[:,8:9] = abs.(est_matrix[:,8:9])

    true_pars[8:9] = abs.(true_pars[8:9])
    for i = 1:n_reps
        rmse += (true_pars - est_matrix[i,:]).^2/n_reps 
        bias += (true_pars - est_matrix[i,:])/n_reps
    end
    rmse = sqrt.(rmse)
    return rmse, bias

end



rmse50, bias50 = est_performance(est_matrix, true_pars)
rmse50_300, bias50_300 = est_performance(est_matrix, true_pars)

scatter(est_matrix[:,1])


mean(est_matrix[:,7])
bias50




#######

est_matrix= BSON.load("NormalDist/MC/02/MC_nf_400_sim_75.bson")

opt_vec = est_matrix["beta_hat"]


[opt_vec[i][:] for i = 1:24]

best_candidate.(opt_vec[:])
best_fitness.(opt_vec[:])













##### Testing bboptimize 



function tst_fun(k)
    eval = x -> sin(x[1])+ sqrt(x[1]*cos(x[2])
    vals = map