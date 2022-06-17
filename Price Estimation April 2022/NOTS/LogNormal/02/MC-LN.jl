### Monte-Carlo -- NPSML estimator -- Normally Distributed
### Simulation using finite market approximation. 
# package for parallel computation

using Distributed, ClusterManagers
pids = addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
# println(pids)


using BSON
# using FLoops
  
# @everywhere using Optim    # Nelder-Mead Local Optimizer
# @everywhere using CMAEvolutionStrategy # Global Optimizer

@everywhere begin
    using LinearAlgebra
    using Random
    # using Evolutionary
    using Distributions
    using BlackBoxOptim
    # using Plots
    using Assignment
    # using BenchmarkTools
    using Optim
    using CMAEvolutionStrategy
    include("JV_DGP-LogNormal.jl")
end

@everywhere begin 
    n_reps =100  # Number of replications (fake datasets)
    true_pars = [-2.5, 1.5, -1.5, -.5, 3.5, 2.5, 1.5, -3, 3, -3.]
    Σ_up = [0 .1;
        0 .2;
        0 .1]


    Σ_down =  [0 .3;
               0 .4;
               0 .1]
    
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
               
end

@everywhere function replicate_byseed(n_rep, n_firms, n_sim, h_scale)

 

    #      [β11u, β12u, β21u, β11u, β11d, β12d, β21u, β13u, β33d]


    up_data, down_data, price_data =
        sim_data_JV_LogNormal(bup, bdown, Σ_up, Σ_down, n_firms, 38+n_rep, false, 0, 0, true_pars[10])



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
    inds = 1:n_firms;
    # # Optimize over choice of h
    res_bcv = Optim.optimize(x->bcv2_fun(x,down_data[1:2,inds],price_data[inds]), [0.1,.1,.1]);

    
    @show h = abs.(Optim.minimizer(res_bcv))
    
    
    
    function loglike(b,h)
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
    
        solve_draw =  x->sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360+x, true, up_data[1:2,:],b[10])
    
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
            println("I'm worker number $(myid()) on thread $(Threads.threadid()), and I reside on machine $(gethostname()).")

            println(" parameter: ", round.(b, digits=4), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        end
        return -ll/n_firms
    end
    # println("worker id: ", myid())
    # return loglike(true_pars)



    # res = Evolutionary.optimize(
    #     loglike, zeros(10),
    #     GA(populationSize = 100, selection = susinv,
    #        crossover = DC, mutation = PLM()))
    # println("min ", Evolutionary.minimizer(res))
    # return 1 
    # # return loglike(vcat(true_pars, 1))
    println("loglike of true pars: ", loglike(true_pars,h))
    init_res_CMAE =  CMAEvolutionStrategy.minimize(x->loglike(x,10*h),ones(10) , 1.,
        lower = -10*ones(10),
        upper = 10*ones(10),
        noise_handling =nothing,
        callback = (object, inputs, function_values, ranks) -> nothing,
        parallel_evaluation = false,
        multi_threading = false,
        verbosity = 1,
        seed = rand(UInt),
        maxtime = 100,
        maxiter = nothing,
        maxfevals = 30000,
        ftarget = nothing,
        xtol = 1e-5,
        ftol = 1e-5)
    
    
    
    res_CMAE =  CMAEvolutionStrategy.minimize(x->loglike(x,h), xbest(init_res_CMAE) , 1.,
        lower = -10*ones(10),
        upper = 10*ones(10),
        noise_handling = CMAEvolutionStrategy.NoiseHandling(2.1),
        callback = (object, inputs, function_values, ranks) -> nothing,
        parallel_evaluation = false,
        multi_threading = false,
        verbosity = 1,
        seed = rand(UInt),
        maxtime = 3600*(n_sim/25),
        maxiter = nothing,
        maxfevals = 30000,
        ftarget = nothing,
        xtol = 1e-5,
        ftol = 1e-5)
    


    # # # Estimated parameters: 
    println("Best Cand:  " ,xbest(res_CMAE))
    return xbest(res_CMAE), fbest(res_CMAE), h 
    # opt = bbsetup(loglike; SearchRange = bbo_search_range, NumDimensions =bbo_ndim,  Method = :simultaneous_perturbation_stochastic_approximation, MaxTime = bbo_max_time)

    # bbsolution = bboptimize(opt)


    # bbo_search_range = (-5,5)
    # bbo_population_size =50
    # bbo_max_time=200
    # bbo_ndim = 10
    # bbo_feval = 1000

    # bbsolution1 = bboptimize(loglike, rand(10); SearchRange = bbo_search_range, 
    #     NumDimensions =bbo_ndim, PopulationSize = bbo_population_size,
    #     Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
    #     MaxTime = bbo_max_time , TraceInterval=10.0, TraceMode=:compact, Workers=[myid()])
    # return bbsolution1
end


# Parameter estimates 
for n_sim =25:25:50
    for n_firms = 100:100:100
        est_pars = pmap(x->replicate_byseed(x, n_firms, n_sim),1:n_reps)
        
        estimation_result = Dict()
        push!(estimation_result, "beta_hat" => reduce(vcat, [est_pars[i][1] for i =1:24]'))
        push!(estimation_result, "fitness" => reduce(vcat, [est_pars[i][2] for i =1:24]'))
        push!(estimation_result, "bw" => reduce(vcat, [est_pars[i][3] for i =1:24]'))
        bson("/home/ak68/02/est_$(n_firms)_sim_$(n_sim).bson", estimation_result)
        # bson("/Users/akp/github/NPSML-Estimator-Price-Data/Price Estimation April 2022/LogNormal Dist/MC/est_$(n_firms)_sim_$(n_sim).bson", estimation_result)

    end
end
