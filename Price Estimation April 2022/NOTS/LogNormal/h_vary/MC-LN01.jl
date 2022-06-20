### Monte-Carlo -- NPSML estimator -- Normally Distributed
### Simulation using finite market approximation. 
# package for parallel computation

using Distributed, ClusterManagers
pids = addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
using BSON

@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
    using Assignment
    using Optim
    using CMAEvolutionStrategy
    include("JV_DGP-LogNormal.jl")
end

@everywhere begin 
    n_reps =nworkers() # Number of replications (fake datasets)
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

 

    #[β11u, β12u, β21u, β11u, β11d, β12d, β21u, β13u, β33d]


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
    
    h[1] = h[1] * h_scale[1]
    h[2] = h[2] * h_scale[2]
    h[3] = h[3] * h_scale[3]
    
    
    
    function loglike(b)
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
        if mod(time(),10)<.01
            println("I'm worker number $(myid()) on thread $(Threads.threadid()), and I reside on machine $(gethostname()).")

            println(" parameter: ", round.(b, digits=4), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        end
        Random.seed!()
        return -ll/n_firms
    end
    bbo_search_range = (-5,5)
    bbo_population_size =75
    bbo_max_time=3600*1.5
    bbo_ndim = 10
    bbo_feval = 50000

    bbsolution1 = bboptimize(loglike; SearchRange = bbo_search_range, 
        NumDimensions =bbo_ndim, PopulationSize = bbo_population_size,
        Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
        MaxTime = bbo_max_time , TraceInterval=60.0, TraceMode=:compact)

    return best_candidate(bbsolution1), best_fitness(bbsolution1), h 
end


# replicate_byseed(2, 100,25) 

h_mat = zeros(27, 3)
scales = [0.5 1. 2.]
count = 1
for i1 = 1:3
    for i2 =  1:3
        for i3  = 1:3
            h_mat[count, :] = [scales[i1] scales[i2] scales[i3]]
            global count+=1
        end
    end
end


# Parameter estimates 
for h_id = 6:10
    for n_sim =25:25:25
        for n_firms = 100:100:100
            est_pars = pmap(x->replicate_byseed(x, n_firms, n_sim, h_mat[h_id,:]), 1:n_reps)
            estimation_result = Dict()
            push!(estimation_result, "beta_hat" => reduce(vcat, [est_pars[i][1] for i =1:n_reps]'))
            push!(estimation_result, "fitness" => reduce(vcat, [est_pars[i][2] for i =1:n_reps]'))
            push!(estimation_result, "bw" => reduce(vcat, [est_pars[i][3] for i =1:n_reps]'))
            bson("/home/ak68/h_vary/est_$(n_firms)_sim_$(n_sim)_$(h_mat[h_id, 1])_$(h_mat[h_id, 2])_$(h_mat[h_id, 3]).bson", estimation_result)
            # bson("/Users/amir/github/NPSML-Estimator-Price-Data/Price Estimation April 2022/NOTS/LogNormal/h_vary/est_$(n_firms)_sim_$(n_sim)_$(h_mat[h_id, 1])_$(h_mat[h_id, 2])_$(h_mat[h_id, 3]).bson", estimation_result)
        end
    end
end


