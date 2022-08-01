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
    include("JV_DGP-mvLogNormal.jl")
end

@everywhere begin 
    n_reps =nworkers() # Number of replications (fake datasets)
    true_pars =  [-1.5, 3.5, -.5, -2.5, .5, -2.5, 1.5, 4, -2, 3]

    xcor = 0.3
    x1var = .1
    x2var = .2
    xcov  = sqrt(x1var)*sqrt(x2var)* xcor

    Σ_up = [x1var xcov 0;
            xcov x2var 0;
            0     0    .1]
    # Σ_up = [0 .1;
    #         0 .2;
    #         0 .1]

    ycor =-0.5
    y1var = .3
    y2var = .4
    ycov  = sqrt(y1var)*sqrt(y2var)* ycor

    Σ_down = [y1var ycov 0;
            ycov y2var 0;
            0     0    .1]


    # Σ_down =  [0 .3;
    #            0 .4;
    #            0 .1]
 
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

@everywhere function replicate_byseed(n_rep, n_firms, n_sim, h_scale,  par_ind, sel_mode, globT, locT, data_mode)

 

    #[β11u, β12u, β21u, β11u, β11d, β12d, β21u, β13u, β33d]

    DGPsize = 3000
    up_data, down_data, price_data =
    sim_data_JV_LogNormal(bup, bdown, Σ_up, Σ_down, DGPsize
        , 38+n_rep, false, 0, 0, true_pars[10])
    
    ind_sample = sample(1:DGPsize, n_firms, replace= false);

    up_data =up_data[:, ind_sample];
    down_data= down_data[:, ind_sample];
    price_data= price_data[ ind_sample];


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
    
    if sum(h .> 10) >0 
        h=[0.04, 0.06, 0.2]
        println("BAD BANDWIDTH")

    end
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
        
        ll=zeros(n_firms)
        n_zeros = 0
        
        for i =1:n_firms
            like =0.
            for j =1:n_sim
                if data_mode == 1 # Only prices 
                    like+=(
                        pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h[3]))
                        )
                elseif data_mode==2 # Only matches
                    like+=(
                        pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                        *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                        )
                elseif  data_mode==3 # Matches and Prices
                    like+=(
                        pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                        *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                        *pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h[3]))
                        )
                end
            end
            # println("like is: ", like, " log of which is: ", log(like/(n_sim*h[1]*h[2]*h[3])))
            if like == 0
            #     # println("Like is zero!!!")
                ll[i] = log(pdf(Normal(),30))
                n_zeros += 1
            else
                if data_mode ==1 # Only prices
                    ll[i]=log(like/(n_sim*h[3]))  
                elseif data_mode==2
                    ll[i]=log(like/(n_sim*h[1]*h[2]))  
                elseif data_mode==3
                    ll[i]=log(like/(n_sim*h[1]*h[2]*h[3]))  
                end
                # ll+=like
            end
    
        end

        sort!(ll)
        drop_thres = max(2, Int(floor(0.05*n_firms)))
        out = mean(ll[drop_thres:end])

        if mod(time(),10)<.01
            println("I'm worker number $(myid()) on thread $(Threads.threadid()), and I reside on machine $(gethostname()).")

            println(" parameter: ", round.(b, digits=4), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        end

        Random.seed!()
        return -out
    end


    # # # Estimated parameters: 

    bbo_search_range = [(-10,10), (-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(0,10), (-10, 0), (-10,10)]
    bbo_population_size =50
    bbo_max_time=globT

    bbo_ndim = length(par_ind)
    bbo_feval = 100000
    
    function fun(x)
        par_point = copy(true_pars)
        par_point[par_ind] = x
        return loglike(par_point)
    end

    cbf = x-> println("parameter: ", round.(best_candidate(x), digits=3), " n_rep: ", n_rep, " fitness: ", best_fitness(x) )
    nopts=1
    opt_mat =zeros(nopts,length(par_ind)+1)

    for i = 1:nopts
        bbsolution1 = bboptimize(fun; SearchRange = bbo_search_range, 
            NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
            Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
            TraceInterval=30.0, TraceMode=:compact, MaxTime = bbo_max_time,
            CallbackInterval=13,
            CallbackFunction= cbf) 
        @show opt_mat[i,:] = vcat(best_candidate(bbsolution1), best_fitness(bbsolution1))'

    
        # @show opt2 = Optim.optimize(fun, best_candidate(bbsolution1), time_limit=locT)
        # @show opt_mat[i,:] = vcat(Optim.minimizer(opt2), Optim.minimum(opt2))'
    end

    return opt_mat
end

# Parameter estimates 


for n_sim =50:25:50
    for n_firms = 200:200
        for data_mode=3:2:3
                est_pars = pmap(x->replicate_byseed(x, n_firms, n_sim,[  1.,  1., 1.], 1:10, "median", 2*3600 ,10, data_mode), 1:n_reps)
                estimation_result = Dict()
                push!(estimation_result, "beta_hat" => est_pars)
                bson("/home/ak68/LNU/10p-01/est_$(n_firms)_sim_$(n_sim)_dmode_$(data_mode).bson", estimation_result)
        end

    end
end
