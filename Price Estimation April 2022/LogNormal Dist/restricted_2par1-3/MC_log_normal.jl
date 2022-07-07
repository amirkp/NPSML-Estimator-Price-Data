### Monte-Carlo -- NPSML estimator -- Normally Distributed
### Simulation using finite market approximation. 
# package for parallel computation
using Distributed
using BSON
# using FLoops
addprocs()    # Cores  (This is for #444 Mac Pro)
@everywhere using Optim    # Nelder-Mead Local Optimizer

@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
    using Plots
    using Assignment
    using BenchmarkTools
    include("JV_DGP-LogNormal.jl")
    # include("LP_DGP.jl")
end

@everywhere begin 
    n_reps =24 # Number of replications (fake datasets)
    true_pars =  [-2.5, 1.5, -1.5, -.5, 3.5, 2.5, 1.5, 3, -3, 3]
    # true_pars = round.(randn(Random.seed!(1224),10)*3, digits = 1)
end




@everywhere function replicate_byseed(n_rep, n_firms, n_sim, par_ind)
    # n_rep =22
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
        sim_data_JV_LogNormal(bup, bdown, Σ_up, Σ_down, 3000
            , 38+n_rep, false, 0, 0, true_pars[10])
        
    ind_sample = sample(1:3000, n_firms, replace= false);
    up_data =up_data[:, ind_sample];
    down_data= down_data[:, ind_sample];
    price_data= price_data[ ind_sample];

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
    # h= h*5



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
        # if mod(time(),10)<0.1
            # println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        # end
        Random.seed!()
        return -ll/n_firms
    end


    # # # Estimated parameters: 

    bbo_search_range = (-10,10)
    bbo_population_size =10
    bbo_max_time=length(par_ind)^2 * 45 *(n_firms/50)
    bbo_ndim = length(par_ind)
    bbo_feval = 100000
    function fun(x)
        par_point = copy(true_pars)
        par_point[par_ind] = x
        return loglike(par_point)
    end

    cbf = x-> println("parameter: ", round.(best_candidate(x), digits=3), " n_rep: ", n_rep, " fitness: ", best_fitness(x) )
    opt_mat =zeros(2,3)
    for i = 1:2
        bbsolution1 = bboptimize(fun; SearchRange = bbo_search_range, 
            NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
            Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
            TraceInterval=30.0, TraceMode=:compact, MaxTime = bbo_max_time,
            CallbackInterval=13,
            CallbackFunction= cbf) 
    
        @show opt2 = Optim.optimize(fun, best_candidate(bbsolution1), time_limit=30)
        @show opt_mat[i,:] = vcat(Optim.minimizer(opt2), Optim.minimum(opt2))'
    end
    return opt_mat
end

# replicate_byseed(2, 100,25) 

# Parameter estimates 

for n_sim =25:25:25
    for n_firms =  50:50:100
        est_pars = pmap(x->replicate_byseed(x, n_firms, n_sim, [1, 3]),1:n_reps )
        estimation_result = Dict()
        push!(estimation_result, "beta_hat" => est_pars)
        bson("/Users/akp/github/NPSML-Estimator-Price-Data"*
        "/Price Estimation April 2022/LogNormal Dist/restricted_2par1-3/"*
        "est_$(n_firms)_sim_$(n_sim)_par_1-3", estimation_result)
    end
end






