### Monte-Carlo -- NPSML estimator -- Normally Distributed
### Simulation using finite market approximation. 
# package for parallel computation
using Distributed
using BSON
# using FLoops
addprocs(24)    # Cores -1 (This is for #444 Mac Pro)
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

@everywhere begin 
    n_reps =50 # Number of replications (fake datasets)
    n_sim = 25
    true_pars = [-1, 1.5, .5, 2.5, 2.5, -2, 1, -1, .5 ]
end




@everywhere function replicate_byseed(n_rep, n_firms)

    Σ_up = [ 
        0 2.;
        0 1.;
        0 1.
        ]


    Σ_down =  [
                0 1.5;
                0 3.;
                0 1.
    ]

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
        sim_data_JV_Normal(bup, bdown, Σ_up, Σ_down, n_firms, 333 + n_rep, false, 0, 0)

    # mean of transfers in the data
    mu_price = mean(price_data)



    # # h: vector of bandwidths
    # # function to be minimized over the choice of h
    # # function uses the fake data above
    # function bcv2_fun(h, down_data, price_data)
    #     h=abs.(h)
    #     ll = 0.0
    #     n_firms = length(price_data)
    #     for i = 1:n_firms
    #         for j=1:n_firms
    #             if (j!=i)
    #                 expr_1 = ((down_data[1,i]-down_data[1,j])/h[1])^2 + ((down_data[2,i]-down_data[2,j])/h[2])^2 + ((price_data[i]-price_data[j])/h[3])^2
    #                 expr_2 = pdf(Normal(),(down_data[1,i]-down_data[1,j])/h[1]) * pdf(Normal(),((down_data[2,i]-down_data[2,j])/h[2])) * pdf(Normal(),((price_data[i]-price_data[j])/h[3]))
    #                 ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
    #             end
    #         end
    #     end
    #     val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1) +
    #                             ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
    #     println("band: ",h," val: ", val)
    #     return val
    # end

    # # only use a sample of size of the nsims not the total observed sample 
    # inds = rand(1:n_firms, n_sim)
    # # Optimize over choice of h
    # res_bcv = Optim.optimize(x->bcv2_fun(x,down_data[1:2,inds],price_data[inds]), rand(3))
    # # res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)
    # h = abs.(Optim.minimizer(res_bcv))

    # Silverman
    m=3
    S=cov(hcat(down_data[1,:], down_data[2,:], price_data))
    H_Silverman = (4/(n_sim*(m+2)))^(2/(m+4)) * S
    @show h= sqrt.(diag(H_Silverman))
    function loglike(b)
        
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
    


        solve_draw =  x->sim_data_JV_Normal(bup, bdown , Σ_up, Σ_down, n_firms, 1234+x, true, up_data[1:2,:], down_data[1:2,:])
        sim_dat = map(solve_draw, 1:n_sim)
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
        # println("n_rep: ", n_rep," parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        return -ll/n_firms
    end


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
    return xbest(res_CMAE)



end



# Parameter estimates 

for n_firms = 100:100:300
    est_pars = pmap(x->replicate_byseed(x, n_firms),1:n_reps)

    estimation_result = Dict()
    push!(estimation_result, "beta_hat" => est_pars)
    push!(estimation_result, "beta" => true_pars)
    bson("NormalDist/MC/MC_50_nf_$n_firms.bson", estimation_result)
end




estimation_result