
using Distributed
using BSON
# addprocs(4)

@everywhere begin
    using Optim
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
    using LineSearches
end
@everywhere include("data_sim_seed.jl")
@everywhere include("data_sim_like.jl")
@everywhere n_firms=1000

# @everywhere function ucv_fun(h)
#     ll = 0.0
#     for i = 1:n_firms
#         for j=1:n_firms
#             if (j!=i)
#                 expr_1=0.0
#                 expr_1 += -0.25*((up_data[1,i]-up_data[1,j])/h[1])^2 -0.25*((down_data[1,i]-down_data[1,j])/h[2])^2 -0.25*((price_data_cf[i]-price_data_cf[j])/h[3])^2
#                 expr_2 =0.0
#                 expr_2 += -0.5*((up_data[1,i]-up_data[1,j])/h[1])^2 -0.5*((down_data[1,i]-down_data[1,j])/h[2])^2 -.5*((price_data_cf[i]-price_data_cf[j])/h[3])^2
#                 ll += exp(expr_1)- (2*2^(3/2))*exp(expr_2)
#             end
#         end
#     end
#     val = ((2*sqrt(pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((2*sqrt(pi))^3 * n_firms^2 *h[1]*h[2]*h[3])^(-1)*ll
#     println("band: ",h," val: ", val)
#     return val
# end
#
# @everywhere function bcv2_fun(h)
#     ll = 0.0
#     for i = 1:n_firms
#         for j=1:n_firms
#             if (j!=i)
#                 # expr_1 = 0.0
#                 expr_1 = ((up_data[1,i]-up_data[1,j])/h[1])^2 + ((down_data[1,i]-down_data[1,j])/h[2])^2 + ((price_data_cf[i]-price_data_cf[j])/h[3])^2
#                 expr_2 = pdf(Normal(),(up_data[1,i]-up_data[1,j])/h[1]) * pdf(Normal(),((down_data[1,i]-down_data[1,j])/h[2])) * pdf(Normal(),((price_data_cf[i]-price_data_cf[j])/h[3]))
#                 ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
#             end
#         end
#     end
#     val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
#     println("band: ",h," val: ", val)
#     return val
# end


@everywhere true_pars = [2., 5.5, 1.5, 2., 3., 4. ]

@everywhere function replicate_byseed(n_rep)
    up_data, down_data, up_profit_data_cf, down_profit_data_cf, price_data_cf, A_mat, β_diff, β_up, β_down, Σ_up, Σ_down =
     sim_data([2.,5.5, 1.], [1.5, 2., 1.], [3., 1., 1.], [2., 1., 3.], n_firms, 25+n_rep)
    Random.seed!(25+n_rep)
    price_data_cf = price_data_cf + rand(Normal(0.,4.), n_firms)

    function bcv2_fun(h)
        ll = 0.0
        for i = 1:n_firms
            for j=1:n_firms
                if (j!=i)
                    # expr_1 = 0.0
                    expr_1 = ((up_data[1,i]-up_data[1,j])/h[1])^2 + ((down_data[1,i]-down_data[1,j])/h[2])^2 + ((price_data_cf[i]-price_data_cf[j])/h[3])^2
                    expr_2 = pdf(Normal(),(up_data[1,i]-up_data[1,j])/h[1]) * pdf(Normal(),((down_data[1,i]-down_data[1,j])/h[2])) * pdf(Normal(),((price_data_cf[i]-price_data_cf[j])/h[3]))
                    ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
                end
            end
        end
        val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
        # println("band: ",h," val: ", val)
        return val
    end
    # res_ucv = Optim.optimize(ucv_fun, rand(3))
    res_bcv = Optim.optimize(bcv2_fun, [0.1,.1,.1])

    # h_ucv = Optim.minimizer(res_ucv)
    @show h_bcv = Optim.minimizer(res_bcv)



    function loglikep(b, h)
        n_sim=500
        solve_draw = x->sim_data_like(up_data[1,:],[b[1], b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5])], n_firms, abs(b[6]), 1234+x)
        sim_dat = map(solve_draw, 1:n_sim)
        ll=0.0
        for i =1:n_firms
            like =0.
            for j =1:n_sim
                like+=pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[2]))*pdf(Normal(),((price_data_cf[i] - sim_dat[j][5][i])/h[3]))
            end
            ll+=log(like/(n_sim*h[2]*h[3]))
        end

        println("parameter: ", b, " function value: ", -ll/n_firms)
        return -ll/n_firms
    end

    # like_1d_1 = x-> loglikep([x, 5.5, 1.5, 2., 3.,4.], h_ucv)
    # @btime like_1d_1(2.)
    # # parameter: [2.0, 5.5, 1.5, 2.0, 3.0, 4.0] function value: 4.520071134964339 (Compare later with distributed form)
    # plot(like_1d_1, -2., 5.)
    # Optim.optimize(like_1d_1, -1., 4)
    #
    #
    # like_1d_2 = x-> loglikep([2., x, 1.5, 2., 3.,4.],h_ucv)
    # p1 = plot(like_1d_2, -10., 10.,  legends = false, xlabel="β_12u", ylabel="log-likelihood",title="h_y=0.02, h_p =0.05" )
    # res_1d2 = Optim.optimize(like_1d_2, -20., 21.005)
    # @show minimizer_1d2 = Optim.minimizer(res_1d2)
    # @show min_1d2 = Optim.minimum(res_1d2)
    #

    like_5d_ucv = x-> loglikep([x[1],x[2],x[3],x[4],x[5],x[6]], h_bcv)
    # p_range = [(0.,5.),(3., 8.0),(-3.0, 3.0),(-3.0,3.0),(0.0005,7.),(0.0005,7.)];
    # res_global = bboptimize(like_5d_ucv; SearchRange = p_range, TraceMode= :silent, NumDimensions = 5,PopulationSize = 40, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime =400.0)
    cand_global =true_pars
    # best_candidate(res_global)
    if like_5d_ucv(cand_global)<Inf
        res = Optim.optimize(like_5d_ucv, cand_global, BFGS(linesearch=LineSearches.BackTracking()),Optim.Options(show_every = 1, time_limit=2500, show_trace = true,extended_trace=true))
    else
        res = Optim.optimize(like_5d_ucv, true_pars+rand(6))
    end
    println("Replication: ", n_rep, " estimates: ", Optim.minimizer(res))
    return Optim.minimizer(res)
end

# n_reps=96
# reps =1:n_reps
reps=80:80
est_pars = pmap(replicate_byseed,reps)

# 195     4.445054e+00     9.244283e-09
# * time: 170.4092879295349
# * step_type: inside contraction
# * centroid: [2.0187411681002927, 5.7763914683949675, 1.7929717166602874, 2.3671704442637713, 4.223155907593449, 3.7166504463718377]

estimation_result = Dict()
push!(estimation_result, "beta_hat" => est_pars)
push!(estimation_result, "beta" => true_pars)


bson("/scratch/ak68/np6par_neldermead/bcv_6par_1500_1000_nm.bson", estimation_result)
