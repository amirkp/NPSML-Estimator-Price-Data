# Pkg.add("Distributed")
# Pkg.add("Optim")
# Pkg.add("BenchmarkTools")
# Pkg.add("Plots")
# Pkg.add("LinearAlgebra")
# Pkg.add("Distributions")
# Pkg.add("BlackBoxOptim")

using Distributed
addprocs(4)

@everywhere begin
    using Optim
    using LinearAlgebra
    using Random
    using Distributions
end
@everywhere include("data_sim_seed.jl")
@everywhere include("data_sim_like.jl")
@everywhere n_firms=500

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

    function ucv_fun(h)
        ll = 0.0
        for i = 1:n_firms
            for j=1:n_firms
                if (j!=i)
                    expr_1=0.0
                    expr_1 += -0.25*((up_data[1,i]-up_data[1,j])/h[1])^2 -0.25*((down_data[1,i]-down_data[1,j])/h[2])^2 -0.25*((price_data_cf[i]-price_data_cf[j])/h[3])^2
                    expr_2 =0.0
                    expr_2 += -0.5*((up_data[1,i]-up_data[1,j])/h[1])^2 -0.5*((down_data[1,i]-down_data[1,j])/h[2])^2 -.5*((price_data_cf[i]-price_data_cf[j])/h[3])^2
                    ll += exp(expr_1)- (2*2^(3/2))*exp(expr_2)
                end
            end
        end
        val = ((2*sqrt(pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((2*sqrt(pi))^3 * n_firms^2 *h[1]*h[2]*h[3])^(-1)*ll
        println("band: ",h," val: ", val)
        return val
    end
    # res_ucv = Optim.optimize(ucv_fun, rand(3))
    # res_bcv = Optim.optimize(bcv2_fun, [0.1,.1,.1])

    # h_ucv = Optim.minimizer(res_ucv)
    # h_bcv = Optim.minimizer(res_bcv)


    function loglikep(b)
        n_sim=500
        solve_draw = x->sim_data_like(up_data[1,:],[b[1], b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5])], n_firms, abs(b[6]), 1234+x)
        sim_dat = pmap(solve_draw, 1:n_sim)
        ll=0.0
        for i =1:n_firms
            m=2
            yy = [sim_dat[s][2][1,i] for s=1:n_sim]
            pp = [sim_dat[s][5][i] for s=1:n_sim]
            S=cov(vcat())
            H_Silverman = (4/(n_firms*(m+2)))^(2/(m+4)) * S
            

            h= sqrt.(diag(H_Silverman))
            # h= sqrt.(diag(H_Scott))
            # println("h is: ", h)
            #
            #
            # hx = h[1]; hy=h[2]; hp=h[3]
            like =0.
            for j =1:n_sim
                like+=pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))*pdf(Normal(),((price_data_cf[i] - sim_dat[j][5][i])/h[2]))
            end
            ll+=log(like/(n_sim*h[1]*h[2]))
        end

        # println("parameter: ", b, " function value: ", -ll/n_firms)
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

    like_5d_ucv = x-> loglikep([x[1],x[2],x[3],x[4],x[5],x[6]])
    res = Optim.optimize(like_5d_ucv, [2.1900093988180003, 5.49047594650638, 1.6534863491899128, -1.994336284673246, -1.5154487714473868, 3.7027830964623494], LBFGS(), Optim.Options(show_every = 1, time_limit=300, show_trace = true, extended_trace= true) )
    return Optim.minimizer(res)
    # loglikep(true_pars)
end


b1 = replicate_byseed(10)


n_reps=4
reps =1:n_reps

est_pars = pmap(replicate_byseed,reps)

@show mean(est_pars)


@show true_pars

mean(est_pars) = [2.3346093670195125, 5.398365100414128, 1.7333944382661417, 2.6067525126617, 4.718895649082294, 3.073075729875808]
true_pars = [2.0, 5.5, 1.5, 2.0, 3.0, 4.0]
