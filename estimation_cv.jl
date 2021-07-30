# Pkg.add("Distributed")
# Pkg.add("Optim")
# Pkg.add("BenchmarkTools")
# Pkg.add("Plots")
# Pkg.add("LinearAlgebra")
# Pkg.add("Distributions")
# Pkg.add("BlackBoxOptim")

using Distributed
addprocs()
using Optim
using BenchmarkTools
using Plots
@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    #using ForwardDiff
end
# using NLopt
# using BSON
using BlackBoxOptim
@everywhere include("data_sim_seed.jl")
@everywhere include("data_sim_like.jl")

@everywhere n_firms=500

@everywhere up_data, down_data, up_profit_data_cf, down_profit_data_cf, price_data_cf, A_mat, β_diff, β_up, β_down, Σ_up, Σ_down =
 sim_data([2.,5.5, 1.], [1.5, 2., 1.], [3., 1., 1.], [2., 1., 3.], n_firms, 25)

@everywhere price_data_cf = price_data_cf + rand(Normal(0.,4.), n_firms)

# scatter(up_data[1,:], down_data[1,:], markersize=2, xlabel="x", ylabel="y", title="Matching- true DGP", ylims=(-5,5))
# scatter(up_data[1,:], price_data_cf, markersize=2,  xlabel="x", ylabel="p", title="Prices- true DGP", ylims=(-1700,30))
# allplots = plot(p1,p2,p3,p4, layout=(2,2), legend=false)



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
    println("band: ",h," val: ", val)
    return val
end

res_ucv = Optim.optimize(ucv_fun, rand(3))
res_bcv = Optim.optimize(bcv2_fun, [0.1,.1,.1])

h_ucv = Optim.minimizer(res_ucv)
h_bcv = Optim.minimizer(res_bcv)

#
# m=3
# S=cov([up_data[1,:]  down_data[1,:]  price_data_cf])
# H_Silverman = (4/(n_firms*(m+2)))^(2/(m+4)) * S
# H_Scott = n_firms^(-2/(m+4)) * S

# h= sqrt.(diag(H_Silverman))
#
#
# hx = h[1]; hy=h[2]; hp=h[3]

function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, eval(Main, Expr(:(=), nm, val)))
    end
end


function sendto(ps::Vector{Int}; args...)
    for p in ps
        sendto(p; args...)
    end
end


@everywhere function loglikep(b, h)
    @everywhere n_sim=500
    solve_draw = x->sim_data_like(up_data[1,:],[b[1], b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5])], n_firms, b[6], 25+x)
    sim_dat = pmap(solve_draw, 1:n_sim)
    sendto(workers(), sim_dat=sim_dat)
    ll = @distributed (+) for i =1:n_firms

        like =0.
        for j =1:n_sim
            # up_data_2, down_data_2, up_profit_data_cf_2, down_profit_data_cf_2, price_data_cf_2, A_mat_2, β_diff_2, β_up_2, β_down_2, Σ_up_2, Σ_down_2 =
            #  sim_data_like(up_data[1,:],[b[1], b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5]) ], n_firms, 25+j)
            # like+=pdf(Normal(),((up_data[1,i] - up_data_2[1,i])/h[1])) *pdf(Normal(),((down_data[1,i] - down_data_2[1,i])/h[2]))*(pdf(Normal(),(price_data_cf[i] - price_data_cf_2[i])/h[3]))
            like+=pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[2]))*pdf(Normal(),((price_data_cf[i] - sim_dat[j][5][i])/h[3]))
            # like+=(down_data[1,i] - down_data_2[1,i])/h[2]
            # like+=(pdf(Normal(),(price_data_cf[i] - price_data_cf_2[i])/h[3]))
        end

        like = log(like/(n_sim*h[2]*h[3]))
        # ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
        # ll[i]=log(like/(n_sim*h[2]*h[3]))
        # ll+=log(like/(h[3]))
        # ll+=log(like/(h[2]))
        # ll+=like/h[2]
    end
    # val = norm(down_data[1,:]- down_data_2[1,:]) + norm(price_data_cf-price_data_cf_2)
    println("parameter: ", b, " function value: ", -ll/n_firms)
    # println("function value: ", -ll/n_firms)

    # println("hi")
    # println("pars: ", b)
    return -ll/n_firms
    # return ll
end


@everywhere function loglikep(b, h)
    @everywhere n_sim=500
    solve_draw = x->sim_data_like(up_data[1,:],[b[1], b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5])], n_firms, b[6], 25+x)
    sim_dat = pmap(solve_draw, 1:n_sim)
    # sendto(workers(), sim_dat=sim_dat)
    ll=0.
    for i =1:n_firms
        like =0.
        for j =1:n_sim
            # up_data_2, down_data_2, up_profit_data_cf_2, down_profit_data_cf_2, price_data_cf_2, A_mat_2, β_diff_2, β_up_2, β_down_2, Σ_up_2, Σ_down_2 =
            #  sim_data_like(up_data[1,:],[b[1], b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5]) ], n_firms, 25+j)
            # like+=pdf(Normal(),((up_data[1,i] - up_data_2[1,i])/h[1])) *pdf(Normal(),((down_data[1,i] - down_data_2[1,i])/h[2]))*(pdf(Normal(),(price_data_cf[i] - price_data_cf_2[i])/h[3]))
            like+=pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[2]))*pdf(Normal(),((price_data_cf[i] - sim_dat[j][5][i])/h[3]))
            # like+=(down_data[1,i] - down_data_2[1,i])/h[2]
            # like+=(pdf(Normal(),(price_data_cf[i] - price_data_cf_2[i])/h[3]))
        end

        # like = log(like/(n_sim*h[2]*h[3]))
        # ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
        ll+=log(like/(n_sim*h[2]*h[3]))
        # ll+=log(like/(h[3]))
        # ll+=log(like/(h[2]))
        # ll+=like/h[2]
    end
    # val = norm(down_data[1,:]- down_data_2[1,:]) + norm(price_data_cf-price_data_cf_2)
    println("parameter: ", b, " function value: ", -ll/n_firms)
    # println("function value: ", -ll/n_firms)

    # println("hi")
    # println("pars: ", b)
    return -ll/n_firms
    # return ll
end


# true paramaeters:[2.,10.5, 1.], [1.5, 2., 1.], [3., 1., 1.], [2., 1., 3.], n_firms, 25)

# println("This is like val", loglikep([200., 25, 1.5, 2., 3.], h_bcv))

like_1d_1 = x-> loglikep([x, 5.5, 1.5, 2., 3.,4.], h_ucv)
@btime like_1d_1(2.)
# parameter: [2.0, 5.5, 1.5, 2.0, 3.0, 4.0] function value: 4.520071134964339 (Compare later with distributed form)
plot(like_1d_1, -2., 5.)
Optim.optimize(like_1d_1, -1., 4)


like_1d_2 = x-> loglikep([2., x, 1.5, 2., 3.,4.],h_ucv)
p1 = plot(like_1d_2, -10., 10.,  legends = false, xlabel="β_12u", ylabel="log-likelihood",title="h_y=0.02, h_p =0.05" )
res_1d2 = Optim.optimize(like_1d_2, -20., 21.005)
@show minimizer_1d2 = Optim.minimizer(res_1d2)
@show min_1d2 = Optim.minimum(res_1d2)


like_2d_12 = x-> loglikep([x[1],x[2], 1.5, 2., 3.,4.], h_ucv)
res_ed = Optim.optimize(like_2d_12, [2., 1.5], LBFGS(), Optim.Options(show_every = 1, time_limit=250, show_trace = true, extended_trace=true) )
@show minimizer_2d_12 = Optim.minimizer(res_ed)
@show min_2d_12 = Optim.minimum(res_ed)

like_2d_12 = x-> loglikep([x[1],x[2], 1.5, 2., 3.,x[3]], h_ucv)
res_ed = Optim.optimize(like_2d_12, [2., 1.5,1.], LBFGS(), Optim.Options(show_every = 1, time_limit=250, show_trace = true, extended_trace=true) )
@show minimizer_2d_12 = Optim.minimizer(res_ed)
@show min_2d_12 = Optim.minimum(res_ed)


like_2d_13 = x-> loglikep([x[1],1.5, x[2], 2., 3.,4.], h_ucv)
res_ed = Optim.optimize(like_2d_13, rand(2), LBFGS(), Optim.Options(show_every = 1, time_limit=250, show_trace = true, extended_trace=true) )
@show minimizer_2d_12 = Optim.minimizer(res_ed)
@show min_2d_12 = Optim.minimum(res_ed)




like_2d_13 = x-> loglikep([x[1],25, x[2], 2., 3.], h_bcv)
res_ed = Optim.optimize(like_2d_13, rand(2), LBFGS(), Optim.Options(show_every = 1, time_limit=250, show_trace = true, extended_trace=true) )
@show minimizer_2d_12 = Optim.minimizer(res_ed)
@show min_2d_12 = Optim.minimum(res_ed)



like_4d_1234 = x-> loglikep([x[1],x[2], x[3], x[4], 3., 4.] , h_ucv)
res_ed = Optim.optimize(like_4d_1234, [2., 15., 2., 2.], LBFGS(), Optim.Options(show_every = 1, time_limit=1500, show_trace = true, extended_trace=true) )
@show minimizer_4d_1234 = Optim.minimizer(res_ed)
@show min_2d_12 = Optim.minimum(res_ed)



# like_5d_12 = x-> loglikep([x[1],x[2],x[3],x[4],x[5]], 0.3, 1.)
 # * x: [-0.09690592086955749, 6.207596278207483, 0.8249907520680574, 2.186328445808209, 0.00045963225032968815]
# like_5d_12([-0.09690592086955749, 6.207596278207483, 0.8249907520680574, 2.186328445808209, 0.00045963225032968815])
# gui()
# gc = [-0.09690592086955749, 6.207596278207483, 0.8249907520680574, 2.186328445808209, 0.00045963225032968815]
# savefig(p1,"plots/b12_smallbw.png")
# savefig(p1,"plots/b12_smallbw_largepar.png")
# savefig(p1,"plots/b12_smallbw_originalpar.png")
Optim.optimize(like_1d_1, -5., 5.)
Optim.optimize(like_1d_2, -10., -.0001 )

like_2d_12([1.,2.])

Optim.optimize(like_2d_12, rand(2), LBFGS(), Optim.Options(show_every = 1, time_limit=1000, show_trace = true, extended_trace=true) )
res_global = bboptimize(like_2d_12; SearchRange = (-10.,10.), Method= :separable_nes, NumDimensions = 2, MaxTime = 80000.0)


like_5d_ucv = x-> loglikep([x[1],x[2],x[3],x[4],x[5],x[6]], h_bcv)
Optim.optimize(like_5d_ucv, rand(6), LBFGS(), Optim.Options(show_every = 1, time_limit=1000, show_trace = true, extended_trace=true) )


# seed 25: parameter: [1.9823556801846092, 6.009794955906781, 1.9983971765117419, 2.1801694583072577, 3.0, 3.6912858368257018] function value: 4.494778674115961
# seed 26: parameter: [2.0721961455232245, 5.817968201422133, 1.828211184463561, 2.432880134435356, 3.0, 3.686186235088775] function value: 4.3851884217388

# seed 25:  * x: [1.9652914247549995, 5.947791868504551, 1.983501559864983, 2.6458970868393847, 7.792318352558163, 3.5674174916778094], 4.488648e+00
# seed 26: parameter: [1.9742458996958692, 5.911086948713182, 1.8215095549153721, 1.5452420372329585, -0.5603035495315276, 3.8948431207989236] function value: 4.381834687477485



stop


#
# #
# # 17     2.329472e+00     2.249265e-05
# # * Current step size: 0.9758512616468187
# # * time: 167.05351400375366
# # * g(x): [-2.2492652838042148e-5, 2.1456770478472654e-6]
# # * x: [2.206365058660887, 15.422906759510761]
# res_global = bboptimize(like_5d_12; SearchRange = (0.,6.), Method= :separable_nes, NumDimensions = 5, MaxTime = 80000.0)
# best_candidate(res_global)
# best_fitness(res_global)
# res_global
#
# Optim.optimize(like_2d_12, rand(2))
# # 23     1.865733e-01     6.307761e-09
# # * Current step size: 1.0198787480018146
# # * time: 125.62300682067871
# # * g(x): [1.7068061036597273e-9, 3.1164522024677745e-9, -6.307760716015719e-9, 3.1154667446607634e-9, -4.774346164421769e-10]
# # * x: [2.2168504022690003, 4.864552323760359, 1.3297787123842137, 2.1156277888831396, 3.1969351434761557]
#
# # plot(like_1d, 0.,10.)
# # plot(like_1d_2, -3050.00001, 3050.)
#
# res = Optim.optimize(like_1d_2,0.,40.)
# # res = Optim.optimize(like_1d_2,-5.,5)
#
# ([2.,1.5, 1.], [1.5, 2., 1.], [3., 1., 1.], [2., 1., 3.]
# function bw_est(h)
#     println(h)
#     like_2d_12 = x-> loglikep([x[1],x[2], 1.5, 2., 3.], h[1], h[2])
#     res = Optim.optimize(like_2d_12, [2., 2.])
#     x_min = Optim.minimizer(res)
#     val = (2. -x_min[1])^2 + (15 - abs(x_min[2]))^2
#     println("x is: ", x_min)
#     return val
#     # like_1d_2 = x->loglikep([2., x, 1.5, 2., 3], abs(h[1]), abs(h[2]) )
#     # res = Optim.optimize(like_1d_2, -50., 0. )
#     # return min(abs(Optim.minimizer(res) - 3.5), abs(Optim.minimizer(res) + 3.5))
# end
#
#
#
# bw_est([0.1,÷c
# Optim.optimize(bw_est, [0.1, .12], Optim.Options(show_every = 1, time_limit=1000, show_trace = true, extended_trace=true))
#
#
# n_points=50
# par_range = range(-3,0.0001, length=n_points)
# likevec = zeros(n_points)
#     # jacobian(central_fdm(5, 1), loglikep,  vcat(par_range[2],cand[2:4],1))[1][1,5]
# for i =1:n_points
#     println("This is i = ", i)
#     # [0.5416550765453207, -0.09899434663251219, 1.862759698103865, 0.9068518984843513, 1.0]
#     # likevec[i] = loglikep([par_range[i],-0.09899434663251219, 1.862759698103865, 0.9068518984843513, 1.])
#     # likevec[i] = loglikep([par_range[i], .5, 3., -1.8, 1.])
#     # likevec[i] = loglikep([2., .5, par_range[i], -1.8, 1.])
#     # likevec[i] = loglikep([2., .5, 3.0, par_range[i], 1.])
#
#     # likevec[i] = loglikep([2.,par_range[i], 3., -1.8, 1.])
#     # likevec[i] = loglikep(vcat(par_range[i],cand[2:4],1))
#     # likevec[i] = loglikep([2.,2.5,1.5,2.,par_range[i]])
#     likevec[i] = like_1d_2(par_range[i])
#     # likevec[i] = loglikep1([2.0, par_range[i],1.5,2.,3.])
#     # likevec[i] = grad_fun(vcat(par_range[i],cand[2:4],1))[1]
#     # println("derivative at ", par_range[i], " is: ", likevec[i])
#     println("func value  at ", par_range[i], " is: ", likevec[i])
#     # likevec[i]=jacobian(central_fdm(5, 1), loglikep,  vcat(par_range[i],cand[2:4],1))[1][1,1]
# end
# # # likevec
# # scatter(par_range, likevec, markersize=2,legends=false, title="Sinkhorn High Precision")
# scatter(par_range, likevec, markersize=2,legends=false)
# #
#
#
#
#
#
#
#
# match_dist_fun11 = x->match_dist_fun3([4.,x, 1.5, 2., 3.])
# match_dist_fun11 = x->match_dist_fun3([x,1.5, 1.5, 2., 3.])
# match_dist_fun2d = x->match_dist_fun3([x[1],x[2], 1.5, 2., 3.])
# match_dist_fun3d = x->match_dist_fun3([x[1],x[2],x[3], 2., 3.])
#
# 1.431e4
# 1.69e4
# 6.671e-10
# 7.311e-10
# Optim.optimize(match_dist_fun11, 0., 10)
# match_dist_fun11(4.)
# match_dist_fun11(xx)
# match_dist_fun11([4.4,1.5])
#
#
# # match_dist_fun11(4.)
# plot(match_dist_fun11, -5,10.)
#
# [2.5,1.5,2.,3.]
# match_dist_fun2([4., 2.5, 1.5, 2., 3., 1.])
#
#
#
# match_dist_fun1([3.9999999999617355, 2.4999999999730025, 1.499999999971974, 1.6379379185084688, 2.012130468713147, 1.0000000000139542, 1.4909569964425031])
#
# x0 = [-.47,5.7,15.,11.]
#
# # Plots.PlotlyBackend()
# xx= [5.002804894787224, 6.516542079419327e-6]
#
# plot(match_dist_fun, 1.5, 2.5, ylims=(0,100))
#
#
#
#
# res =optimize(match_dist_fun, x0,LBFGS(); auto)
#
# res =optimize(match_dist_fun2d,[1e-10,1e-10],LBFGS(),Optim.Options(show_every = 1, time_limit=200, store_trace = true, show_trace = true, extended_trace=true))
#
# res =optimize(match_dist_fun3d,[1e-10,1e-10,1e-10],LBFGS(),Optim.Options(show_every = 1, time_limit=200, store_trace = true, show_trace = true, extended_trace=true))
#
#
#
#
#
#
# [2.5,1.5,2.,3.]
# cand = Optim.minimizer(res)
# #
#
# res_global = bboptimize(match_dist_fun11; SearchRange = (0.,6.), Method= :separable_nes, NumDimensions = 2, MaxTime = 80000.0)
#
# cand1= best_candidate(res_global)
# res_global
# stop here
#
# # up_data, down_data, up_profit_data_cf, down_profit_data_cf, price_data_cf, A_mat, β_diff, β_up, β_down, Σ_up, Σ_down =
# #  sim_data([9.222284953198086,-0.4752353168934279, 1.], [5.722284953198086, 15.978001243791626, 1.], [1.5, 1., 1.], [1.4, 1., 11.065128049954906], n_firms, 25)
# #
# # 19     2.734059e-01     1.395439e+02
# # * Current step size: 97.97056712676077
# # * time: 268.9692180156708
# # * g(x): [-2.6051825994896546e-5, -139.5438547140169]
# # * x: [5.002804894787224, 6.516542079419327e-6]
#
#
# # [9.222284953198086, -0.4752353168934279, 5.722284953198086, 15.978001243791626, 11.065128049954906]
#
#
# up_data_cf = copy(up_data)
# down_data_cf = copy(down_data)
#
#
# C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus
#
# const μ_n = ones(n_firms)/n_firms;
# const ν_m = ones(n_firms)/n_firms;
#
# a,b,c  = sinkhorn_stabilized1(μ_n, ν_m, C, 1, absorb_tol = 1e3, max_iter = 10000, verbose = true)
# c=-c; b=-b;
#
#
# # down_match_data_maxrow = down_data[:,argmax.(eachrow(a))]
# down_match_data_exp = n_firms*Transpose(a*Transpose(down_data))
#
#
# # down_match_data_maxrow_index = argmax.(eachrow(a))
# # up_valuation_sinkhorn = diag(up_data'*β_up*down_match_data_maxrow)
# up_valuation_sinkhorn_exp = diag(up_data'*β_up*down_match_data_exp)
#
#
#
# # down_valuation_sinkhorn = diag(up_data'*β_down*down_match_data_maxrow)
# down_valuation_sinkhorn_exp = diag(up_data'*β_down*down_match_data_exp)
#
# # down_match_profit_sinkhorn = c[down_match_data_maxrow_index[:]]
# down_match_profit_sinkhorn_exp =n_firms* a*c
#
# # pairwise_production_profits_sinkhorn = b + down_match_profit_sinkhorn
# pairwise_production_profits_sinkhorn_exp = b + down_match_profit_sinkhorn_exp
#
# # pairwise_production_sinkhorn= diag(up_data'*A_mat*down_match_data_maxrow)
# pairwise_production_sinkhorn_exp= diag(up_data'*A_mat*down_match_data_exp)
#
# # production_bias = pairwise_production_sinkhorn - pairwise_production_profits_sinkhorn
# production_bias_exp = pairwise_production_sinkhorn_exp - pairwise_production_profits_sinkhorn_exp
#
# alpha = 0.5
# # up_profit_data_sinkhorn = b + (alpha .*production_bias)
# up_profit_data_sinkhorn_exp = b + (alpha .*production_bias_exp)
#
#
# # down_match_profit_sinkhorn = down_match_profit_sinkhorn +((1-alpha) .*production_bias)
# down_match_profit_sinkhorn_exp = down_match_profit_sinkhorn_exp +((1-alpha) .*production_bias_exp)
#
# # mindprof = minimum(down_match_profit_sinkhorn)
# mindprof_exp = minimum(down_match_profit_sinkhorn_exp)
#
# # down_match_profit_sinkhorn = down_match_profit_sinkhorn .- mindprof
# down_match_profit_sinkhorn_exp = down_match_profit_sinkhorn_exp .- mindprof_exp
#
# # up_profit_data_sinkhorn = up_profit_data_sinkhorn .+ mindprof
# up_profit_data_sinkhorn_exp = up_profit_data_sinkhorn_exp .+ mindprof_exp
#
#
# # up_prices_sinkhorn = up_profit_data_sinkhorn - up_valuation_sinkhorn
# up_prices_sinkhorn_exp = up_profit_data_sinkhorn_exp - up_valuation_sinkhorn_exp
#
# # norm(down_match_data_exp[1,:]-down_match_data_maxrow[1,:])
# # norm(up_prices_sinkhorn-up_prices_sinkhorn_exp)
#
# # scatter(up_data[1,:], up_prices_sinkhorn, markersize=2)
#
# scatter(up_data[1,:], up_prices_sinkhorn_exp, markersize=2)
#
# scatter(up_data[1,:], down_match_data_exp[1,:], markersize=2)
# # scatter(up_data[1,:], down_match_data_maxrow[1,:], markersize=2)
# scatter(up_data[1,:], down_data[1,:], markersize=2)
# scatter(up_data[1,:], price_data_cf, markersize=2)
# # down_prices_sinkhorn = -(down_match_profit_sinkhorn - down_valuation_sinkhorn)
# down_prices_sinkhorn_exp = -(down_match_profit_sinkhorn_exp - down_valuation_sinkhorn_exp)
#
#
# function tao_func(x,h)
#     δ=30
#     if abs(x)< h^δ
#         return 0.
#     elseif abs(x)>2*h^δ
#         return 1.
#     elseif 2*h^δ>abs(x)>h^δ
#         return (4*(x-h^δ)^3/h^(3*δ)) - (3*(x-h^δ)^4)/h^(4*δ)
#     end
# end
#
#
# @everywhere begin
#     const x=$up_data[1,:]
#     const y=$down_data[1,:]
#     const y_match=$down_data[1,:]
#     const price = $price_data_cf
# end
#
#
# m=2
# S=cov([y  price])
# H_Silverman = (4/(n_firms*(m+2)))^(2/(m+4)) * S
# H_Scott = n_firms^(-2/(m+4)) * S
#
# const hx, hp = sqrt.(diag(H_Silverman))
#
# @everywhere const n_sim = 40;
#
# @everywhere function loglikep(p::Array{T}) where {T<:Real}
#     β_up_=zeros(T,3,3)
#     β_down_=zeros(T,3,3)
#     β_up_[1,1]=p[1]
#     β_up_[1,2]=p[2]
#     # β_up_[1,3]=1
#     β_up_[2,3]=1.
#     β_down_[1,1]= p[3]
#     β_down_[2,1]= p[4]
#     β_down_[3,1]= 1.
#
#     A_mat_ = β_up_ + β_down_
#
#     Σ_up = [1.0    0;
#             0    1.0]
#     Σ_down = [1.0    0;
#               0      p[5]]
#
#     p_up_ = MvNormal([0;0], Σ_up)
#     p_down_ = MvNormal([0;0], Σ_down )
#
#     function solve_draw(i)
#         Random.seed!(i*1234)
#         # price_noise = rand(Normal(0,10.), n_firms)
#         up_data_sim = vcat(x', rand(p_up_, n_firms))
#         down_data_sim = vcat(y', rand(p_down_, n_firms))
#         C = -1*Transpose(up_data_sim) * A_mat_ * down_data_sim;
#         a,b,c  = sinkhorn_stabilized1(μ_n, ν_m, C, .5, absorb_tol = 1e3, max_iter = 500, verbose = false)
#         c=-c; b=-b;
#
#
#         # down_match_data_maxrow = down_data[:,argmax.(eachrow(a))]
#         down_match_data_exp = n_firms*Transpose(a*Transpose(down_data_sim))
#
#
#         # down_match_data_maxrow_index = argmax.(eachrow(a))
#         # up_valuation_sinkhorn = diag(up_data'*β_up*down_match_data_maxrow)
#         up_valuation_sinkhorn_exp = diag(up_data_sim'*β_up_*down_match_data_exp)
#
#
#
#         # down_valuation_sinkhorn = diag(up_data'*β_down*down_match_data_maxrow)
#         down_valuation_sinkhorn_exp = diag(up_data_sim'*β_down_*down_match_data_exp)
#
#         # down_match_profit_sinkhorn = c[down_match_data_maxrow_index[:]]
#         down_match_profit_sinkhorn_exp =n_firms* a*c
#
#         # pairwise_production_profits_sinkhorn = b + down_match_profit_sinkhorn
#         pairwise_production_profits_sinkhorn_exp = b + down_match_profit_sinkhorn_exp
#
#         # pairwise_production_sinkhorn= diag(up_data'*A_mat*down_match_data_maxrow)
#         pairwise_production_sinkhorn_exp= diag(up_data_sim'*A_mat_*down_match_data_exp)
#
#         # production_bias = pairwise_production_sinkhorn - pairwise_production_profits_sinkhorn
#         production_bias_exp = pairwise_production_sinkhorn_exp - pairwise_production_profits_sinkhorn_exp
#
#         alpha = 0.5
#         # up_profit_data_sinkhorn = b + (alpha .*production_bias)
#         up_profit_data_sinkhorn_exp = b + (alpha .*production_bias_exp)
#
#
#         # down_match_profit_sinkhorn = down_match_profit_sinkhorn +((1-alpha) .*production_bias)
#         down_match_profit_sinkhorn_exp = down_match_profit_sinkhorn_exp +((1-alpha) .*production_bias_exp)
#
#         # mindprof = minimum(down_match_profit_sinkhorn)
#         mindprof_exp = minimum(down_match_profit_sinkhorn_exp)
#
#         # down_match_profit_sinkhorn = down_match_profit_sinkhorn .- mindprof
#         down_match_profit_sinkhorn_exp = down_match_profit_sinkhorn_exp .- mindprof_exp
#
#         # up_profit_data_sinkhorn = up_profit_data_sinkhorn .+ mindprof
#         up_profit_data_sinkhorn_exp = up_profit_data_sinkhorn_exp .+ mindprof_exp
#
#
#         # up_prices_sinkhorn = up_profit_data_sinkhorn - up_valuation_sinkhorn
#         up_prices_sinkhorn_exp = up_profit_data_sinkhorn_exp - up_valuation_sinkhorn_exp
#
#         # down_match_data_maxrow = down_data_sim[:,argmax.(eachrow(a))]
#         # down_match_data_maxrow_index = argmax.(eachrow(a))
#         # up_valuation_sinkhorn = diag(up_data_sim'*β_up_*down_match_data_maxrow)
#         # down_valuation_sinkhorn = diag(up_data_sim'*β_down_*down_match_data_maxrow)
#         # down_match_profit_sinkhorn = c[down_match_data_maxrow_index[:]]
#         # pairwise_production_profits_sinkhorn = b + down_match_profit_sinkhorn
#         # pairwise_production_sinkhorn= diag(up_data_sim'*A_mat_*down_match_data_maxrow)
#         # production_bias = pairwise_production_sinkhorn - pairwise_production_profits_sinkhorn
#         #
#         # alpha = .5
#         # up_profit_data_sinkhorn = b + (alpha .*production_bias)
#         # down_match_profit_sinkhorn = down_match_profit_sinkhorn +((1-alpha) .*production_bias)
#         # mindprof = minimum(down_match_profit_sinkhorn)
#         # down_match_profit_sinkhorn = down_match_profit_sinkhorn .- mindprof
#         # up_profit_data_sinkhorn = up_profit_data_sinkhorn .+ mindprof
#         # up_prices_sinkhorn = up_profit_data_sinkhorn - up_valuation_sinkhorn
#         return up_data_sim[1,:], down_data_sim[1,:], down_match_data_exp[1,:], up_prices_sinkhorn_exp
#     end
#
#     sim_dat = pmap(solve_draw,1:n_sim)
#     # return sim_dat[1][3][2]
#     count_zero =0
#     count_half_zero=0
#     ll=0.
#     for i = 1:n_firms
#         like=0.
#         for j=1:n_sim
#             like+=pdf(Normal(),(y_match[i] - sim_dat[j][3][i])/hx)*(pdf(Normal(),(price[i] - sim_dat[j][4][i])/hp))
#         end
#         # println("like is: ", like)
#         if like/(n_sim*hx*hp)==0
#             # ll+=log(1e-100)
#             ll+= log(2.5e-250)
#             # ll+=0.
#             # println("price is: ", price[i])
#             count_zero+=1
#             # println("Value close to zero: ", like )
#         else
#             ll+= log(like/(n_sim*hx*hp))
#         end
#         # println("i is ", i, " like  is ", ll)
#
#         # if tao_func(like/(n_sim*hx*hp),hx) ==0
#         #     ll+=0.
#         #     count_zero+=1
#         #
#         #     # println("low like val")
#         #     # ll +=  log(like/(n_sim*hx*hp))
#         # else
#         #     # ll +=  log(like/(n_sim*hx*hp))
#         #     if tao_func(like/(n_sim*hx*hp),hx)<1.0
#         #         println("tao func is: ", tao_func(like/(n_sim*hx*hp),hx))
#         #     end
#         #
#         #     ll += tao_func(like/(n_sim*hx*hp),hx) * log(like/(n_sim*hx*hp))
#         # end
#     end
#
#         # println("parameter value: ", p, "likelihood value: ", -ll*(1/n_firms))
#         # println("total number of zero: ", count_zero)
#         # println("total number of zero: ", count_zero)
#         println("Call to the function with: ", p )
#         # println("hi")
#         println("likelihood value: ", -ll*(1/n_firms))
#         return -ll*(1/n_firms)
# end
#
# # true_like = loglikep([3,2.5,1.5,2.,3.])
# @time true_like = loglikep([2., 2.5, 1.5, 2., 3.])
# [2., 2.5, 1.5, 2., 3.]
#
# [2., 2.5, 1.5, 2., 3.]
#
# loglikep1d = x-> loglikep([x, 2.5, 1.5, 2., 3.])
#
# plot(loglikep1d,0,6)
# @show true_like = loglikep(tst_x)
# res = Optim.optimize(loglikep1d,0.,5)
#
# loglikep1d(2.3)
#
# loglikep1d(1.8)
#
# 2.41 --->10 sims
# 2.28 ---> 40 sims
#
# 2.3 --- >3.207
#
#
#
#
# @show tst_x
#
#
# 2.89
#
#
# # @show loglikep([-3,2.5,1.5,2., 3.])
# grad_fun = x -> ForwardDiff.gradient(loglikep, x)
# #
# grad_fun([3.,2.5,1.5,2.,3.])
# # cand = [-0.591517468108101, 2.8141919046245603, 1.0295406719019349, 2.038108051264322]
#
#
#
# #
# n_points=50
# par_range = range(.005,10., length=n_points)
# likevec = zeros(n_points)
#     # jacobian(central_fdm(5, 1), loglikep,  vcat(par_range[2],cand[2:4],1))[1][1,5]
# for i =1:n_points
#     println("This is i = ", i)
#     # [0.5416550765453207, -0.09899434663251219, 1.862759698103865, 0.9068518984843513, 1.0]
#     # likevec[i] = loglikep([par_range[i],-0.09899434663251219, 1.862759698103865, 0.9068518984843513, 1.])
#     # likevec[i] = loglikep([par_range[i], .5, 3., -1.8, 1.])
#     # likevec[i] = loglikep([2., .5, par_range[i], -1.8, 1.])
#     # likevec[i] = loglikep([2., .5, 3.0, par_range[i], 1.])
#
#     # likevec[i] = loglikep([2.,par_range[i], 3., -1.8, 1.])
#     # likevec[i] = loglikep(vcat(par_range[i],cand[2:4],1))
#     likevec[i] = loglikep([2.,2.5,1.5,2.,par_range[i]])
#     # likevec[i] = loglikep1([2.0, par_range[i],1.5,2.,3.])
#     # likevec[i] = grad_fun(vcat(par_range[i],cand[2:4],1))[1]
#     # println("derivative at ", par_range[i], " is: ", likevec[i])
#     println("func value  at ", par_range[i], " is: ", likevec[i])
#     # likevec[i]=jacobian(central_fdm(5, 1), loglikep,  vcat(par_range[i],cand[2:4],1))[1][1,1]
# end
# # # likevec
# # scatter(par_range, likevec, markersize=2,legends=false, title="Sinkhorn High Precision")
# scatter(par_range, likevec, markersize=2,legends=false, title="Log-likelihood σ_η_2, true value: 3.0")
# #
# # # scatter(par_range, likevec, markersize=2,legends=false, title="Sinkhorn Mid Precision, over smoothing")
# # # scatter(par_range, likevec, markersize=2,legends=false, title="Likelihood Trimming Low")
# # #
# # # scatter(par_range, likevec, markersize=2,legends=false, title="Likelihood Trimming mid")
# # #
# # # scatter(par_range, likevec, markersize=2,legends=false, title="Likelihood Trimming delta 5")
# # #
# # #
# # #
# loglike_2d= b->loglikep([b[1], b[2], 1.5, 2., 3.])
# res_local = Optim.optimize(loglike_2d, [2,2.5], LBFGS(),Optim.Options(show_every = 1, time_limit=200, store_trace = true, show_trace = true, extended_trace=true); autodiff=:forward)
#
#
#
# res_global = bboptimize(loglikep; SearchRange = p_range, NumDimensions = 5,PopulationSize = 40, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 11000.0)
# cand_global = best_candidate(res_global)
#
#
# #
# #
# res_local = Optim.optimize(loglike_ld, [2,2.5,1.5,2.], LBFGS(),Optim.Options(show_every = 1, time_limit=4600, store_trace = true, show_trace = true, extended_trace=true); autodiff=:forward)
#
#
#
#
#
#
# Optim.minimum(res_local)
# println("minimizer is: ", Optim.minimizer(res_local))
#
#
#
# #
# #
# # loglike_ld= b->loglikep([b[1], b[2], b[3], b[4], 1.])
# # println("Likelihood value at true parametes: ", true_like)
# #
# #
# # p_range = [(-3.,3.),(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0),(0.0005,15.)];
# # res_global = bboptimize(loglikep; SearchRange = p_range, NumDimensions = 5,PopulationSize = 40, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 11000.0)
# # cand_global = best_candidate(res_global)
# # println("Min Candidate by GS: ", cand_global)
# # println("Min value by GS: ", best_fitness(res_global))
# # println("Now starting the local solver")
#
# # res_local = Optim.optimize(loglikep, cand_global, LBFGS(),Optim.Options(show_every = 1, x_tol =1e-4, time_limit=3600, store_trace = true, show_trace = true, extended_trace=true); autodiff=:forward)
# # println("Min Candidate by LS: ", Optim.minimizer(res_local))
# # println("Min value by LS: ", Optim.minimum(res_local))
# #
# #
# # res = Dict()
# # push!(res, "global sol" => cand_global)
# # push!(res, "local sol" => Optim.minimizer(res_local))
# # push!(res, "global min" => best_fitness(res_global))
# # push!(res, "local min" => Optim.minimum(res_local))
# # push!(res, "true like" => true_like)
# #
# #
# # solution = copy(BlackBoxOptim.best_candidate(bbsolution))
# # push!(bbo, "solution" => solution)
# # fitness = copy(BlackBoxOptim.best_fitness(bbsolution))
# # push!(bbo, "fitness" => fitness)
# # bson("/outputs/estimation_results_$(task_id).bson", bbo)
#
#
#
#
#
#
# using KDEstimation
#
# x =randn(1000)
#
# lscv_res = lscv(Normal,x,FFT())
#
#
# using Plots; pyplot()
# using KernelDensity
#
# UX= kde(up_data[1,:])
# UP = kde(price_data_cf)
# UX1 = kde_lscv(up_data[1,:])
# den = x-> pdf(U, x)
# den1 = x-> pdf(UX1, x)
# denp= x-> pdf(UP, x)
#
# den(1.)
#
# plot(den, -5,5)
# plot(den1, -5,5)
# plot(denp, -15,5)
