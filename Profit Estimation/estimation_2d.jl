# Estimation with profit data



using Distributed
using BSON
addprocs(4)

@everywhere begin
    using Optim
    using LinearAlgebra
    using Random
    using Distributions
    using JuMP
    using Gurobi
    using KernelDensity
end


@everywhere include("LP_DGP-mc-2d.jl")
@everywhere n_firms=1000
@everywhere function rep_fun(i)
    # i=1
    n_rep=i

    A_mat = [
     3. 2. 1.;
     1. 1. 0.;
     1. 0. 1.0
    ]
    sig_up = [
     0.2 .2;
     0.3 .2;
     0.0 .4]
    sig_down = [
     0.5 .25;
     0.1 .5;
     0.0 0.3]
    # ud, dd, up, dp,down =  sim_data_LP(b_up,b_down,sig_up,sig_down,1500,2)
    up_data, down_data, up_profit_data, down_profit_data, dd =
     sim_data_LP(A_mat, sig_up, sig_down, n_firms, 25+n_rep)

    data = zeros(n_firms, 6)
    for i = 1:n_firms
        data[i,1] = up_data[1,i]
        data[i,2] = up_data[2,i]
        data[i,3] = down_data[1,i]
        data[i,4] = down_data[2,i]
        data[i,5] = up_profit_data[i]
        data[i,6] = down_profit_data[i]
        # data[i,5] = up_data[2,i]
        # data[i,6] = down_data[2,i]
    end


     # scatter(data[:,7],data[:,8])


    function est_cdf_step(y_data, x1_data, x2_data,  y, x, h)
        num= 0.0
        den= 0.0
        for i =1:size(y_data)[1]
            num+= (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2])
            den+= (pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]))* pdf(Normal(0,1), (x[2]-x2_data[i])/h[2])
            # println("x is: ", x1_data[i], " y is: ", y_data[i], " value is: ", (y_data[i] < y) * pdf(Normal(0,1), (x[1]-x1_data[i])/h[1]) * pdf(Normal(0,1), (x[2]-x2_data[i])/h[2]))
        end
        return (num/den)
    end



    # est_cdf_step(data[:,5], data[:,1], data[:,2], 1.74, [1,1.58], [0.11, 0.13])
    #
    x1_den = kde(data[:,1])
    x2_den = kde(data[:,2])
    # y_den = kde(data[:,2])
    # plot(x->pdf(x_den,x), -10,10.)
    function cv_fun(y_data, x1_data, x2_data,  y, h)
        cv=0.0
        for i = 1:n_firms
            cv += ((y_data[i] < y) - est_cdf_step(y_data[1:end .!= i], x1_data[1:end .!= i], x2_data[1:end .!= i],  y, [x1_data[i] x2_data[i]], h) )^2 * pdf(x1_den, x1_data[i])* pdf(x2_den, x2_data[i])
        end
        # println("h: ", h, " value: ", n_firms^(-1)*cv  )
        return n_firms^(-1)*cv
    end

    # res = Optim.optimize(x->cv_fun(data[:,5], data[:,1], data[:,2], 3. ,x), [0.02, 0.02])
    # Optim.minimizer(res)
    #
    #

    #
    #
    function cv_int(h1)
        step_size = floor(Int, n_firms/10)
        y_grid = 1:step_size:n_firms
        y_vec  = data[y_grid[1:end],5]
        # println(cv.(data[:,3], data[:,1], y_vec ,h))
        # @show cv(data[:,3], data[:,1], y_vec[1] ,h)
        tmp_f = x->cv_fun(data[:,5], data[:,1], data[:,2], x ,h1)
        cv_vals = map(tmp_f , y_vec)
        println("h: ", h1, " value: ", sum(cv_vals))
        return sum(cv_vals)
    end
    #
    # @show cv_int(1.2)
    # resh = Optim.optimize(cv_int, [0.1, .1])
    # @show h = Optim.minimizer(resh)
    # h= 0.05

    #
    #
    # function cv_y(y_data, x_data,  y, h)
    #     cv=0.0
    #     for i = 1:n_firms
    #         cv += ((y_data[i] < y) - est_cdf(y_data[1:end .!= i], x_data[1:end .!= i],  y, x_data[i], h) )^2 * pdf(y_den, x_data[i])
    #     end
    #     # println("h: ", h, " value: ", n_firms^(-1)*cv  )
    #     return n_firms^(-1)*cv
    # end
    #
    # function cv_int_y(h)
    #     y_grid = 1:10:1500
    #     y_vec  = data[y_grid[1:end],4]
    #     cv_vals = pmap(x->cv_y(data[:,4], data[:,2], x ,h), y_vec)
    #     println("h: ", h, " value: ", sum(cv_vals)  )
    #     return sum(cv_vals)
    # end
    #


    # cv_int_y([1.2,0.2])
    # res_y = Optim.optimize(cv_int_y, [1.,1.])

    # h_down = Optim.minimizer(res_y)
    data=hcat(data, zeros(n_firms, 2))
    h=[0.1, 0.2]
    for i = 1:n_firms
        data[i,7] = est_cdf_step(data[:,5], data[:,1], data[:,2], data[i,5], [data[i,1] data[i,2]] ,h)
        data[i,8] = est_cdf_step(data[:,6], data[:,3], data[:,4], data[i,6], [data[i,3] data[i,4]] , h)
    end



    # Objective function: Argument is the vector of parameters beta
    # It solves for the market with the realized characteristics,
    # This means using the quantiles that are implied from the profits.
    function md_objective(b)
        # A_mat = [
        #  3. 2. 1.;
        #  1. 1. 0.5;
        #  1. 1. 1.0
        # ]
        # sig_up = [
        #  0.2 .2;
        #  0.3 .2;
        #  0.0 .4]
        # sig_down = [
        #  0.5 .25;
        #  0.1 .5;
        #  0.0 0.3]
        A_mat = [b[1] b[2] 1.;
                 b[3] b[4] 0.;
                 1. 0. 1.]
        # [3., 2., 1., 1., 1, 1., .4 .3]
        # sig_up = [1. .2; 0 .2]
        # sig_down = [0.5 .25; 0. .5]
        # ud, dd, up, dp,down =  sim_data_LP(b_up,b_down,sig_up,sig_down,1500,2)
        eps_vec = quantile.(LogNormal(0.,abs(b[5])),data[:,7])
        eta_vec = quantile.(LogNormal(0., abs(b[6])) ,data[:,8])
        # eta_vec = quantile.(LogNormal(0., abs(b[10])),data[:,8])
        surplus_vec = diag(hcat(data[:,1:2], eps_vec) * A_mat * Transpose(hcat(data[:,3:4], eta_vec)))
        total_profit = data[:,5]+ data[:,6]
        # println("par is ", b,  " value is: ", sum((total_profit - surplus_vec).^2))
        return sum((total_profit - surplus_vec).^2)
    end

    # res= Optim.optimize(md_objective, rand(10))
    res= Optim.optimize(md_objective, [3., 2., 1., 1., .4 ,.3])
    # res= Optim.optimize(md_objective, [-3, 2, 1., 1., -1, 0.5, 1. , 1. , .4, .3], iterations=1000000)
    # res= Optim.optimize(md_objective, [-3, 2, 1., -1., .4, .3], iterations=1000000)

    # res= Optim.optimize(md_objective, rand(6), iterations=1000000)

    # Optim.minimizer(res)
#
    return vcat(Optim.minimizer(res),Optim.minimum(res))
end
#
#
# scatter(data[:,1], data[:,2], markersize=2, legends=false,
#    xlabel="upstream x", ylabel="downstream y")
# savefig("/Users/amir/Downloads/figs/matching")
#
#
# scatter(data[:,1], data[:,3], markersize=2, legends=false,
#    xlabel="upstream x", ylabel="upstream profit")
# savefig("/Users/amir/Downloads/figs/up_profit")
#
#
#
# scatter(data[:,5], data[:,6], markersize=2, legends=false,
#    xlabel="upstream eps", ylabel="downstream eta",
#     xlims=(0,2), ylims=(0,2))
# savefig("/Users/amir/Downloads/figs/unobs_real")
#
# scatter(data[:,7], data[:,8], markersize=2, legends=false,
#    xlabel="normalized upstream eps", ylabel="normalized downstream eta ")
# savefig("/Users/amir/Downloads/figs/unobs_est")
#
# eps_vec = quantile.(LogNormal(0.,abs(0.2)),data[:,7])
# eta_vec = quantile.(LogNormal(-1, abs(0.5)),data[:,8])
#
#
# scatter(eps_vec, eta_vec, markersize=2, legends=false,
#    xlabel="estimated upstream eps", ylabel="estimated downstream eta "
#    , xlims=(0,2), ylims=(0,2))
# savefig("/Users/amir/Downloads/figs/unobs_est_inverted")
#

reps=1:80
est_pars = pmap(rep_fun,reps)
# @show est_pars

# mean(abs(est_pars))





estimation_result = Dict()
push!(estimation_result, "beta_hat" => est_pars)
# push!(estimation_result, "beta" => A_mat)


bson("/Users/amir/Downloads/mc_profs_01_tst.bson", estimation_result)


# A_mat = [
#  3. 2. 1.;
#  1. 1. 0.5;
#  1. 1. 1.0
# ]
# sig_up = [
#  0.2 .2;
#  0.3 .2;
#  0.0 .4]
# sig_down = [
#  0.5 .25;
#  0.1 .5;
#  0.0 0.3]
#


est_bcv_500 = BSON.load("/Users/amir/Downloads/mc_profs_01_tst.bson")
beta500 = est_bcv_500["beta_hat"]
# true_pars=[-3., .7, 3, .2, .5]
true_pars = [3., 2., 1., 1., .4 ,.3, 0. ]
# true_pars=[3., 2., 1., 1., -1, 1., .4, .3, 0]
# true_pars=[3., 2., 1., 1., 1, 1., .5, 1., 1. , .4, 0]
errs = zeros(80, 7)
for i =1:80
        beta500[i]=(beta500[i])
        println(round.(beta500[i]; digits=2))
        beta500[i][4] = (beta500[i][4])
        beta500[i][5] = abs(beta500[i][5])
        println(round.(beta500[i]; digits=2))
        errs[i,:]=beta500[i] - true_pars
end
mse_500 =((mean(errs.^2,dims=1)))


bias_500 = mean((beta500))-true_pars

println("nfirms: ", 1500)

println("bias: ",round.(bias_500; digits=4))
println("rmse: ",round.(sqrt.(mse_500); digits=2))
# println("mse: ",round.((mse_500); digits=4))
# bias: [0.6671, -0.2114, -0.5944, 0.0756, 0.0918, 151.8326]
# rmse: [0.87 0.4 0.77 0.12 0.16 170.25]
