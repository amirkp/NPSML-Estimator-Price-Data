using Distributed
# addprocs(4)
using Optim
using BenchmarkTools
using Plots
@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using ForwardDiff
end
# using NLopt
# using BSON
using BlackBoxOptim
include("data_sim_seed.jl")


@everywhere function getK1(C, alpha, beta, eps, mu, nu)
    return (exp.(-(C .- alpha .- beta')/eps).*mu.*nu')
end

@everywhere function sinkhorn_stabilized1(mu, nu, C, eps; absorb_tol = 1e3, max_iter = 10000, tol = 1e-9, alpha = nothing, beta = nothing, return_duals = false, verbose = true)
    if isnothing(alpha) || isnothing(beta)
        alpha = zeros(size(mu)); beta = zeros(size(nu))
    end

    u = ones(size(mu)); v = ones(size(nu))
    K = (exp.(-(C)/eps).*mu.*nu')
    i = 0
    # if !(sum(mu) ≈ sum(nu))
    #     throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    # end

    while true
        u = mu./(K*v )
        v = nu./(K'*u )
        # if (max(norm(u, Inf), norm(v, Inf)) > absorb_tol)
        #     if verbose; println("Absorbing (u, v) into (alpha, beta)"); end
        #     # absorb into α, β
        #     alpha = alpha + eps*log.(u); beta = beta + eps*log.(v)
        #     u = ones(size(mu)); v = ones(size(nu))
        #     K = getK1(C, alpha, beta, eps, mu, nu)
        # end
        # if i % 100 == 0
        #     # check marginal
        #     gamma = getK1(C, alpha, beta, eps, mu, nu).*(u.*v')
        #     err_mu = norm(gamma*ones(size(nu)) - mu, Inf)
        #     err_nu = norm(gamma'*ones(size(mu)) - nu, Inf)
        #     # if verbose;
        #     # println(string("Iteration ", i, ", err = ", 0.5*(err_mu + err_nu))); end
        #     if 0.5*(err_mu + err_nu) < tol
        #         break
        #     end
        # end

        if i > max_iter
            if verbose; println("Warning: exited before convergence"); end
            break
        end
        i+=1
    end
    # println("iteration number: ", i)

    alpha = alpha + eps*log.(u); beta = beta + eps*log.(v)
    if return_duals
        return eps*log.(u), eps*log.(v)
    end

    return getK1(C, alpha, beta, eps, mu, nu),alpha, beta
end


##############################
#############################
##############################
#############################


@everywhere const n_firms=500

# task_id = Base.parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

up_data, down_data, up_profit_data_cf, down_profit_data_cf, price_data_cf, A_mat, β_diff, β_up, β_down, Σ_up, Σ_down =
 sim_data([2., 2.5, 1.], [1.5, 2., 1.], [1.5, 1., 1.], [1.4, 1., 3.], n_firms, 20)

up_data_cf = copy(up_data)
down_data_cf = copy(down_data)


C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus

const μ_n = ones(n_firms)/n_firms;
const ν_m = ones(n_firms)/n_firms;

a,b,c  = sinkhorn_stabilized1(μ_n, ν_m, C, .2, absorb_tol = 1e3, max_iter = 1000, verbose = true)
c=-c; b=-b;


down_match_data_maxrow = down_data[:,argmax.(eachrow(a))]
down_match_data_exp = n_firms*Transpose(a*Transpose(down_data))



scatter(up_data[1,:],down_match_data_exp[1,:], markersize=2)
scatter(up_data[1,:],down_match_data_maxrow[1,:], markersize=2)

down_match_data_maxrow_index = argmax.(eachrow(a))
up_valuation_sinkhorn = diag(up_data'*β_up*down_match_data_maxrow)
up_valuation_sinkhorn_exp = diag(up_data'*β_up*down_match_data_exp)



down_valuation_sinkhorn = diag(up_data'*β_down*down_match_data_maxrow)
down_valuation_sinkhorn_exp = diag(up_data'*β_down*down_match_data_exp)

down_match_profit_sinkhorn = c[down_match_data_maxrow_index[:]]
down_match_profit_sinkhorn_exp =n_firms* a*c

pairwise_production_profits_sinkhorn = b + down_match_profit_sinkhorn
pairwise_production_profits_sinkhorn_exp = b + down_match_profit_sinkhorn_exp

pairwise_production_sinkhorn= diag(up_data'*A_mat*down_match_data_maxrow)
pairwise_production_sinkhorn_exp= diag(up_data'*A_mat*down_match_data_exp)

production_bias = pairwise_production_sinkhorn - pairwise_production_profits_sinkhorn
production_bias_exp = pairwise_production_sinkhorn_exp - pairwise_production_profits_sinkhorn_exp

alpha = 0.5
up_profit_data_sinkhorn = b + (alpha .*production_bias)
up_profit_data_sinkhorn_exp = b + (alpha .*production_bias_exp)


down_match_profit_sinkhorn = down_match_profit_sinkhorn +((1-alpha) .*production_bias)
down_match_profit_sinkhorn_exp = down_match_profit_sinkhorn_exp +((1-alpha) .*production_bias_exp)

mindprof = minimum(down_match_profit_sinkhorn)
mindprof_exp = minimum(down_match_profit_sinkhorn_exp)

down_match_profit_sinkhorn = down_match_profit_sinkhorn .- mindprof
down_match_profit_sinkhorn_exp = down_match_profit_sinkhorn_exp .- mindprof_exp

up_profit_data_sinkhorn = up_profit_data_sinkhorn .+ mindprof
up_profit_data_sinkhorn_exp = up_profit_data_sinkhorn_exp .+ mindprof_exp


up_prices_sinkhorn = up_profit_data_sinkhorn - up_valuation_sinkhorn
up_prices_sinkhorn_exp = up_profit_data_sinkhorn_exp - up_valuation_sinkhorn_exp

norm(down_match_data_exp[1,:]-down_match_data_maxrow[1,:])
norm(up_prices_sinkhorn-up_prices_sinkhorn_exp)

scatter(up_data[1,:], up_prices_sinkhorn, markersize=2)

scatter(up_data[1,:], up_prices_sinkhorn_exp, markersize=2)

scatter(up_data[1,:], down_match_data_exp[1,:], markersize=2)
scatter(up_data[1,:], down_match_data_maxrow[1,:], markersize=2)

down_prices_sinkhorn = -(down_match_profit_sinkhorn - down_valuation_sinkhorn)
down_prices_sinkhorn_exp = -(down_match_profit_sinkhorn_exp - down_valuation_sinkhorn_exp)


function tao_func(x,h)
    δ=30
    if abs(x)< h^δ
        return 0.
    elseif abs(x)>2*h^δ
        return 1.
    elseif 2*h^δ>abs(x)>h^δ
        return (4*(x-h^δ)^3/h^(3*δ)) - (3*(x-h^δ)^4)/h^(4*δ)
    end
end


@everywhere begin
    const x=$up_data[1,:]
    const y=$down_data[1,:]
    const y_match=$down_data[1,:]
    const price = $price_data_cf
end


m=2
S=cov([y  price])
H_Silverman = (4/(n_firms*(m+2)))^(2/(m+4)) * S
H_Scott = n_firms^(-2/(m+4)) * S

const hx, hp = sqrt.(diag(H_Silverman))

@everywhere const n_sim = 10;

@everywhere function loglikep(p::Array{T}) where {T<:Real}
    β_up_=zeros(T,3,3)
    β_down_=zeros(T,3,3)
    β_up_[1,1]=p[1]
    β_up_[1,2]=p[2]
    # β_up_[1,3]=1
    β_up_[2,3]=1.
    β_down_[1,1]= p[3]
    β_down_[2,1]= p[4]
    β_down_[3,1]= 1

    A_mat_ = β_up_ + β_down_

    Σ_up = [1.0    0;
            0    1.0]
    Σ_down = [1.0    0;
              0      p[5]]

    p_up_ = MvNormal([0;0], Σ_up)
    p_down_ = MvNormal([0;0], Σ_down )

    function solve_draw(i)
        Random.seed!(i*1234)
        price_noise = rand(Normal(0,10.), n_firms)
        up_data_sim = vcat(x', rand(p_up_, n_firms))
        down_data_sim = vcat(y', rand(p_down_, n_firms))
        C = -1*Transpose(up_data_sim) * A_mat_ * down_data_sim;
        a,b,c  = sinkhorn_stabilized1(μ_n, ν_m, C, 1., absorb_tol = 1e3, max_iter = 200, verbose = false)
        c=-c; b=-b;
        down_match_data_maxrow = down_data_sim[:,argmax.(eachrow(a))]
        down_match_data_maxrow_index = argmax.(eachrow(a))
        up_valuation_sinkhorn = diag(up_data_sim'*β_up_*down_match_data_maxrow)
        down_valuation_sinkhorn = diag(up_data_sim'*β_down_*down_match_data_maxrow)
        down_match_profit_sinkhorn = c[down_match_data_maxrow_index[:]]
        pairwise_production_profits_sinkhorn = b + down_match_profit_sinkhorn
        pairwise_production_sinkhorn= diag(up_data_sim'*A_mat_*down_match_data_maxrow)
        production_bias = pairwise_production_sinkhorn - pairwise_production_profits_sinkhorn

        alpha = .5
        up_profit_data_sinkhorn = b + (alpha .*production_bias)
        down_match_profit_sinkhorn = down_match_profit_sinkhorn +((1-alpha) .*production_bias)
        mindprof = minimum(down_match_profit_sinkhorn)
        down_match_profit_sinkhorn = down_match_profit_sinkhorn .- mindprof
        up_profit_data_sinkhorn = up_profit_data_sinkhorn .+ mindprof
        up_prices_sinkhorn = up_profit_data_sinkhorn - up_valuation_sinkhorn
        return up_data_sim[1,:], down_data_sim[1,:], down_match_data_maxrow[1,:], up_prices_sinkhorn
    end

    sim_dat = pmap(solve_draw,1:n_sim)
    return sim_dat[1][3][2]
    count_zero =0
    count_half_zero=0
    ll=0.
    for i = 1:n_firms
        like=0.
        for j=1:n_sim
            like+=pdf(Normal(),(y_match[i] - sim_dat[j][3][i])/hx)
            # *(pdf(Normal(),(price[i] - sim_dat[j][4][i])/hp))
        end

        if like/(n_sim*hx*hp)==0
            # ll+=log(1e-100)
            ll+= log(2.5e-250)
            # ll+=0.
            println("price is: ", price[i])
            count_zero+=1
            # println("Value close to zero: ", like )
        else
            ll+= log(like/(n_sim*hx*hp))
        end
        #
        # if tao_func(like/(n_sim*hx*hp),hx) ==0
        #     ll+=0.
        #     count_zero+=1
        #
        #     # println("low like val")
        #     # ll +=  log(like/(n_sim*hx*hp))
        # else
        #     # ll +=  log(like/(n_sim*hx*hp))
        #     if tao_func(like/(n_sim*hx*hp),hx)<1.0
        #         println("tao func is: ", tao_func(like/(n_sim*hx*hp),hx))
        #     end
        #
        #     ll += tao_func(like/(n_sim*hx*hp),hx) * log(like/(n_sim*hx*hp))
        # end
    end

        # println("parameter value: ", p, "likelihood value: ", -ll*(1/n_firms))
        println("total number of zero: ", count_zero)
        # println("Call to the function with: ", p )
        # println("likelihood value: ", -ll*(1/n_firms))
        return -ll*(1/n_firms)
end

@show true_like = loglikep([1.00112, .5, 3.,-1.8,1.])

grad_fun = x -> ForwardDiff.gradient(loglikep, x)

grad_fun([1.000, .5, 3.,-1.8,1.])
cand = [-0.591517468108101, 2.8141919046245603, 1.0295406719019349, 2.038108051264322]




n_points=50
par_range = range(-10,10., length=n_points)
likevec = zeros(n_points)
    # jacobian(central_fdm(5, 1), loglikep,  vcat(par_range[2],cand[2:4],1))[1][1,5]
for i =1:n_points
    println("This is i = ", i)
    # [0.5416550765453207, -0.09899434663251219, 1.862759698103865, 0.9068518984843513, 1.0]
    # likevec[i] = loglikep([par_range[i],-0.09899434663251219, 1.862759698103865, 0.9068518984843513, 1.])
    # likevec[i] = loglikep([par_range[i], .5, 3., -1.8, 1.])
    # likevec[i] = loglikep([2., .5, par_range[i], -1.8, 1.])
    # likevec[i] = loglikep([2., .5, 3.0, par_range[i], 1.])

    # likevec[i] = loglikep([2.,par_range[i], 3., -1.8, 1.])
    likevec[i] = loglikep(vcat(par_range[i],cand[2:4],1))

    # likevec[i] = grad_fun(vcat(par_range[i],cand[2:4],1))[1]
    # println("derivative at ", par_range[i], " is: ", likevec[i])
    println("func value  at ", par_range[i], " is: ", likevec[i])
    # likevec[i]=jacobian(central_fdm(5, 1), loglikep,  vcat(par_range[i],cand[2:4],1))[1][1,1]
end
# likevec
# scatter(par_range, likevec, markersize=2,legends=false, title="Sinkhorn High Precision")
scatter(par_range, likevec, markersize=2,legends=false, title="Match 1's price for different values of β_1 in Sim #1")

# scatter(par_range, likevec, markersize=2,legends=false, title="Sinkhorn Mid Precision, over smoothing")
# scatter(par_range, likevec, markersize=2,legends=false, title="Likelihood Trimming Low")
#
# scatter(par_range, likevec, markersize=2,legends=false, title="Likelihood Trimming mid")
#
# scatter(par_range, likevec, markersize=2,legends=false, title="Likelihood Trimming delta 5")
#
#
#
# loglike_ld= b->loglikep([b[1], b[2], b[3], b[4], 1.])
#
#
# res_local = Optim.optimize(loglike_ld, ones(4), LBFGS(),Optim.Options(show_every = 1, time_limit=3600, store_trace = true, show_trace = true, extended_trace=true); autodiff=:forward)
# Optim.minimum(res_local)
# cand=Optim.minimizer(res_local)
#
#
# loglike_ld= b->loglikep([b[1], b[2], b[3], b[4], 1.])
# println("Likelihood value at true parametes: ", true_like)
#
#
# p_range = [(-3.,3.),(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0),(0.0005,15.)];
# res_global = bboptimize(loglikep; SearchRange = p_range, NumDimensions = 5,PopulationSize = 40, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 11000.0)
# cand_global = best_candidate(res_global)
# println("Min Candidate by GS: ", cand_global)
# println("Min value by GS: ", best_fitness(res_global))
# println("Now starting the local solver")

# res_local = Optim.optimize(loglikep, cand_global, LBFGS(),Optim.Options(show_every = 1, x_tol =1e-4, time_limit=3600, store_trace = true, show_trace = true, extended_trace=true); autodiff=:forward)
# println("Min Candidate by LS: ", Optim.minimizer(res_local))
# println("Min value by LS: ", Optim.minimum(res_local))
#
#
# res = Dict()
# push!(res, "global sol" => cand_global)
# push!(res, "local sol" => Optim.minimizer(res_local))
# push!(res, "global min" => best_fitness(res_global))
# push!(res, "local min" => Optim.minimum(res_local))
# push!(res, "true like" => true_like)
#
#
# solution = copy(BlackBoxOptim.best_candidate(bbsolution))
# push!(bbo, "solution" => solution)
# fitness = copy(BlackBoxOptim.best_fitness(bbsolution))
# push!(bbo, "fitness" => fitness)
# bson("/outputs/estimation_results_$(task_id).bson", bbo)
