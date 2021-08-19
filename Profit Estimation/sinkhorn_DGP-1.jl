function sim_data_sinkhorn(β_up, β_down, Σ_up, Σ_down, n_firms,i)
    function getK1(C, alpha, beta, eps, mu, nu)
        return (exp.(-(C .- alpha .- beta')/eps).*mu.*nu')
    end

    function sinkhorn_stabilized1(mu, nu, C, eps; absorb_tol = 1e3, max_iter = 10000, tol = 1e-9, alpha = nothing, beta = nothing, return_duals = false, verbose = true)
        if isnothing(alpha) || isnothing(beta)
            alpha = zeros(size(mu)); beta = zeros(size(nu))
        end

        u = ones(size(mu)); v = ones(size(nu))
        K = (exp.(-(C)/eps).*mu.*nu')
        i = 0

        while true
            u = mu./(K*v )
            v = nu./(K'*u )

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

    Random.seed!(12345+i)
    up_data = zeros(2, n_firms)
    up_data[1,:] = rand(LogNormal(Σ_up[1,1], Σ_up[1,2]), n_firms)
    up_data[2,:] = rand(LogNormal(Σ_up[2,1], Σ_up[2,2]), n_firms)

    down_data = zeros(2, n_firms)
    down_data[1,:] = rand(LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms)
    down_data[2,:] = rand(LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms)

    A_mat = β_up + β_down

    C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus


    μ_n = ones(n_firms)/n_firms;
    ν_m = ones(n_firms)/n_firms;

    a,b,c  = sinkhorn_stabilized1(μ_n, ν_m, C, .2, absorb_tol = 1e3, max_iter = 10000, verbose = true)
    c=-c; b=-b;


    down_match_data_maxrow = down_data[:,argmax.(eachrow(a))]
    down_match_data_exp = n_firms*Transpose(a*Transpose(down_data))



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

    return up_data, down_match_data_exp, up_profit_data_sinkhorn, down_match_profit_sinkhorn
end
