function sim_data_JV_Normal(β_up::AbstractMatrix{T}, β_down::AbstractMatrix{T}, Σ_up, Σ_down, n_firms,i, flag, obs_up, obs_down, d_min_prof_input) where T
    if flag == false

        up_data = Array{Float64, 2}(undef, 3, n_firms)
        Random.seed!(1234+i)
        up_data[1,:] = rand(Normal(Σ_up[1,1], sqrt(Σ_up[1,2])), n_firms)
        up_data[2,:] = rand(Normal(Σ_up[2,1], sqrt(Σ_up[2,2])), n_firms)
        up_data[3,:] = rand(Normal(Σ_up[3,1], sqrt(Σ_up[3,2])), n_firms)

        Random.seed!(1234+i)

        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1,:] = rand(Normal(Σ_down[1,1], sqrt(Σ_down[1,2])), n_firms)
        down_data[2,:] = rand(Normal(Σ_down[2,1], sqrt(Σ_down[2,2])), n_firms)
        down_data[3,:] = rand(Normal(Σ_down[3,1], sqrt(Σ_down[3,2])), n_firms)
    elseif flag==true
        Random.seed!(1234+i)
        up_data = Array{Float64, 2}(undef, 3, n_firms)
        up_data[1:2,:] = obs_up
        up_data[3,:] = rand(Normal(Σ_up[3,1], sqrt(Σ_up[3,2])), n_firms)

        Random.seed!(1234+i)

        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1,:] = rand(Normal(Σ_down[1,1], sqrt(Σ_down[1,2])), n_firms)
        down_data[2,:] = rand(Normal(Σ_down[2,1], sqrt(Σ_down[2,2])), n_firms)
        down_data[3,:] = rand(Normal(Σ_down[3,1], sqrt(Σ_down[3,2])), n_firms)
    end

    #
    A_mat = β_up + β_down
    C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus

    # C=rand(500,500)
    match, up_profit_data, down_profit_data = find_best_assignment(C)

    # down_match_data=  Array{T, 2}(undef, 3, n_firms)
    down_match_data= zeros(T, 3, n_firms)

    for i=1:n_firms
        down_match_data[:,i] = down_data[:, match[i][2]]
    end

    # down_match_profit_data =  Array{T, 1}(undef, n_firms)
    down_match_profit_data = zeros(T, n_firms)
    for i=1:n_firms
        down_match_profit_data[i] = down_profit_data[match[i][2]]
    end


    # up_valuation = diag(up_data'*β_up*down_match_data)
    # up_prices = up_profit_data - up_valuation

    d_min_prof = findmin(down_match_profit_data)[1]

    profit_diff = d_min_prof_input - d_min_prof
    down_match_profit_data .= down_match_profit_data .+ profit_diff

    down_valuation = diag(up_data'*β_down*down_match_data) 
    down_prices = down_valuation - down_match_profit_data 


    # return up_data, down_match_data, up_prices
    return up_data, down_match_data, down_prices, up_profit_data, down_match_profit_data

    # , up_profit_data, down_match_profit_data
end
#
#
# Random.seed!(1234)
#
# b_up = [2 0.5; 1. 0.2]
# b_down = [1 0.2; 2. 0.3]
#
# sig_up = [0. 1.5; 0 .6]
# sig_down = [0. 1; 0 .3]
# ud, dd, up, dp,down =  sim_data_LP(b_up,b_down,sig_up,sig_down,1500,2)
