#LOG NORMAL DGP
function sim_data_JV(β_up, β_down, Σ_up, Σ_down, n_firms,i, flag, obs_up, obs_down)
    if flag == false

        up_data = Array{Float64, 2}(undef, 3, n_firms)
        Random.seed!(1234+i)
        up_data[1,:] = rand(LogNormal(Σ_up[1,1], Σ_up[1,2]), n_firms)
        up_data[2,:] = rand(LogNormal(Σ_up[2,1], Σ_up[2,2]), n_firms)
        up_data[3,:] = rand(LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)


        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1,:] = rand(LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms)
        down_data[2,:] = rand(LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms)
        down_data[3,:] = rand(LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)
    elseif flag==true # take observed data as given (do not generate observed chars)
        Random.seed!(1234+i)
        up_data = Array{Float64, 2}(undef, 3, n_firms)
        up_data[1:2,:] = obs_up
        up_data[3,:] = rand(LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)

        Random.seed!(1234+i)
        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1:2,:] = obs_down
        down_data[3,:] = rand(LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)
    end

    #
    A_mat = β_up + β_down
    C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus

    # C=rand(500,500)
    match, up_profit_data, down_profit_data = find_best_assignment(C)

    down_match_data=  Array{Float64, 2}(undef, 3, n_firms)
    for i=1:n_firms
        down_match_data[:,i] = down_data[:, match[i][2]]
    end

    down_match_profit_data =  Array{Float64, 1}(undef, n_firms)
    for i=1:n_firms
        down_match_profit_data[i] = down_profit_data[match[i][2]]
    end


    up_valuation = diag(up_data'*β_up*down_match_data)
    up_prices = up_profit_data - up_valuation

    return up_data, down_match_data, up_prices
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
function sim_data_JV_up_obs(β_up, β_down, Σ_up, Σ_down, n_firms,i, flag, obs_up)
    if flag == false

        up_data = Array{Float64, 2}(undef, 3, n_firms)
        Random.seed!(1234+i)
        up_data[1,:] = rand(LogNormal(Σ_up[1,1], Σ_up[1,2]), n_firms)
        up_data[2,:] = rand(LogNormal(Σ_up[2,1], Σ_up[2,2]), n_firms)
        up_data[3,:] = rand(LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)


        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1,:] = rand(LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms)
        down_data[2,:] = rand(LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms)
        down_data[3,:] = rand(LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)
    elseif flag==true # take observed data as given (do not generate observed chars)
        Random.seed!(1234+i)
        up_data = Array{Float64, 2}(undef, 3, n_firms)
        up_data[1:2,:] = obs_up
        up_data[3,:] = rand(LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)

        Random.seed!(1234+i)
        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1,:] = rand(LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms)
        down_data[2,:] = rand(LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms)
        down_data[3,:] = rand(LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)
    end

    #
    A_mat = β_up + β_down
    C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus

    # C=rand(500,500)
    match, up_profit_data, down_profit_data = find_best_assignment(C)

    down_match_data=  Array{Float64, 2}(undef, 3, n_firms)
    for i=1:n_firms
        down_match_data[:,i] = down_data[:, match[i][2]]
    end

    down_match_profit_data =  Array{Float64, 1}(undef, n_firms)
    for i=1:n_firms
        down_match_profit_data[i] = down_profit_data[match[i][2]]
    end


    up_valuation = diag(up_data'*β_up*down_match_data)
    up_prices = up_profit_data - up_valuation

    return up_data, down_match_data, up_prices
end
