#LOG NORMAL DGP
function sim_data_JV_LogNormal(β_up, β_down, Σ_up, Σ_down, n_firms,i, flag, obs_up, obs_down, d_min_prof_input)
    if flag == false

        up_data = Array{Float64, 2}(undef, 3, n_firms)
        up_data[1,:] = rand(Random.seed!(1234+i), LogNormal(Σ_up[1,1], Σ_up[1,2]), n_firms)
        up_data[2,:] = rand(Random.seed!(1234+2i), LogNormal(Σ_up[2,1], Σ_up[2,2]), n_firms)
        up_data[3,:] = rand(Random.seed!(1234+3i), LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)


        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1,:] = rand(Random.seed!(1234+4i), LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms)
        down_data[2,:] = rand(Random.seed!(1234+5i), LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms)
        down_data[3,:] = rand(Random.seed!(1234+6i), LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)

    elseif flag==true # take observed data as given (do not generate observed chars)
        
        up_data = Array{Float64, 2}(undef, 3, n_firms)
        up_data[1:2,:] = obs_up
        up_data[3,:] = rand(Random.seed!(1234+i), LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)

        
        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1:2,:] = obs_down
        down_data[3,:] = rand(Random.seed!(1234+6i), LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)
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


    # d_min_prof = findmin(down_match_profit_data)[1]
    d_min_prof = median(down_match_profit_data)

    profit_diff = d_min_prof_input - d_min_prof
    down_match_profit_data .= down_match_profit_data .+ profit_diff

    down_valuation = diag(up_data'*β_down*down_match_data) 
    down_prices = down_valuation - down_match_profit_data 


    return up_data, down_match_data, down_prices, up_profit_data, down_match_profit_data
end


function sim_data_JV_up_obs(β_up, β_down, Σ_up, Σ_down, n_firms,i, flag, obs_up, d_min_prof_input)
    
    if flag == false
        up_data = Array{Float64, 2}(undef, 3, n_firms)
    
        up_data[1,:] = rand(Random.seed!(1234+i), LogNormal(Σ_up[1,1], Σ_up[1,2]), n_firms)
        up_data[2,:] = rand(Random.seed!(1234+2i), LogNormal(Σ_up[2,1], Σ_up[2,2]), n_firms)
        up_data[3,:] = rand(Random.seed!(1234+3i), LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)
        

        down_data = Array{Float64, 2}(undef, 3, n_firms)
        down_data[1,:] = rand(Random.seed!(1234+4i), LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms)
        down_data[2,:] = rand(Random.seed!(1234+5i), LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms)
        down_data[3,:] = rand(Random.seed!(1234+6i), LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)
    elseif flag==true # take observed data as given (do not generate observed chars)
        
        up_data = Array{Float64, 2}(undef, 3, n_firms)
        up_data[1:2,:] = obs_up[1:2,:]
        
        up_data[3,:] = rand(Random.seed!(1234+3i),LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)

        
        down_data = Array{Float64, 2}(undef, 3, n_firms)
        
        down_data[1,:] = rand(Random.seed!(1234+4i), LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms)
        down_data[2,:] = rand(Random.seed!(1234+5i), LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms)
        down_data[3,:] = rand(Random.seed!(1234+6i), LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms)
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

    # d_min_prof = findmin(down_match_profit_data)[1]
    d_min_prof = median(down_match_profit_data)


    profit_diff = d_min_prof_input - d_min_prof
    down_match_profit_data .= down_match_profit_data .+ profit_diff

    down_valuation = diag(up_data'*β_down*down_match_data) 
    down_prices = down_valuation - down_match_profit_data 

    # up_valuation = diag(up_data'*β_up*down_match_data)
    # up_prices = up_profit_data - up_valuation

    # return up_data, down_match_data, up_prices, up_profit_data, down_match_profit_data
    return up_data, down_match_data, down_prices, up_profit_data, down_match_profit_data
end
