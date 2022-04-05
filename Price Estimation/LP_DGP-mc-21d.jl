function sim_data_LP(up_, down_,β_up, β_down, A_mat, Σ_up, Σ_down, n_firms,i)
    Random.seed!(12345+i)

    up_data = zeros(2, n_firms)
    up_data[1,:] = up_
    up_data[2,:] = rand(LogNormal(Σ_up[1], Σ_up[2]), n_firms)
    # up_data[3,:] = rand(LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms)

    down_data = zeros(3, n_firms)
    down_data[1,:] = down_[1,:]
    down_data[2,:] = down_[2,:]
    down_data[3,:] = rand(LogNormal(Σ_down[1], Σ_down[2]), n_firms)

    # A_mat = β_up + β_down
    C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus
    # The below loop adds in the fixed effects commented out for now
    # for j = 1:n_firms
    #     for i = 1:n_firms
    #         C[i,j] += -(bx * up_data[1,i] + by[1]*down_data[1,j]+ by[2]*down_data[2,j])
    #     end
    # end




    model = Model(optimizer_with_attributes(Gurobi.Optimizer, "Threads" => 1));
    MOI.set(model, MOI.Silent(), true)
    @variable(model, 0<= matching_mat[i=1:n_firms,j=1:n_firms]<=1);
    @constraint(model ,conD[i=1:n_firms], sum(matching_mat[:,i])==1);
    @constraint(model ,conU[i=1:n_firms], sum(matching_mat[i,:])==1);
    @objective(model,Min, sum(matching_mat.*C) )
    optimize!(model)
    γ = value.(matching_mat);
    u = -dual.(conU);
    v = -dual.(conD);
    down_match_data_lp = Transpose(γ *Transpose(down_data))


    minv= minimum(v)
    up_profit_data_lp= u.+ minv
    down_profit_data_lp= v.- minv
    # println("Minimum down profit was: ", minv, " is: ", minimum(v))

    down_match_profit_data_lp =  γ*down_profit_data_lp
    up_valuation = diag(up_data'*β_up*down_match_data_lp)
    up_prices_lp = up_profit_data_lp - up_valuation
    # down_prices_lp= γ'*up_prices_lp

    # return up_data, down_match_data_lp, up_profit_data_lp, down_match_profit_data_lp, down_data
    return up_data, down_match_data_lp[1,:], down_match_data_lp[2,:],  up_prices_lp
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
