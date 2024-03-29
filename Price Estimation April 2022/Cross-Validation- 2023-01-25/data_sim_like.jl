function sim_data_like(up_x, β_up, β_down, Σ_up, Σ_down, n_firms, i, in_med)
    # β_up=zeros(3,3)
    # β_down=zeros(3,3)
    # β_up[1,1]=pup[1]
    # β_up[1,2]=pup[2]
    # β_up[2,3]=pup[3]
    # # β_up[1,3]=pup[3]

    # β_down[1,1]= pdown[1]
    # β_down[2,1]= pdown[2]
    # β_down[3,1]= pdown[3]

    A_mat = β_up + β_down
    β_diff = β_down - β_up

    # Σ_up = zeros(3,3)
    # Σ_up[1,1]= sigup[1]; Σ_up[2,2]=sigup[2];  Σ_up[3,3]=sigup[3];
    # Σ_down = zeros(3,3)
    # Σ_down[1,1]= sigdown[1]; Σ_down[2,2]=sigdown[2];  Σ_down[3,3]=sigdown[3];
    p_up = MvNormal([0; 0;0], Σ_up)
    p_down = MvNormal([0; 0;0], Σ_down)


    # Solving for equilibrium and generating the matching (fake) data
    # These are closed form formulas from the Galichon's paper
    d_matrix1 = sqrt(Σ_down)* transpose(A_mat) * Σ_up * A_mat * sqrt(Σ_down)
    inv_d_1 = inv(d_matrix1)
    sqrt_inv_d_1 = sqrt(inv_d_1)
    if sum(imag(sqrt_inv_d_1))!=0.0
        println("error imaginary")
    else
        sqrt_inv_d_1=real(sqrt_inv_d_1)
    end

    # t-matrix is the equilibrium assignment matrix for all dimensions except the--
    #-- additive error term
    t_matrix = sqrt(Σ_down)*sqrt_inv_d_1*sqrt(Σ_down)*transpose(A_mat)

    # Data Generation

    Random.seed!(1234+i)
    up_data = vcat(up_x, (rand(p_up, n_firms)[3,:])')
    down_matched_data_cf = t_matrix*up_data
    down_data=copy(down_matched_data_cf)
    up_profit_data_cf = Float64[]
    down_profit_data_cf = Float64[]
    price_data_cf=Float64[]
    for i =1:n_firms
        push!(price_data_cf, 0.5*Transpose(up_data[:,i]) * β_diff * down_matched_data_cf[:,i])
        push!(up_profit_data_cf, 0.5*Transpose(up_data[:,i]) * A_mat * down_matched_data_cf[:,i])
        push!(down_profit_data_cf, 0.5*Transpose(up_data[:,i]) * A_mat * down_matched_data_cf[:,i])
    end


    ## median of downstream profits
    mid_down = median(down_profit_data_cf)
    
    ## difference between the input median and the down profit median

    down_profit_diff = mid_down .- in_med



    down_profit_data_cf = down_profit_data_cf .- down_profit_diff
    up_profit_data_cf = up_profit_data_cf .+ down_profit_diff
    price_data_cf = price_data_cf .+ down_profit_diff

    return up_data, down_data, price_data_cf

end
