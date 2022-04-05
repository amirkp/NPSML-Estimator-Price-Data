
using Optim
using LinearAlgebra
using Random
using Distributions
using BlackBoxOptim
using LineSearches
using Plots
# @everywhere include("data_sim_seed.jl")
# include("data_sim_like.jl")
include("data_sim_like_2d_2d_diff.jl")
include("data_sim_like_2d_2d_match_only.jl")

n_firms=500

# @everywhere function replicate_byseed(n_rep)

bup = [1. 1.5 0;
       .5 2.5 0;
       0.0 0  0. ]
bdown = [-2.5 -2 0;
        1  -1.5 0;
        0 0 1]
B= bup+bdown

up_data, down_data, price_data_cf, tmat =
    sim_data_like( -1, bup, bdown, [2, 1., 1.5], [.5, 3, 1.], n_firms, 205, 2.5)
tmat

trpar = [1., 1.5, .5, 2.5,-2.5,-2, 1, -1.5, 1.5]
function loglikep(b)
    n_sim=500

    A = [
    vcat(b[1:2],b[4])';
    vcat(b[3],1, 0)';
    vcat(b[5], 0, 1)'
    ]
    # bup = [
    #     vcat(b[1:2],0)';
    #     vcat(b[3:4], 0)';
    #     vcat(0, 0, 0)'
    # ]
    #
    #
    # bdown = [
    #     vcat(b[5], b[6],0)';
    #     vcat(b[7],b[8], 0)';
    #     vcat(0, 0, 1)'
    # ]
    # B1 = bup+bdown
    # norm_ = norm(B[1:2,1:2])/norm(B1[1:2,1:2])
    # bup[1:2,1:2] = bup[1:2,1:2]*norm_
    # bdown[1:2,1:2] = bdown[1:2,1:2]*norm_

    # bup = [b[1:3]' ;
    #         b[4:6]'  ;
    #         vcat(0.5,0,0)']
    # bdown = [b[7:9]' ;
    #         b[10:12]'  ;
    #         vcat(0,0,1)' ]
 #
 # bup = [b[1:3]' ;
 #         b[4:6]'  ;
 #         vcat(b[7],0,0)']
 # bdown = [b[10:12]' ;
 #         b[13:15]'  ;
 #         vcat(b[16],0,1)' ]
    solve_draw = x->sim_data_like_mo(up_data[1:2,:],A, [2, 1., 1.5], [.5, 3, 1.], n_firms, 1234+x,2.5)
    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0
    h=[0.2, 0.2, .3]

    for i =1:n_firms
        like =0.
        for j =1:n_sim
            if sim_dat[j]==99999
                return 200.
            end


            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                # *pdf(Normal(),((price_data_cf[i] - sim_dat[j][3][i])/h[3]))
                )
        end
        ll+=log(like/(n_sim*h[1]*h[2]))
    end
    println("parameter: ", b, " function value: ", -ll/n_firms)
    return -ll/n_firms
end



########################
##################
#### MAIN LIKELIHOOD FUNCTION
#############
######################
###################
mu_price = mean(price_data_cf)
function loglikepr(b)
    n_sim=100

    bup = [
        vcat(b[1:2],0)';
        vcat(b[3:4], 0.)';
        vcat(0., 0, 0)'
    ]


    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], b[8], 0)';
        vcat(0. ,0., 1)'
     ]

    solve_draw = x->sim_data_like(up_data[1:2,:],bup, bdown , [2, 1., abs.(b[9])], [.5, 3,1.], n_firms, 1234+x, 2.5)
    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0
    h=[0.1, 0.2, 1.5]
    for j=1:n_sim
        pconst = mean(sim_dat[j][3])-mu_price
        sim_dat[j][3][:] = sim_dat[j][3] .+ pconst
    end

    for i =1:n_firms
        like =0.
        for j =1:n_sim
            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                *pdf(Normal(),((price_data_cf[i] - sim_dat[j][3][i])/h[3]))
                )
        end
        ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
    end
    println("parameter: ", round.(b,digits=3), " function value: ", round(-ll/n_firms,digits=5))
    return -ll/n_firms
end

loglikepr(trpar)
res_1 = Optim.optimize(loglikepr,tp1)
res_1 = Optim.optimize(loglikepr, rand(9))
res_1 = Optim.optimize(loglikepr, trpar)
