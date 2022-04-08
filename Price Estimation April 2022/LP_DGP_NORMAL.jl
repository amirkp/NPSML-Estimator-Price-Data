using Gurobi
using JuMP
using Distributions
using Random
using LinearAlgebra
using Hungarian
using Assignment
using BenchmarkTools
# model = JuMP.direct_model(Gurobi.Optimizer());

n_firms = 500
model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "Presolve", -1)
set_optimizer_attribute(model, "OutputFlag", 1)


MOI.set(model, MOI.Silent(), true)

@variable(model, 0<= matching_mat[i=1:n_firms,j=1:n_firms]<=1);
@constraint(model ,conD[i=1:n_firms], sum(matching_mat[:,i])==1);
@constraint(model ,conU[i=1:n_firms], sum(matching_mat[i,:])==1);





# Random.seed!(1234+i)


β_up = [1. 1.5 -1;
       .5 2.5 0;
      0 0  0 ]
β_down = [2.5 -2 0;
        1  0 0;
        0 0 .5]

Σ_up = diagm([2,1,1.])
Σ_down = diagm([.5,3,1])

up_data = zeros(3, n_firms)
up_data[1,:] = rand(Normal(0,Σ_up[1,1]), n_firms)
up_data[2,:] = rand(Normal(0,Σ_up[2,2]), n_firms)
up_data[3,:] = rand(Normal(0,Σ_up[3,3]), n_firms)

down_data = zeros(3, n_firms)
down_data[1,:] = rand(Normal(0, Σ_down[1,1]), n_firms)
down_data[2,:] = rand(Normal(0, Σ_down[2,2]), n_firms)
down_data[3,:] = rand(Normal(0, Σ_down[3,3]), n_firms)


A_mat = β_up + β_down
C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus

function gt(i)
    @objective(model,Min, sum(matching_mat.*(randn(500,500))) )
    optimize!(model)
end


@benchmark gt(data) setup=(data=rand(1:30))

@benchmark hungarian(data) setup=(data=rand(40,40))

@benchmark find_best_assignment(data) setup=(data=rand(40,40))


@time






sol = find_best_assignment(rand(3,3))

sol[1]


sol[4]




sol


c = rand(1:22, 3,3)


c=-c

c





sol2 = find_best_assignment(c, true)

sol[4]






ui = sol[3]



ui= -ui




vj= sol[2]



vj=-vj


(1,2) = 6       -2 + 6 = 4 --> 0, 6

(2,5) = 6       0 + 6  = 6



sum(ui + vj)


ui = ui .+ 26


vj = vj .+ 26


sol = find_best_assignment(c)

Assignment.Assignment_Solution(c,c)

@objective(model,Min, sum(matching_mat.*C +randn(500,500)) )
optimize!(model)
using BenchmarkTools
@benchmark gt(1)
γ = value.(matching_mat);
u = -dual.(conU);
v = -dual.(conD);
down_match_data_lp = Transpose(γ *Transpose(down_data))


scatter(up_data[1,:], down_match_data_lp[3,:])



set_optimizer_attribute(model, "Presolve", 2)
using FLoops
# @floop begin
    for i = 1:10
        Random.seed!(1234+i)


        up_data = zeros(3, n_firms)
        up_data[1,:] = rand(Normal(0,Σ_up[1,1]), n_firms)
        up_data[2,:] = rand(Normal(0,Σ_up[2,2]), n_firms)
        up_data[3,:] = rand(Normal(0,Σ_up[3,3]), n_firms)

        down_data = zeros(3, n_firms)
        down_data[1,:] = rand(Normal(0, Σ_down[1,1]), n_firms)
        down_data[2,:] = rand(Normal(0, Σ_down[2,2]), n_firms)
        down_data[3,:] = rand(Normal(0, Σ_down[3,3]), n_firms)
        C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus
        @objective(model,Min, sum(matching_mat.*C ) )
        optimize!(model);
        println("model ", i)
    end
# end


@floop begin
    for i = 1:10
        Random.seed!(1234+i)

        #
        # up_data = zeros(3, n_firms)
        # up_data[1,:] = rand(Normal(0,Σ_up[1,1]), n_firms)
        # up_data[2,:] = rand(Normal(0,Σ_up[2,2]), n_firms)
        # up_data[3,:] = rand(Normal(0,Σ_up[3,3]), n_firms)
        #
        # down_data = zeros(3, n_firms)
        # down_data[1,:] = rand(Normal(0, Σ_down[1,1]), n_firms)
        # down_data[2,:] = rand(Normal(0, Σ_down[2,2]), n_firms)
        # down_data[3,:] = rand(Normal(0, Σ_down[3,3]), n_firms)

        C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus
        @objective(model,Min, sum(matching_mat.*(C+randn(500,500)) ) )
        optimize!(model);
        println("model ", i)
    end
end
function tst(i)
    Random.seed!(1234+i)

    #
    up_data = zeros(3, n_firms)
    up_data[1,:] = rand(Normal(0,Σ_up[1,1]), n_firms)
    up_data[2,:] = rand(Normal(0,Σ_up[2,2]), n_firms)
    up_data[3,:] = rand(Normal(0,Σ_up[3,3]), n_firms)

    down_data = zeros(3, n_firms)
    down_data[1,:] = rand(Normal(0, Σ_down[1,1]), n_firms)
    down_data[2,:] = rand(Normal(0, Σ_down[2,2]), n_firms)
    down_data[3,:] = rand(Normal(0, Σ_down[3,3]), n_firms)

    C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus
    # @objective(model,Min, sum(matching_mat.*(C+randn(500,500)) ) )
    # optimize!(model);
    println("model ", i)
end

MOI.set(model, MOI.Silent(), true)
map(tst,1:10)

matching = Hungarian.munkres(C)

@objective(model,Min, sum(matching_mat.*C ) )
optimize!(model);
m2 = value.(matching_mat);
m2

m2 -matching
m2*ones(500)
m2[3,229]

function sim_data_LP(β_up, β_down, Σ_up, Σ_down, n_firms,i)
    up_data = zeros(3, n_firms)
    Random.seed!(1234+i)
    up_data[1,:] = rand(Normal(0,Σ_up[1,1]), n_firms)
    up_data[2,:] = rand(Normal(0,Σ_up[2,2]), n_firms)
    up_data[3,:] = rand(Normal(0,Σ_up[3,3]), n_firms)

    down_data = zeros(3, n_firms)
    down_data[1,:] = rand(Normal(0, Σ_down[1,1]), n_firms)
    down_data[2,:] = rand(Normal(0, Σ_down[2,2]), n_firms)
    down_data[3,:] = rand(Normal(0, Σ_down[3,3]), n_firms)

    A_mat = β_up + β_down
    C = -1*Transpose(up_data)*A_mat*down_data #pairwise surplus

    model = JuMP.direct_model(Gurobi.Optimizer());
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


    # minv= minimum(v)
    # up_profit_data_lp= u.+ minv
    up_profit_data_lp= u
    # down_profit_data_lp= v.- minv
    down_profit_data_lp = v
    # println("Minimum down profit was: ", minv, " is: ", minimum(v))

    down_match_profit_data_lp =  γ*down_profit_data_lp
    up_valuation = diag(up_data'*β_up*down_match_data_lp)
    up_prices_lp = up_profit_data_lp - up_valuation
    down_prices_lp= γ'*up_prices_lp

    return up_data, down_match_data_lp, up_profit_data_lp, down_match_profit_data_lp, down_data
end
#

bup = [1. 1.5 -1;
       .5 2.5 0;
      0 0  0 ]
bdown = [2.5 -2 0;
        1  0 0;
        0 0 .5]

sigup = diagm([2,1,1.])
sigdown = diagm([.5,3,1])

@benchmark sim_data_LP(bup, bdown, sigup, sigdown, 500, 2)











@benchmark sim_data_LP(bup, bdown, sigup, sigdown, 500, 2)
ui
