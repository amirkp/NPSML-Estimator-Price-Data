using Assignment 
using Gurobi
using JuMP
using Random
using Distributions
using Plots

function par_gen(b)
    bup = [
        vcat(b[1:2],b[8])';
        vcat(b[3:4], 0.)';
        vcat(0 , 0, 0)'
    ]


    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., b[9] )'
    ]

    return bup, bdown
end

true_pars =  [-2.5, 1.5, -1.5, -.5, 3.5, 2.5, 1.5, 3, -3];

β_up, β_down = par_gen(true_pars);

n_firms=500;
i = 10;
Σ_up = [0 .1;
        0 .2;
        0 .1];


Σ_down =  [0 .3;
          0 .4;
          0 .1];


up_data = Array{Float64, 2}(undef, 3, n_firms);
up_data[1,:] = rand(Random.seed!(1234+i), LogNormal(Σ_up[1,1], Σ_up[1,2]), n_firms);
up_data[2,:] = rand(Random.seed!(1234+2i), LogNormal(Σ_up[2,1], Σ_up[2,2]), n_firms);
up_data[3,:] = rand(Random.seed!(1234+3i), LogNormal(Σ_up[3,1], Σ_up[3,2]), n_firms);


down_data = Array{Float64, 2}(undef, 3, n_firms);

down_data[1,:] = rand(Random.seed!(1234+4i), LogNormal(Σ_down[1,1], Σ_down[1,2]), n_firms);
down_data[2,:] = rand(Random.seed!(1234+5i), LogNormal(Σ_down[2,1], Σ_down[2,2]), n_firms);
down_data[3,:] = rand(Random.seed!(1234+6i), LogNormal(Σ_down[3,1], Σ_down[3,2]), n_firms);

A_mat = β_up + β_down;
C = -1*Transpose(up_data)*A_mat*down_data;


####################################
### Gurobi- Linear Program #########
####################################


model = Model(optimizer_with_attributes(Gurobi.Optimizer));
MOI.set(model, MOI.Silent(), true);
@variable(model, 0<= matching_mat[i=1:n_firms,j=1:n_firms]<=1);
@constraint(model ,conD[i=1:n_firms], sum(matching_mat[:,i])==1);
@constraint(model ,conU[i=1:n_firms], sum(matching_mat[i,:])==1);
@objective(model,Min, sum(matching_mat.*C) );
optimize!(model);
γ = getvalue.(matching_mat);
u = -dual.(conU);
v = -dual.(conD);
down_match_data_lp = Transpose(γ *Transpose(down_data));

#Lowest downstream profit =0 

minv= minimum(v);
up_profit_data_lp= u.+ minv;
down_profit_data_lp= v.- minv;
down_match_profit_data_lp =  γ*down_profit_data_lp;




##################################################
########### JV Algorithm #########################
##################################################

match, up_profit_data, down_profit_data = find_best_assignment(C);
down_match_data=  Array{Float64, 2}(undef, 3, n_firms);
for i=1:n_firms
    down_match_data[:,i] = down_data[:, match[i][2]];
end

down_match_profit_data =  Array{Float64, 1}(undef, n_firms);
for i=1:n_firms
    down_match_profit_data[i] = down_profit_data[match[i][2]];
end

# minimum downstream profit =0 
profit_diff = findmin(down_match_profit_data)[1]
down_match_profit_data .= down_match_profit_data .+ profit_diff;



## Matching is the same (returns true)
down_match_data == down_match_data_lp

# Profits are approximately the same, except for small error
# plotting downstream profits from the two methods against each other
# point are all approximately on the 45-degree line


plot(x->x, 0,10, linewidth=7)
scatter!(down_match_profit_data, down_match_profit_data_lp,
    markersize=2, color=:red)
