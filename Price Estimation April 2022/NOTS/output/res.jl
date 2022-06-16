using BSON
using DataFrames
using BlackBoxOptim

############################3333
######################################
function res_fun(opt_vec)
    MC_est= [best_candidate(opt_vec[i])  for i = 1:n_reps ]
    MC_fit= [best_fitness(opt_vec[i]) for i = 1:n_reps ]
    MC_est= reduce(vcat, MC_est')
    MC_est =hcat(MC_est, MC_fit)
    return MC_est
end

# BSON.load("/Users/akp/output/MC-LN/est_100_sim_25.bson")
out_200 = BSON.parse("/Users/akp/output/MC-LN/est_100_sim_25.bson")[:data];

n_reps=100
out_200 = out_200[2][1];

est_200 = res_fun(out_200);

pars10 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "eqsel", "llike"];
est_200 = DataFrame(est_200, pars10);


sort!(est_200, ["llike"]);
/Users/akp/output/MC-LN