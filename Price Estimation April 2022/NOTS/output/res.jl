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
out = BSON.load("/Users/akp/output/02/est_100_sim_25.bson")

# n_reps=100
est = out["beta_hat"];
est
# fit = out["fitness"]
# hcat(est, fit)

bias = mean(est, dims=1) -true_pars'
MSE = sqrt.(mean((est .- true_pars').^2, dims=1))



out_200
est_200 = res_fun(out_200);
est = reduce(hcat, est)
est[1,2]
est[1]


pars10 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "eqsel", "llike"];
est_200 = DataFrame(est_200, pars10);


sort!(est_200, ["llike"]);
/Users/akp/output/MC-LN