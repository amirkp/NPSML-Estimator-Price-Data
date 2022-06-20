using BSON
using DataFrames
using BlackBoxOptim
using Statistics
############################3333
######################################
# function res_fun(opt_vec)
#     MC_est= [best_candidate(opt_vec[i])  for i = 1:n_reps ]
#     MC_fit= [best_fitness(opt_vec[i]) for i = 1:n_reps ]
#     MC_est= reduce(vcat, MC_est')
#     MC_est =hcat(MC_est, MC_fit)
#     return MC_est
# end

# BSON.load("/Users/akp/output/MC-LN/est_100_sim_25.bson")
out1 = BSON.load("/Users/amir/out/01/est_100_sim_25.bson")
out1 = BSON.load("/Users/amir/github/NPSML-Estimator-Price-Data/Price Estimation April 2022/NOTS/LogNormal/01/est_100_sim_25.bson")

out1 = BSON.load("/Users/amir/out/01/est_100_sim_25.bson")
out1["beta_hat"]
out1["fitness"]
# n_reps=100
est = out1["beta_hat"];
# reduce(hcat,best_candidate.(est))
fit = out1["fitness"]

hcat(est, fit)
function res_fun(PATH, true_pars, verbose=:false)
    out = BSON.load(PATH)
    est = out["beta_hat"];

    fit = out["fitness"]
    if verbose==true
        for i = 1:size(est)[1]
            println(round.(est[i,:], digits=3),round.(fit[i], digits=2))
            sleep(0.5)
        end
    end


    bw = out["bw"]

    bias = mean(est, dims=1) -true_pars'

    MSE = sqrt.(mean((est .- true_pars').^2, dims=1))
    m_h = mean(bw, dims=1)
    # println("True parameters: ", true_pars)
    # sleep(1)
    # println("Bias: ", round.(bias, digits=2))
    # sleep(1)
    println("RMSE: ",round.(MSE, digits=2))
    # sleep(1)
    println("Mean BW: ",round.(m_h, digits=2))
    pars10 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "eqsel", "llike"];
    est = hcat(est, fit);

    est_df = DataFrame(est, pars10);
    return est_df;
end



h_mat = zeros(27, 3)
scales = [0.5 1. 2.]
count = 1
for i1 = 1:3
    for i2 =  1:3
        for i3  = 1:3
            h_mat[count, :] = [scales[i1] scales[i2] scales[i3]]
            global count+=1
        end
    end
end


for h_id = 1:23 
    path= "/Users/amir/out/h_vary1/est_100_sim_25_$(h_mat[h_id, 1])_$(h_mat[h_id, 2])_$(h_mat[h_id, 3]).bson"
    est=  res_fun(path, true_pars);
end

h_id = 1
path= "/Users/amir/out/h_vary1/est_100_sim_25_$(h_mat[h_id, 1])_$(h_mat[h_id, 2])_$(h_mat[h_id, 3]).bson"
est=  res_fun(path, true_pars, true);













path = "/Users/amir/out/02/est_100_sim_25.bson";
est_100_25 = res_fun(path, true_pars)

path = "/Users/amir/out/02/est_100_sim_50.bson";
res_fun(path, true_pars);

path = "/Users/amir/out/03/03/est_200_sim_25.bson";
est_200_25 = res_fun(path, true_pars);

path = "/Users/amir/out/03/03/est_200_sim_50.bson";
est_200_25 = res_fun(path, true_pars);


hcat(est, fit)







bias = mean(est, dims=1) -true_pars'

MSE = sqrt.(mean((est .- true_pars').^2, dims=1))



est



out_200
est_200 = res_fun(out_200);
est = reduce(hcat, est)
est[1,2]
est[1]


pars10 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "eqsel", "llike"];
est_200 = DataFrame(est_200, pars10);


sort!(est_200, ["llike"]);
/Users/akp/output/MC-LN