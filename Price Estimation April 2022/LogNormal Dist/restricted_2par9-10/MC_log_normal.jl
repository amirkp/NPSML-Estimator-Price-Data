### Monte-Carlo -- NPSML estimator -- Normally Distributed
### Simulation using finite market approximation. 
# package for parallel computation
using Distributed
using BSON
# using FLoops
addprocs()    # Cores  (This is for #444 Mac Pro)
@everywhere using Optim    # Nelder-Mead Local Optimizer

@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
    using Plots
    using Assignment
    using BenchmarkTools
    # include("JV_DGP-LogNormal.jl")
    include("JV_DGP-mvLogNormal.jl")
    # include("LP_DGP.jl")
end

@everywhere begin 
    n_reps =24 # Number of replications (fake datasets)
    true_pars =  [-2.5, 1.5, -1.5, -.5, 3.5, 2.5, 1.5, 3, -3, 3]
    # true_pars =  [2.5, .5, -1.5, -1.5, -3.5, 2.5, 2.5, 1, -3, 3]

    # true_pars = round.(randn(Random.seed!(1224),10)*3, digits = 2)
end




@everywhere function replicate_byseed(n_rep, n_firms, n_sim, par_ind)
    # n_rep =22
    Σ_up = zeros(3,3)
    tmp = [.3 .1; .4 -.2]
    Σ_up[1:2, 1:2] = tmp*tmp'
    Σ_up[3,3] = .1

    Σ_down = zeros(3,3)
    tmp = [.6 -.1; .4 -.2]
    Σ_down[1:2,1:2] = tmp*tmp'
    Σ_down[3,3] = .1


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


    bup, bdown = par_gen(true_pars)
    up_data, down_data, price_data =
        sim_data_JV_LogNormal(bup, bdown, Σ_up, Σ_down, 1000
            , 38+n_rep, false, 0, 0, true_pars[10])
        
    ind_sample = sample(1:1000, n_firms, replace= false);
    up_data =up_data[:, ind_sample];
    down_data= down_data[:, ind_sample];
    price_data= price_data[ ind_sample];

    # println("hi after fake data")
    # mean of transfers in the data
    # mu_price = mean(price_data)



    # # h: vector of bandwidths
    # # function to be minimized over the choice of h
    # # function uses the fake data above
    function bcv2_fun(h, down_data, price_data)
        h=abs.(h)
        ll = 0.0
        n_firms = length(price_data)
        for i = 1:n_firms
            for j=1:n_firms
                if (j!=i)
                    expr_1 = ((down_data[1,i]-down_data[1,j])/h[1])^2 + ((down_data[2,i]-down_data[2,j])/h[2])^2 + ((price_data[i]-price_data[j])/h[3])^2
                    expr_2 = pdf(Normal(),(down_data[1,i]-down_data[1,j])/h[1]) * pdf(Normal(),((down_data[2,i]-down_data[2,j])/h[2])) * pdf(Normal(),((price_data[i]-price_data[j])/h[3]))
                    ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
                end
            end
        end
        val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1) +
                                ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
        # println("band: ",h," val: ", val)
        return val
    end

    # # only use a sample of size of the nsims not the total observed sample 
    # inds = rand(1:n_firms, n_sim)
    inds = 1:n_firms
    # # Optimize over choice of h
    res_bcv = Optim.optimize(x->bcv2_fun(x,down_data[1:2,inds],price_data[inds]), [0.1,.1,.1])
    # # res_bcv = Optim.optimize(bcv2_fun, [0.1,.1,.1])
    @show h = abs.(Optim.minimizer(res_bcv))
    # h= h*5



    function loglike(b)
        
    
        bup = [
            vcat(b[1:2], (b[8]))';
            vcat(b[3:4], 0.)';
            vcat(1 , 0, 0)'
        ]
    
        bdown = [
            vcat(b[5], b[6],0)';
            vcat(b[7], 0, 0)';
            vcat(0 ,0., (b[9]) )'
         ]
    

    
        solve_draw =  x->sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360+x, true, up_data[1:2,:],b[10])
    
        sim_dat = map(solve_draw, 1:n_sim)
    

        

        ll=0.0
        n_zeros = 0
        for i =1:n_firms
            like =0.
            for j =1:n_sim
                like+=(
                    pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                    *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                    *pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h[3]))
                    )
            end
            # println("like is: ", like, " log of which is: ", log(like/(n_sim*h[1]*h[2]*h[3])))
            if like == 0
            #     # println("Like is zero!!!")
                ll+= -n_firms
                n_zeros += 1
            else
                ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
                # ll+=like
            end
    
    
        end
        # if mod(time(),10)<0.1
            # println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        # end
        Random.seed!()
        return -ll/n_firms
    end



    # # # Estimated parameters: 

    bbo_search_range = (-30,30)
    bbo_population_size =10
    bbo_max_time=length(par_ind)^2 * 50
    bbo_ndim = length(par_ind)
    bbo_feval = 100000
    function fun(x)
        par_point = copy(true_pars)
        par_point[par_ind] = x
        return loglike(par_point)
    end

    cbf = x-> println("parameter: ", round.(best_candidate(x), digits=3), " n_rep: ", n_rep, " fitness: ", best_fitness(x) )
    nopts=1
    opt_mat =zeros(nopts,2)
    for i = 1:nopts
        bbsolution1 = bboptimize(fun; SearchRange = bbo_search_range, 
            NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
            Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
            TraceInterval=30.0, TraceMode=:compact, MaxTime = bbo_max_time,
            CallbackInterval=13,
            CallbackFunction= cbf) 
    
        @show opt2 = Optim.optimize(fun, best_candidate(bbsolution1), time_limit=30)
        @show opt_mat[i,:] = vcat(Optim.minimizer(opt2), Optim.minimum(opt2))'
    end
    return opt_mat
end

# replicate_byseed(2, 100,25) 

# Parameter estimates 
for j = 9:9
    for n_sim =25:25:25
        for n_firms =  50:50:50
            est_pars = pmap(x->replicate_byseed(x, n_firms, n_sim, [j+1]),1:24 )
            estimation_result = Dict()
            push!(estimation_result, "beta_hat" => est_pars)
            bson("/Users/akp/github/NPSML-Estimator-Price-Data"*
            "/Price Estimation April 2022/LogNormal Dist/restricted_2par9-10/"*
            "est_$(n_firms)_sim_$(n_sim)_par_$(j)_1d", estimation_result)
        end
    end
end




res_1p = zeros(2, 10)
for j = 1:10
    res = BSON.load("/Users/akp/github/"*
                "NPSML-Estimator-Price-Data/Price Estimation April 2022/LogNormal Dist"*
                "/restricted_2par9-10/est_100_sim_50_par_9");
    est = reduce(vcat,res["beta_hat"]')
    bias =est .- true_pars[j]
    res_1p[1,j] = mean(bias)
    res_1p[2,j] = sqrt(mean(bias.^2))
end

res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/Price Estimation April 2022/LogNormal Dist/restricted_2par9-10/est_50_sim_25_par_9_1d")


res["beta_hat"]
true_pars'

scatter(up_data[2,:], down_data[2,:], markersize=2)
scatter(up_data[1,:], down_data[2,:])
scatter(up_data[1,:], price_data)





1×3 Matrix{Float64}:
 0.0407936  -0.927353  3.86432

 1×3 Matrix{Float64}:
 0.623851  -0.39205  3.65069

 1×3 Matrix{Float64}:
 0.623851  -0.39205  3.65069


 1×3 Matrix{Float64}:
 0.623851  -0.39205  3.65069

 1×3 Matrix{Float64}:
 0.188597  -0.86655  0.40441
 1×3 Matrix{Float64}:
 -0.0203868  -1.0999  0.951601


 0.164359  11.0318  0.651847
 1×3 Matrix{Float64}:
 -1.79046  9.12177  0.715513

 



scatter(down_data[2,:], price_data, markersize =2)
scatter(up_data[1,:], down_data[3,:], markersize = 2)





From worker 25:   parameter: [-2.5, 1.5, -1.5, -0.5, 3.5, 2.5, 1.5, -3.0, -0.003, -1.547] function value: 3.155296138701713 Number of zeros: 0






interrupt()

########################################
########################################
##### Comparing the results ############
########################################
########################################



########### BBOPTIM results
using DataFrames
function res_fun(opt_vec)
    MC_est= [best_candidate(opt_vec[i])  for i = 1:n_reps ]
    MC_fit= [best_fitness(opt_vec[i]) for i = 1:n_reps ]
    MC_est= reduce(vcat, MC_est')
    MC_est =hcat(MC_est, MC_fit)
    return MC_est
end



MC_out_half_both = BSON.load("LogNormal Dist/MC/03/MC_half_bw_nf_$((1)*100)_sim_25.bson")
MC_out_half_both = MC_out_half_both["beta_hat"]

est_half_bw = res_fun(MC_out_half_both)

DataFrame(est_half_bw, pars)

pars = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d","eqsel", "llike"  ]
est_bw_half_10par = DataFrame(est_half_bw, pars)


mapcols(mean, est_bw_half_10par)


sort!(est_bw_half_10par, ["llike"])
est_bw_half_10par

##############################3
##############################
###############################

MC_out_half_price = BSON.load("LogNormal Dist/MC/03/MC_half_bwprice_nf_$((1)*100)_sim_25.bson")
MC_out_half_price= MC_out_half_price["beta_hat"];

est_half_bwprice = res_fun(MC_out_half_price);

pars = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d","eqsel", "llike"  ]
est_bw_halfprice_10par = DataFrame(est_half_bwprice, pars)


sort!(est_bw_halfprice_10par, ["llike"])


est_bw_half_10par

#################################
#################################
#################################


MC_out_half_eqsel = BSON.load("LogNormal Dist/MC/03/MC_half_eqsel_bw_nf_$((1)*100)_sim_25.bson");
MC_out_half_eqsel= MC_out_half_eqsel["beta_hat"];

est_half_bw_eqsel = res_fun(MC_out_half_eqsel);

pars9 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "llike"  ];
est_bw_half_eqsel = DataFrame(est_half_bw_eqsel, pars9);


sort!(est_bw_half_eqsel, ["llike"]);

#################################
################################
################################

MC_out_half_eqsel_price = BSON.load("LogNormal Dist/MC/03/MC_half_price_eqsel_bw_nf_$((1)*100)_sim_25.bson");
MC_out_half_eqsel_price= MC_out_half_eqsel_price["beta_hat"];

est_half_bw_eqsel_price = res_fun(MC_out_half_eqsel_price);

pars9 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "llike"  ];
est_bw_half_eqsel_price = DataFrame(est_half_bw_eqsel_price, pars9);


sort!(est_bw_half_eqsel_price, ["llike"]);



########################################


#################################
################################
################################


out_100 = BSON.load("LogNormal Dist/MC/01/MC_nf_100_sim_25.bson");
out_100 = out_100["beta_hat"];

est_100 = res_fun(out_100);

pars10 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d","eqsel", "llike"  ];
est_100 = DataFrame(est_100, pars10);


sort!(est_100, ["llike"]);

############################3333
######################################

out_100_9 = BSON.load("LogNormal Dist/MC/02/MC_nf_100_sim_25.bson");
out_100_9 = out_100_9["beta_hat"];

est_100_9 = res_fun(out_100_9);

pars9 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "llike"  ];
est_100_9 = DataFrame(est_100_9, pars9);


sort!(est_100_9, ["llike"]);


############################3333
######################################

out_100_9_abs = BSON.load("LogNormal Dist/MC/04/est_abs_100_sim_25.bson");
out_100_9_abs = out_100_9_abs["beta_hat"];

est_100_9_abs = res_fun(out_100_9_abs);

pars9 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "llike"  ];
est_100_9_abs = DataFrame(est_100_9_abs, pars9);


sort!(est_100_9_abs, ["llike"]);


############################3333
######################################

out_200 = BSON.load("LogNormal Dist/MC/01/MC_nf_200_sim_25.bson");
out_200 = out_200["beta_hat"];

est_200 = res_fun(out_200);

pars10 = ["b11u", "b12u", "b21u", "b22u", "b11d", "b12d", "b21d", "b13u", "b33d", "eqsel", "llike"];
est_200 = DataFrame(est_200, pars10);


sort!(est_200, ["llike"]);


#######################3
##########################
###########################

mapcols(mean, est_100)
mapcols(mean, est_200)

mapcols(mean, est_bw_half_10par)
mapcols(mean, est_bw_halfprice_10par)


mapcols(mean, est_100_9)
mapcols(mean, est_100_9_abs)
mapcols(mean, est_bw_half_eqsel)
mapcols(mean, est_bw_half_eqsel_price)


est_100_9










MC_out_200 = BSON.load("LogNormal Dist/MC/01/MC_nf_$((2)*100)_sim_25.bson")
MC_out_200 = MC_out_200["beta_hat"]
est_200 = res_fun(MC_out_200)


MC_out = BSON.load("LogNormal Dist/MC/01/MC_nf_$((1)*100)_sim_25.bson")
MC_out = MC_out["beta_hat"]
MC_est= [best_candidate(MC_out[i])  for i = 1:n_reps ]
MC_fit= [best_fitness(MC_out[i]) for i = 1:n_reps ]


MC_est= reduce(vcat, MC_est')
MC_est =hcat(MC_est, MC_fit)
MC_est


# 50 simulations
MC_out_50 = BSON.load("LogNormal Dist/MC/01/MC_nf_$((1)*100)_sim_50.bson");
MC_out_50 = MC_out_50["beta_hat"];
MC_est_50= [best_candidate(MC_out_50[i])  for i = 1:n_reps ];
MC_fit_50= [best_fitness(MC_out_50[i]) for i = 1:n_reps ];


MC_est_50= reduce(vcat, MC_est_50');
MC_est_50 =hcat(MC_est_50, MC_fit_50)


MC_out2 = BSON.load("LogNormal Dist/MC/02/MC_nf_$((1)*100)_sim_25.bson")
MC_out2 = MC_out2["beta_hat"]
MC_est2= [best_candidate(MC_out2[i])  for i = 1:n_reps ]
MC_fit2= [best_fitness(MC_out2[i]) for i = 1:n_reps ]

MC_est2= reduce(vcat, MC_est2')
MC_est2 =hcat(MC_est2, MC_fit2)

MC_out2_50 = BSON.load("LogNormal Dist/MC/02/MC_nf_$((1)*100)_sim_50.bson");
MC_out2_50 = MC_out2_50["beta_hat"];
MC_est2_50= [best_candidate(MC_out2_50[i])  for i = 1:n_reps ];
MC_fit2_50= [best_fitness(MC_out2_50[i]) for i = 1:n_reps ];


MC_est2_50= reduce(vcat, MC_est2_50');
MC_est2_50 =hcat(MC_est2_50, MC_fit2_50)

MC_fit - MC_fit2


# MC_est_50[:,8:9] = abs.(MC_est_50[:,8:9])
# MC_est2_50[:,8:9] = abs.(MC_est2_50[:,8:9])

# est_matrix = Array{Float64, 3}(undef, 5, n_reps, 10)

mean(MC_est, dims=1)
mean(MC_est_50, dims=1)
mean(est_200, dims =1)
mean(MC_est2, dims=1)
mean(MC_est2_50, dims=1)


MC_out = BSON.load("LogNormal Dist/MC/01/MC_nf_$((1)*100)_sim_25.bson")
MC_out = MC_out["beta_hat"]
MC_est= [MC_out[i][1] for i = 1:n_reps ]
MC_fit= [MC_out[i][2] for i = 1:n_reps ]

MC_est= reduce(vcat, MC_est')


MC_out2 = BSON.load("LogNormal Dist/MC/02/MC_nf_$((1)*100)_sim_25.bson")
MC_out2 = MC_out2["beta_hat"]
MC_est2= [MC_out2[i][1] for i = 1:n_reps ]
MC_fit2= [MC_out2[i][2] for i = 1:n_reps ]

MC_est2= reduce(vcat, MC_est2')

hcat(MC_est2, MC_fit2 )



sum(MC_fit2.<0)


println(mean(estimates, dims =2))
best_candidate(tst["beta_hat"][1])
for j = 1:2
    for i = 1:n_reps
        tmp_est = BSON.load("LogNormal Dist/MC/01/MC_nf_$((j)*100)_sim_25.bson")
        est_matrix[j,i,:] = vcat(tmp_est["beta_hat"][i][1][:], tmp_est["beta_hat"][i][2])
        est_matrix[j,i,8:9]=abs.(est_matrix[j,i,8:9])
    end
end


mse_vec = zeros(5, 10)
bias_vec = zeros(5,10)
true_pars[8:9] = abs.(true_pars[8:9])
true_pars= vcat(true_pars, 0 )

for j = 1:5 
    for i = 1: n_reps
        mse_vec[j,:] += (est_matrix[j,i,:] - true_pars).^2/n_reps
        bias_vec[j,:] += (est_matrix[j,i,:] - true_pars)/n_reps
        est_matrix[j,i,8:9]=abs.(est_matrix[j,i,8:9])
    end
    
end

sqrt.(mse_vec)
bias_vec



############# Spec 2

est_matrix2 = Array{Float64, 3}(undef, 3, n_reps, 9)


for j = 1:3
    for i = 1: n_reps
        tmp_est, tpar = BSON.load("NormalDist/MC/MC_half_$(j*100)_sim_$(25 +0*(j-1)*25).bson")
        est_matrix2[j,i,:] = tmp_est[2][i][:]
        est_matrix2[j,i,8:9]=abs.(est_matrix[j,i,8:9])
    end
end


mse_vec2 = zeros(5, 9)
bias_vec2 = zeros(5,9)
true_pars[8:9] = abs.(true_pars[8:9])

for j = 1:3
    for i = 1: n_reps
        mse_vec2[j,:] += (est_matrix2[j,i,:] - true_pars).^2/n_reps
        bias_vec2[j,:] += (est_matrix2[j,i,:] - true_pars)/n_reps
    end
    
end

sqrt.(mse_vec2)
bias_vec2















function est_performance(est_matrix, true_pars)
    rmse = zeros(9)
    bias = zeros(9)
    est_matrix[:,8:9] = abs.(est_matrix[:,8:9])

    true_pars[8:9] = abs.(true_pars[8:9])
    for i = 1:n_reps
        rmse += (true_pars - est_matrix[i,:]).^2/n_reps 
        bias += (true_pars - est_matrix[i,:])/n_reps
    end
    rmse = sqrt.(rmse)
    return rmse, bias

end



rmse50, bias50 = est_performance(est_matrix, true_pars)
rmse50_300, bias50_300 = est_performance(est_matrix, true_pars)

scatter(est_matrix[:,1])


mean(est_matrix[:,7])
bias50




#######

est_matrix= BSON.load("NormalDist/MC/02/MC_nf_400_sim_75.bson")

opt_vec = est_matrix["beta_hat"]


[opt_vec[i][:] for i = 1:24]

best_candidate.(opt_vec[:])
best_fitness.(opt_vec[:])













##### Testing bboptimize 







est_matrix= zeros(n_reps, 11)
for n_rep = 1:2
        Σ_up = [0 .1;
                0 .2;
                0 .1]


        Σ_down =  [0 .3;
                   0 .4;
                   0 .1]

        #      [β11u, β12u, β21u, β11u, β11d, β12d, β21u, β13u, β33d]

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


        bup, bdown = par_gen(true_pars)
        up_data, down_data, price_data =
         sim_data_JV_LogNormal(bup, bdown, Σ_up, Σ_down, n_firms, 38+n_rep, false, 0, 0, 3.)

    function bcv2_fun(h, down_data, price_data)
        h=abs.(h)
        ll = 0.0
        n_firms = length(price_data)
        for i = 1:n_firms
            for j=1:n_firms
                if (j!=i)
                    expr_1 = ((down_data[1,i]-down_data[1,j])/h[1])^2 + ((down_data[2,i]-down_data[2,j])/h[2])^2 + ((price_data[i]-price_data[j])/h[3])^2
                    expr_2 = pdf(Normal(),(down_data[1,i]-down_data[1,j])/h[1]) * pdf(Normal(),((down_data[2,i]-down_data[2,j])/h[2])) * pdf(Normal(),((price_data[i]-price_data[j])/h[3]))
                    ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
                end
            end
        end
        val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1) +
                                ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
        # println("band: ",h," val: ", val)
        return val
    end




    # # Optimize over choice of h
    inds = 1:n_firms
    res_bcv = Optim.optimize(x->bcv2_fun(x,down_data[1:2,inds],price_data[inds]), [0.1,.1,.1])
    h = (abs.(Optim.minimizer(res_bcv)))

    function loglike(b)
        # n_sim=50
        
    
        bup = [
            vcat(b[1:2], b[8])';
            vcat(b[3:4], 0.)';
            vcat(0 , 0, 0)'
        ]
    
        bdown = [
            vcat(b[5], b[6],0)';
            vcat(b[7], 0, 0)';
            vcat(0 ,0., b[9] )'
        ]
    
        # bup = [
        #     vcat(b[1:2], 1.)';
        #     vcat(b[3:4], 0.)';
        #     vcat(0 , 0, 0)'
        # ]
    
        # bdown = [
        #     vcat(b[5], b[6],0)';
        #     vcat(b[7], 0, 0)';
        #     vcat(0 ,0., 1. )'
        #  ]
    
        solve_draw =  x->sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360+x, true, up_data[1:2,:],3.)
    
        sim_dat = map(solve_draw, 1:n_sim)
        # sim_dat = solve_draw.(1:n_sim)
        # sim_dat = solve_draw.(1:n_sim)
        # sim_dat = []
        # for i = 1:n_sim
        #     push!(sim_dat, solve_draw(i))
        # end
        ll=0.0
        n_zeros = 0
        for i =1:n_firms
            like =0.
            for j =1:n_sim
                like+=(
                    pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                    *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                    *pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h[3]))
                    )
            end
            # println("like is: ", like, " log of which is: ", log(like/(n_sim*h[1]*h[2]*h[3])))
            if like == 0
            #     # println("Like is zero!!!")
                ll+= -n_firms
                n_zeros += 1
            else
                ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
                # ll+=like
            end
    
    
        end
        if mod(time(),10)<0.1
            println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
        end

        return -ll/n_firms
    end


    bbo_search_range = (-5,5)
    bbo_population_size =50
    bbo_max_time=100
    bbo_ndim = 10
    bbo_feval = 10000

        # opt = bbsetup(loglike; SearchRange = bbo_search_range, 
        #   NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
        #   Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
        #   TraceInterval=10.0, TraceMode=:compact)


    bbsolution1 = bboptimize(loglike; SearchRange = bbo_search_range, 
            NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
            Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
            TraceInterval=10.0, TraceMode=:compact, Workers= [myid()]) 

    est_matrix[n_rep,1:10] = best_candidate(bbsolution1)
    est_matrix[n_rep,11] = best_fitness(bbsolution1)

end
