# K-fold Cross-Validation #
## The goal is to implement the K-fold cross-validation for choosing the tuning parameters
## of the NPSML estimator—i.e., characteristics and price bandwidth.
## The goal is to divide the sample into a training and validation sample. 
## 
## For a specific choice of bandwidths, BW, we estimate the parameters of the model using the training samples.  
## Next we use the validation sample to evaluate the fit of the model.
## To evaluate the fit, we calculate a measure of average forcast error 

## Say K=10 and sample size is N=500. 
## Partition the sample into 10 equal subsets of size 50 each
## Each set characterizes a training sample of size 450 and a validation set of size 50 
## for a choice of bandwidth h, we obtain 10 estimates from each training sample of size 450

## For the validation step: 
## use each vector of estimates corresponding to a training and validation sample, 
## to generate n_sim number of market outcomes for random realization of the unobserved variables. 
## Then we use the difference between the predicted market outcomes with the actual outcome in the validation set
## The absolute error term for each characteristic or price is scale-normalized by dividing it by the standard deviation of 
# the characteristic or price. 



############## Cross validation for the normal case
using LinearAlgebra
using Random
using Distributions
using BlackBoxOptim
using Plots
using Optim    # Nelder-Mead Local Optimizer
using CMAEvolutionStrategy # Global Optimizer
    



include("data_sim_like.jl")
include("data_sim_seed.jl")






######################################
######################################
######################################
#### GENERATING THE FAKE DATASET #####
######################################
######################################
######################################


## FAKE DATA SET IS DEFINED ON ALL WORKERS
# IN CASE NEEDED IN LATER VERSIONS OF THE ESTIMATOR
n_firms=500;
par_length = 10; 
bup = [
    -1.     1.5    -1;
    0     2.5     0;
    0      0      0
]

bdown = [
        2.5      -2      0;
        1         0      0;
        0         0     -1.5
];



sigup =     [
    2. 0 0;
    0 1. 0;
    0 0 0.3
]


sigdown = [
        2.5 0 0.;
        0 3. 0 ;
        0 0  0.4
]


up_data, down_data, price_data, up_profit, down_profit =
    sim_data(bup, bdown, sigup, sigdown, n_firms, 23, 1.0
);


###########################################################
###########################################################
####### BANDWIDTH/TUNING PARAMETERS SELECTION #############
###########################################################
###########################################################

# This part of the code deals with bandwidth selection
# At the moment it uses bcv2 criterion function
# Reference: Sain, S. R., Baggerly, K. A., & Scott, D. W. (1994). Cross-validation of multivariate densities. Journal of the American Statistical Association,
##################################
##################################


# h: vector of bandwidths
# function to be minimized over the choice of h
# function uses the fake data above
function bcv2_fun(h)
    h=abs.(h)
    ll = 0.0
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

# bcv2_fun([-.1, 1.0, 1.])


# Optimize over choice of h
res_bcv = Optim.optimize(bcv2_fun, rand(3))
# res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)

h = abs.(Optim.minimizer(res_bcv))


### alternative candidate for bandwidth 
### we just choose it as twice res_bcv
h_alt = 2*h
# #################################
# #################################
# ######### Silverman #############
# #################################

# n_sim =50
# m=3
# S=cov(hcat(down_data[1,:], down_data[2,:], price_data))
# H_Silverman = (4/(n_sim*(m+2)))^(2/(m+4)) * S
# @show h= sqrt.(diag(H_Silverman))




# generating hold out samples 

# Partition the sample into 10 parts
# Each time we hold-out one of the parts for validation

k_fold = 10

hold_id = sample(1:n_firms, n_firms, replace = false)


# Each row represents the id of the firms that are held out 
hold_id = reshape(hold_id, (k_fold,Int(n_firms/k_fold)) )




################################################
################################################
################################################
################################################
#### MAIN LIKELIHOOD FUNCTION ##################
################################################
################################################
################################################

# The likelihood function, used for estimation
# Added feature: leave the 
function loglike(b,hold_vec,h)
    n_sim=25

    bup = [
        vcat(b[1:2],b[8])';
        vcat(b[3:4], 0.)';
        vcat(0 , 0, 0)'
    ];


    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., b[9] )'
    ]

    solve_draw =  x->sim_data_like(up_data[1:2,:], bup, bdown , sigup, sigdown, n_firms, 1234+x, b[10])

    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0
    n_zeros = 0
    for i =1:n_firms
        if !in(hold_vec).(i) # only use observations in the training sample 
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
                # println("Like is zero!!!")
                ll+= -n_firms
                n_zeros += 1
            else
                ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
            end
        end
    end
    println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    return -ll/(n_firms-length(hold_vec))
end



# Vector of true parameters

tpar = [-1, 1.5, .0, 2.5, 2.5, -2, 1, -1, -1.5, 1 ]

# Log-likelihood value at the truth
loglike(tpar,hold_id[3,:],h)














(loglike(tpar,hold_id[1,:],h), + loglike(tpar,hold_id[2,:],h))/2
loglike(tpar,hold_id[2,:])

loglike_nohold = x->loglike(x, hold_id[2,:],h)
loglike_nohold = x->loglike(x, [],h)

loglike_nohold = x->loglike(x, hold_id[2,:],h)
res_CMAE = CMAEvolutionStrategy.minimize(loglike_nohold,rand(10), 1.,
        lower = nothing,
         upper = nothing,
         noise_handling = 1.,
         callback = (object, inputs, function_values, ranks) -> nothing,
         parallel_evaluation = false,
         multi_threading = false,
         verbosity = 1,
         seed = rand(UInt),
         maxtime = 10000,
         maxiter = nothing,
         maxfevals = nothing,
         ftarget = nothing,
         xtol = nothing,
         ftol = 1e-3)



# # # Estimated parameters: 
est_pars = xbest(res_CMAE)





## Function should take in parameters and return average error
## Observed sample is just defined in the global scope of this file
## For each vector of estimated parameters that corresponds to a trainingsample:
## simulate the model say 100 times. calculate the average absolute error for each outcome in the relevant holdout sample

##  
# ests: estimated parameters, matrix of size  K times number of parameters
# hold_id: matrix of holdout indices, should correspond to estimated parameters in est


function avg_err(ests, hold_id, n_sim)
    n_samples = size(hold_id)[1]; #number of training/validation samples
    toterr = 0; #variable storing the total forecast error over all validation samples
    for sample_id =1:n_samples #iterate through the validation samples
        b = ests[sample_id, :]; #use the estimates obtained from "sample_id" training sample 
        
        bup = [
            vcat(b[1:2],b[8])';
            vcat(b[3:4], 0.)';
            vcat(0 , 0, 0)'
        ]; 

        bdown = [
            vcat(b[5], b[6],0)';
            vcat(b[7], 0, 0)';
            vcat(0 ,0., b[9] )'
        ]

        # simulate n_sim market outcomes under the parameter estimates from training sample "sample_id"
        solve_draw =  x->sim_data_like(up_data[1:2,:], bup, bdown , sigup, sigdown, n_firms, 20212223+x, b[10]);
        sim_dat = map(solve_draw, 1:n_sim);


        # Standard deviations of the characteristics in the full sample (data)
        sd_y1 = std(down_data[1,:]);
        sd_y2 = std(down_data[2,:]);
        sd_p = std(price_data);

        ll=0.0
        # n_zeros = 0;
       
        for i =1:n_firms
            if in(hold_id[sample_id,:]).(i) 
                err =0.;
                for j =1:n_sim
                    err+=(
                        abs((down_data[1,i] - sim_dat[j][2][1,i])/sd_y1) 
                        +abs((down_data[2,i] - sim_dat[j][2][2,i])/sd_y2)
                        +abs((price_data[i] - sim_dat[j][3][i])/sd_p)
                        ); #absolute errors are normalized by the scale each variable 
                end            
                toterr += err;
            end

        end
    end
    return toterr/(n_sim*n_firms)

end

# Choose zero for no noise
# noise=0 means using true parameters as instead of estimated parameters
noise_par = 2;

est_pars = [
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
    (tpar+noise_par*rand(10))';
]


avg_err(est_pars, hold_id, 50)



# minimize avg_err by searching for the optimal bandwidth
# Very computationally expensive. 
# instead consider a finite grid of reasonable bandwidths
# compare performance 











