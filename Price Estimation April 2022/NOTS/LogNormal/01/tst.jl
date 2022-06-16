using LinearAlgebra
using Random
using Evolutionary
using Distributions
using BlackBoxOptim
# using Plots
using Assignment
# using BenchmarkTools
using Optim
using CMAEvolutionStrategy
include("JV_DGP-LogNormal.jl")


n_reps = 2 # Number of replications (fake datasets)
n_firms=100
n_sim=25
n_rep =1 
true_pars = [-2.5, 1.5, -1.5, -.5, 3.5, 2.5, 1.5, -3, 3, -3.]
Σ_up = [0 .1;
0 .2;
0 .1]


Σ_down =  [0 .3;
           0 .4;
           0 .1]

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
    sim_data_JV_LogNormal(bup, bdown, Σ_up, Σ_down, n_firms, 38+n_rep, false, 0, 0, true_pars[10])


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
inds = 1:n_firms;
# # Optimize over choice of h
res_bcv = Optim.optimize(x->bcv2_fun(x,down_data[1:2,inds],price_data[inds]), [0.1,.1,.1]);


@show h = abs.(Optim.minimizer(res_bcv))



function loglike(b)
    bup = [
        vcat(b[1:2], (b[8]))';
        vcat(b[3:4], 0.)';
        vcat(0 , 0, 0)'
    ]

    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., (b[9]) )'
     ]

    solve_draw =  x->sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360, true, up_data[1:2,:],b[10])

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
            ll+= -n_firms +rand()*2
            n_zeros += 1
        else
            ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
            # ll+=like
        end


    end
    # if mod(time(),10)<11
        # println("I'm worker number $(myid()) on thread $(Threads.threadid()), and I reside on machine $(gethostname()).")

        println(" parameter: ", round.(b, digits=20), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    # end
    sleep(1)
    return -ll/n_firms
end

function rosenbrock2d(x)
    println("par", x)
    sleep(2)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
bbo_search_range = (-5,5)
bbo_population_size =10
bbo_max_time=60 
bbo_ndim = 10
bbo_feval = 1000

opt = bbsetup(loglike; Method= :adaptive_de_rand_1_bin,SearchRange = bbo_search_range, 
    NumDimensions =bbo_ndim, PopulationSize = 10 , MaxTime = bbo_max_time);

bbsolution1 = bboptimize(opt)