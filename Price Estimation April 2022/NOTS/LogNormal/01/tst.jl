using LinearAlgebra
using Random
# using Evolutionary
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
global Σ_up = [0 .1;
    0 .2;
    0 .1]


global Σ_down =  [0 .3;
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


global bup, bdown = par_gen(true_pars)




global up_data, down_data, price_data =
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


@show global h = abs.(Optim.minimizer(res_bcv))



function loglike(b)
    # println("par: ", b)
    # b=true_pars
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

    # solve_draw =  x->sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360, true, up_data[1:2,:],b[10])

    # sim_dat = map(solve_draw, 1:n_sim)
    sim_dat = zeros(n_sim, n_firms, 5)
    for i =1:n_sim
        u,d, pr = sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360+i, true, up_data[1:2,:],b[10])
        sim_dat[i,:,1:2] = u[1:2,:]'
        sim_dat[i,:,3:4] = d[1:2,:]'
        sim_dat[i,:,5] = pr
    end

    
    # # for i =1:n_sim
    # #     u,d, pr = sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360+n_i, true, up_data[1:2,:],b[10])
    # #     sim_dat[i,:,1] = u




    # sim_dat = sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360, true, up_data[1:2,:],b[10])
    ll=0.0
    n_zeros = 0
    for i =1:n_firms
        like =0.
        for j =1:n_sim
            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j,i,3])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j,i,4])/h[2]))
                *pdf(Normal(),((price_data[i] - sim_dat[j,i,5])/h[3]))
                )
        end
        # println("like is: ", like, " log of which is: ", log(like/(n_sim*h[1]*h[2]*h[3])))
        if like == 0
        #     # println("Like is zero!!!")
            ll+= -n_firms*2
            n_zeros += 1
        else
            ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
            # ll+=like
        end


    end
    if mod(time(),10)<11
        # println("I'm worker number $(myid()) on thread $(Threads.threadid()), and I reside on machine $(gethostname()).")

        println(" parameter: ", round.(b, digits=4), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    end
    sleep(1)
    return -ll/n_firms
end


loglike(true_pars)
function rosenbrock2d(x)
    println("pars ", x)
    x=[1.1, .5]
    println("par", x)
    sleep(.02)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
bbo_search_range = (-5,5)
bbo_population_size =10
bbo_max_time=300 
bbo_ndim = 10
bbo_feval = 1000

opt = bbsetup(x->loglike(vcat(x,-3)); SearchRange = bbo_search_range, 
    NumDimensions =bbo_ndim, PopulationSize = 50 , MaxTime = bbo_max_time);
# opt = bbsetup(rosenbrock2d; Method= :adaptive_de_rand_1_bin,SearchRange = bbo_search_range, 
#     NumDimensions =2, PopulationSize = 10 , MaxTime = bbo_max_time);

bbsolution1 = bboptimize(opt)

res_CMAE = CMAEvolutionStrategy.minimize(loglike, [-4.98049897223355, 2.885159097751907, -0.01774248537947806, -1.1616458864032642, 4.999890683011327, 4.241280623662581, 0.24261151499704525, 2.879762019036851, 4.620425497447701, -1.1850257237839097], 1.,
        lower = -10*ones(10),
        upper = 10*ones(10),
        noise_handling = CMAEvolutionStrategy.NoiseHandling(2.1),
        callback = (object, inputs, function_values, ranks) -> nothing,
        parallel_evaluation = false,
        multi_threading = false,
        verbosity = 1,
        seed = rand(UInt),
        maxtime = 100,
        maxiter = nothing,
        maxfevals = 30000,
        ftarget = nothing,
        xtol = nothing,
        ftol = 1e-4)
    











n_sim=10
sim_dat = zeros(n_sim, n_firms, 5)


for i =1:n_sim
    u,d, pr = sim_data_JV_up_obs(bup, bdown , Σ_up, Σ_down, n_firms, 360+i, true, up_data[1:2,:],3)
    sim_dat[i,:,1:2] = u[1:2,:]'
    sim_dat[i,:,3:4] = d[1:2,:]'
    sim_dat[i,:,5] = pr
end

sim_dat[1,2,:]