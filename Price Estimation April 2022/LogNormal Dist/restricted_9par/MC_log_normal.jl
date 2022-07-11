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
    true_pars =  [2.5, 1.5, -1.5, -.5, -3.5, 2.5, 1.5, 3, -3, 3]
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
    tmp = [.3 -.1; .4 -.2]
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
    @show res_bcv = Optim.optimize(x->bcv2_fun(x,down_data[1:2,inds],price_data[inds]), [0.01,.02,.2])


    # # res_bcv = Optim.optimize(bcv2_fun, [0.01 ,.01,.01])
    @show h = abs.(Optim.minimizer(res_bcv))
    # return h 
    if sum(h .> 1) >0 
        h=[0.04, 0.06, 0.2]
        println("BAD BANDWIDTH")

    end




  
    function trim(x,h, delta)
        if abs(x)>2*h^delta
            return 1 
        elseif abs(x)<h^delta
            return 0 
        else
            return ((4*(x-(h^delta))^3)/(h^(3*delta))) - ((3*(x-h^delta)^4)/(h^(4*delta)))
        end
    end    
    
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
    

        

        ll=zeros(n_firms)
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


            if like == 0
                ll[i] = log(pdf(Normal(),30))
                n_zeros += 1
            else
                ll[i]=log(like/(n_sim*h[1]*h[2]*h[3]))
            end

            sort!(ll)




            # val = like/(n_sim*h[1]*h[2]*h[3])
            # if trim(val, min(mean(h),.5),10)==0
            #     ll+=0
            #     println("ZERO!")
            # else 
            #     ll += trim(val, min(mean(h),.5),10)*log(like/(n_sim*h[1]*h[2]*h[3]))
            #     println("Trimming: ", trim(val, min(mean(h),.5),10)*log(like/(n_sim*h[1]*h[2]*h[3])))
            #     # ll+=like
            # end
    
    
        end
        out = mean(ll[3:end])
        if mod(time(),10)<0.1
            println("parameter: ", round.(b, digits=3), " function value: ", -out, " Number of zeros: ", n_zeros)
        end
        Random.seed!()
        return -out
    end



    # # # Estimated parameters: 

    bbo_search_range = (-10,10)
    bbo_population_size =100
    # bbo_max_time=length(par_ind)^2 * 1
    bbo_max_time=60*(n_sim/50)

    bbo_ndim = length(par_ind)
    bbo_feval = 100000
    
    function fun(x)
        par_point = copy(true_pars)
        par_point[par_ind] = x
        return loglike(par_point)
    end

    cbf = x-> println("parameter: ", round.(best_candidate(x), digits=3), " n_rep: ", n_rep, " fitness: ", best_fitness(x) )
    nopts=1
    opt_mat =zeros(nopts,length(par_ind)+1)

    for i = 1:nopts
        bbsolution1 = bboptimize(fun; SearchRange = bbo_search_range, 
            NumDimensions =bbo_ndim, PopulationSize = bbo_population_size, 
            Method = :adaptive_de_rand_1_bin_radiuslimited, MaxFuncEvals = bbo_feval,
            TraceInterval=30.0, TraceMode=:compact, MaxTime = bbo_max_time,
            CallbackInterval=13,
            CallbackFunction= cbf) 
    
        @show opt2 = Optim.optimize(fun, best_candidate(bbsolution1), time_limit=20)
        @show opt_mat[i,:] = vcat(Optim.minimizer(opt2), Optim.minimum(opt2))'
    end

    return opt_mat
end

# replicate_byseed(2, 100,25) 

# tmpres = pmap(x->replicate_byseed(x, 100, 50, vcat(1:8,10)),1:100 )

# bws = reduce(hcat, tmpres)
# ct_div = 0 
# for i =1:100
#     if bws[1,i]>1
#         ct_div += 1
#     end
# end

# plot(x->loglike(vcat(true_pars[1:8],x,true_pars[10])), -6, 0)
# Parameter estimates 
for j = 9:9
    for n_sim =50:50:100
        for n_firms =  50:50:50
            est_pars = pmap(x->replicate_byseed(x, n_firms, n_sim, 1:4) ,1:24 )
            estimation_result = Dict()
            push!(estimation_result, "beta_hat" => est_pars)
            bson("/Users/akp/github/NPSML-Estimator-Price-Data"*
            "/Price Estimation April 2022/LogNormal Dist/restricted_9par/"*
            "est_$(n_firms)_sim_$(n_sim)_1-4", estimation_result)
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

res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_200_sim_25_1-4")
# res = BSON.load("/Users/akp/out/03/est_200_sim_50.bson")


# res["beta_hat"]
# pars = copy(res["beta_hat"])

res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_50_sim_25_1-4")
pars = reduce(vcat,res["beta_hat"])
scatter(pars[:,1], pars[:,2], markersize = 4)

res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_50_sim_50_1-4")
pars = reduce(vcat,res["beta_hat"])
scatter!(pars[:,1], pars[:,2], markersize = 8)

res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_50_sim_100_1-4")
pars = reduce(vcat,res["beta_hat"])
scatter!(pars[:,1], pars[:,2], markersize = 7)




res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_100_sim_25_1-4")
pars = reduce(vcat,res["beta_hat"])
scatter!(pars[:,1], pars[:,2])

res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_200_sim_25_1-4")
pars = reduce(vcat,res["beta_hat"])
scatter!(pars[:,1], pars[:,2], markersize =3)


res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_200_sim_50_1-4")
pars = reduce(vcat,res["beta_hat"])
scatter!(pars[:,1], pars[:,2], markersize=2)







res = BSON.load("/Users/akp/github/NPSML-Estimator-Price-Data/"*
    "Price Estimation April 2022/LogNormal Dist/restricted_9par/est_50_sim_25_1-4")
pars = reduce(vcat,res["beta_hat"])
scatter(pars[:,1], pars[:,2], markersize = 4)


















loglike(res["beta_hat"][1])
loglike(true_pars)

true_pars'

scatter(up_data[2,:], down_data[2,:], markersize=2)
scatter(up_data[1,:], down_data[2,:])
scatter(up_data[1,:], price_data)
scatter(up_data[1,:], up_data[2,:])
using MLJ
using OutlierDetection
using OutlierDetectionData: ODDS


X= Matrix(up_data[1:2,:]')

X=rand(1000,6)

y = copy(price_data)

y = Matrix(vcat(down_data[1:2,:], price_data')')

# X = rand(1000, 6)
X1, y = ODDS.load("thyroid")
# X

# use 50% of the data for training
train, test = partition(eachindex(y), 0.5, shuffle=true)

# load the detector
KNN = @iload KNNDetector pkg=OutlierDetectionNeighbors

# instantiate a detector with default parameters, returning scores
knn = KNN(k=1)

# bind the detector to data and learn a model with all data
knn_raw = machine(knn, X) |> fit!

transform(knn_raw, rows=test)

knn_probas = machine(ProbabilisticDetector(knn), X) |> fit!

predict(knn_probas, rows=test)


knn_classifier = machine(DeterministicDetector(knn), X) |> fit!


predict(knn_classifier, rows=test)









scatter(down_data[2,:], price_data, markersize =2)
scatter(up_data[1,:], down_data[3,:], markersize = 2)

X



