
using Optim
using LinearAlgebra
using Random
using Distributions
using BlackBoxOptim
using LineSearches
using Plots
# @everywhere include("data_sim_seed.jl")
# include("data_sim_like.jl")
include("data_sim_like_2d_2d_diff.jl")
include("data_sim_like_2d_2d_match_only.jl")

n_firms=500

# @everywhere function replicate_byseed(n_rep)
tpar = [1, 1.5, .5, 2.5, 2.5, -2, 1, 1.5, 2]

bup = [1. 1.5 1.5;
       .5 2.5 0;
       0.0 0  0 ]
bdown = [2.5 -2 0;
        1  0 0;
        0 0 2]
B= bup+bdown

up_data, down_data, price_data_cf, tmat =
    sim_data_like( -1, bup, bdown, [2, 1., 1.], [.5, 3, 1.], n_firms, 205, 2.5)
tmat


cor(up_data[1,:], down_data[2,:])
cor(up_data1[1,:], down_data1[2,:])
cor(up_data2[1,:], down_data2[2,:])

cor(up_data[2,:], down_data[2,:])
cor(up_data1[1,:], down_data1[1,:])
cor(up_data2[1,:], down_data2[1,:])

cor(down_data[2,:], price_data_cf)
cor(down_data1[2,:], price_data_cf1)
cor(down_data2[2,:], price_data_cf2)

mean(price_data_cf)
mean(price_data_cf1)
mean(price_data_cf2)

var(price_data_cf)
var(price_data_cf1)
var(price_data_cf2)

scatter(up_data[1,:], down_data[2,:])
scatter!(up_data1[1,:], down_data1[2,:],msize =2.5, color= "darkorange")
scatter!(up_data2[1,:], down_data2[2,:],msize =1. , color= "black")


scatter!(down_data1[2,:], price_data_cf1, msize =2.5, color= "darkorange")
scatter!(down_data2[2,:], price_data_cf2, msize=2, color="black")




scatter(down_data[2,:], price_data_cf)
scatter!(down_data1[2,:], price_data_cf1, msize =2.5, color= "darkorange")
scatter!(down_data2[2,:], price_data_cf2, msize=2, color="black")







scatter(up_data[3,:], down_data[1,:])
scatter(up_data1[2,:], price_data_cf1)
scatter(up_data[2,:], price_data_cf)

scatter(down_data[1,:], price_data_cf)
norm(B)
function bcv2_fun(h)
    ll = 0.0
    for i = 1:n_firms
        for j=1:n_firms
            if (j!=i)
                # expr_1 = 0.0
                # expr_1 = ((up_data[1,i]-up_data[1,j])/h[1])^2 + ((down_data[1,i]-down_data[1,j])/h[2])^2 + ((price_data_cf[i]-price_data_cf[j])/h[3])^2
                # expr_2 = pdf(Normal(),(up_data[1,i]-up_data[1,j])/h[1]) * pdf(Normal(),((down_data[1,i]-down_data[1,j])/h[2])) * pdf(Normal(),((price_data_cf[i]-price_data_cf[j])/h[3]))
                expr_1 = ((down_data[1,i]-down_data[1,j])/h[1])^2 + ((down_data[2,i]-down_data[2,j])/h[2])^2 + ((price_data_cf[i]-price_data_cf[j])/h[3])^2
                expr_2 = pdf(Normal(),(down_data[1,i]-down_data[1,j])/h[1]) * pdf(Normal(),((down_data[2,i]-down_data[2,j])/h[2])) * pdf(Normal(),((price_data_cf[i]-price_data_cf[j])/h[3]))
                ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
            end
        end
    end
    val = ((sqrt(2*pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((4*n_firms*(n_firms-1))*h[1]*h[2]*h[3])^(-1) * ll
    # println("band: ",h," val: ", val)
    return val
end
# res_ucv = Optim.optimize(ucv_fun, rand(3))
res_bcv = Optim.optimize(bcv2_fun, [0.1,.1,.1])

# h_ucv = Optim.minimizer(res_ucv)
@show h_bcv = Optim.minimizer(res_bcv)




function loglikep(b)
    n_sim=500

    A = [
    vcat(b[1:2],b[4])';
    vcat(b[3],1, 0)';
    vcat(b[5], 0, 1)'
    ]
    # bup = [
    #     vcat(b[1:2],0)';
    #     vcat(b[3:4], 0)';
    #     vcat(0, 0, 0)'
    # ]
    #
    #
    # bdown = [
    #     vcat(b[5], b[6],0)';
    #     vcat(b[7],b[8], 0)';
    #     vcat(0, 0, 1)'
    # ]
    # B1 = bup+bdown
    # norm_ = norm(B[1:2,1:2])/norm(B1[1:2,1:2])
    # bup[1:2,1:2] = bup[1:2,1:2]*norm_
    # bdown[1:2,1:2] = bdown[1:2,1:2]*norm_

    # bup = [b[1:3]' ;
    #         b[4:6]'  ;
    #         vcat(0.5,0,0)']
    # bdown = [b[7:9]' ;
    #         b[10:12]'  ;
    #         vcat(0,0,1)' ]
 #
 # bup = [b[1:3]' ;
 #         b[4:6]'  ;
 #         vcat(b[7],0,0)']
 # bdown = [b[10:12]' ;
 #         b[13:15]'  ;
 #         vcat(b[16],0,1)' ]
    solve_draw = x->sim_data_like_mo(up_data[1:2,:],A, [2, 1., 1.5], [.5, 3, 1.], n_firms, 1234+x,2.5)
    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0
    h=[0.2, 0.2, .3]

    for i =1:n_firms
        like =0.
        for j =1:n_sim
            if sim_dat[j]==99999
                return 200.
            end


            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                # *pdf(Normal(),((price_data_cf[i] - sim_dat[j][3][i])/h[3]))
                )
        end
        ll+=log(like/(n_sim*h[1]*h[2]))
    end
    println("parameter: ", b, " function value: ", -ll/n_firms)
    return -ll/n_firms
end



########################
##################
#### MAIN LIKELIHOOD FUNCTION
#############
######################
###################
mu_price = mean(price_data_cf)
function loglikepr(b)
    n_sim=100

    bup = [
        vcat(b[1:2],b[8])';
        vcat(b[3:4], 0.)';
        vcat(0, 0, 0)'
    ]


    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., b[9] )'
     ]

    solve_draw = x->sim_data_like(up_data[1:2,:],bup, bdown , [2, 1., 1], [.5, 3, 1], n_firms, 1234+x, 2.5)
    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0
    h=[0.1, 0.2, 1.5]

    for j=1:n_sim
        pconst = mean(sim_dat[j][3])-mu_price
        sim_dat[j][3][:] = sim_dat[j][3] .+ pconst
    end

    for i =1:n_firms
        like =0.
        for j =1:n_sim
            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                *pdf(Normal(),((price_data_cf[i] - sim_dat[j][3][i])/h[3]))
                )
        end
        ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
    end
    println("parameter: ", b, " function value: ", -ll/n_firms)
    return -ll/n_firms
end

loglikepr(tpar)

res_1 = Optim.optimize(loglikepr, rand(9) )
res_1 = Optim.optimize(loglikepr,tpar)

t2= Optim.minimizer(res_1)
loglikepr(t)
loglikepr(tpar)
1.512322906068312

0.7558032758352348 - 0.7558036136422682
b = Optim.minimizer(res_1)
parameter: [-1.8260370269037118, -0.4876967995080679, 1.4998452139728955, 0.9229265454623277, 1.136420175269226]
function value: 0.21025018069156237

res_1 = Optim.optimize(loglikep, b)
A1 = [
 vcat(b[1:2],b[4])';
 vcat(b[3],1, 0)';
 vcat(b[5], 0, 1)'
 ]


up_data1, down_data1, price_data_cf1, tmat1 =
    sim_data_like_mo( -1, A1, [2, 1., 1.5], [.5, 3, 1.], n_firms, 205, 2.5)

tmat1



tmat




tmat[3,1]*tmat[1,1]*2 +tmat[1,2]*tmat[3,2]*1 +tmat[1,3]*tmat[3,3]* 1.5

B = bup + bdown




opnorm(B)

opnorm(B1)

B1 = bup1 + bdown1



parameter: [0.9628968751365233, 1.508701143966611, 0.5133254719778029, 2.501724453441883, -2.5496371276281873, -2.007927796769958, 0.9891528670397571]
 function value: 1.4057599723340386

res_2 = Optim.optimize(loglikepr, [-1., 1.5, .5, 2.5, -2.5, -2, 1.,1.,-1])


res_2 = Optim.optimize(loglikepr,vcat(rand(7),0.8787112488369927, 0.046396032141094314))
res_2 = Optim.optimize(loglikepr,tmp)

par_2 = Optim.minimizer(res_2)
res_2 = Optim.optimize(loglikepr, par_2)
par_2 = Optim.minimizer(res_2)


bup2 = [
    vcat(par_2[1:2],1)';
    vcat(par_2[3:4], 0)';
    vcat(0, 0, 0)'
]


bdown2 = [
    vcat(par_2[5], par_2[6],0)';
    vcat(par_2[7],1-par_2[4], 0)';
    vcat(1, 0, 1)'
]




up_data2, down_data2, price_data_cf2, tmat2 =
    sim_data_like( -1, bup2, bdown2, [2, 1., 1.5], [.5, 3, 1.], n_firms, 205, 2.5)

tmat1

tmat





tmat2





bup-bdown




bup1-bdown1



bup2-bdown2

# res3= Optim.optimize(loglikep,par2)
#
# res4 =Optim.optimize(loglikepr, par3)
#
# par4 = Optim.minimizer(res4)
#
#
#
#
# res_n1 = Optim.optimize(loglikep, rand(18))
#
# par_n1 = Optim.minimizer(res_n1)

res_n2 = Optim.optimize(loglikepr, par_n1)




tmpp=[1.1100277867209452, 0.23489947478969747, -0.12396110619885363, -0.3177212340345263, 1.8375868889286162, -0.04313430953787366, 0.8543383710490815, -1.1215334100544552, 0.06524686526927598, -2.2087271257948466, 0.07558305688933442, 0.33751948603028115, 0.9125135542253319, -1.6339396234632482, -0.036043371993725155, -0.0073430225088438605, 0.7275450851131774, 0.7591812562602592]
res_n2 = Optim.optimize(loglikepr, tmpp)

res_n3 = Optim.optimize(loglikepr, Optim.minimizer(res_n2))

par_n1 = Optim.minimizer(res_2)


bup2 = [
    vcat(par_n1[1:2],0)';
    vcat(par_n1[3:4], 0)';
    vcat(0.5, 0, 0)'
]


bdown2 = [
    vcat(-1.5-par_n1[1], par_n1[5],1)';
    vcat(par_n1[6],par_n1[7], 0)';
    vcat(0, 0, 1)'
]


#
# bup1 = [par_n1[1:3]' ;
#         par_n1[4:6]'  ;
#         vcat(par_n1[7:8],0)']
# bdown1 = [par_n1[10:12]' ;
#         par_n1[13:15]'  ;
#         vcat(par_n1[16:17],1)' ]
# bup1 = reshape(par_n1[1:9], (3,3))'
#
# bdown1 = reshape(par4[10:18], (3,3))'





up_data1, down_data1, price_data_cf1, tmat1 =
    sim_data_like( -1, bup1, bdown1, [1, 1., 1.5], [.5, 0.5, 1.], n_firms, 205, 2.5)



tmat1





tmat





(bup-bdown)*tmat






(bup1-bdown1)*tmat1




(bup-bdown)-(bup1-bdown1)










loglikepr(par3)



reshape(par[1:9], (3,3))'







: 1.9579330013064662Optim.optimize(loglikep, [0.028595639282063212, 0.24498629296593477, 0.5974044302210253, 1.5156511299137327, 0.1561447478696154, 0.7122195529569815, 0.7199426548458561, 0.9396350380129006])

st = [0.7230336733215906, 0.6547612183096653, 0.2487357905365139, 2.4717003347611337, -2.499171120799732, -1.5574491135524282, 2.2014122514121834, -6.212287467310848, 0.24916271029753406]

Optim.optimize(loglikep, [-1.0740458620806528, 1.3257940863292743, 0.3766799236201755, 2.0603895178019385, -2.4998534100937686, -1.8337623723061482, 1.1820404105539586, 1.5315208044300768])



Optim.optimize(loglikep,  [0.9503652074237323, 1.4843880016693163, 0.9690054609031011, 2.403356882285043, -2.674204996586298, -2.261782648318702, 1.3215761526455718, -0.9684839093165668])


Optim.optimize(loglikep,[3, 1.5, .5, 2,
-2.5, -2, 1, -1.5,
 5.5])
# Optim.optimize(loglikep,ones(10)+rand(10))


bboptimize(loglikep, rand(9); NumDimensions = 9, Method=:separable_nes)




# like_1d_1 = x-> loglikep([x, 5.5, 1.5, 2., 3.,4.], h_ucv)
# @btime like_1d_1(2.)
# # parameter: [2.0, 5.5, 1.5, 2.0, 3.0, 4.0] function value: 4.520071134964339 (Compare later with distributed form)
# plot(like_1d_1, -2., 5.)
# Optim.optimize(like_1d_1, -1., 4)
#
#
# like_1d_2 = x-> loglikep([2., x, 1.5, 2., 3.,4.],h_ucv)
# p1 = plot(like_1d_2, -10., 10.,  legends = false, xlabel="β_12u", ylabel="log-likelihood",title="h_y=0.02, h_p =0.05" )
# res_1d2 = Optim.optimize(like_1d_2, -20., 21.005)
# @show minimizer_1d2 = Optim.minimizer(res_1d2)
# @show min_1d2 = Optim.minimum(res_1d2)
#

like_5d_ucv = x-> loglikep([x[1],x[2],x[3],x[4],x[5],x[6]], h_bcv)
# p_range = [(0.,5.),(3., 8.0),(-3.0, 3.0),(-3.0,3.0),(0.0005,7.),(0.0005,7.)];
# res_global = bboptimize(like_5d_ucv; SearchRange = p_range, TraceMode= :silent, NumDimensions = 5,PopulationSize = 40, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime =400.0)
cand_global =true_pars
# best_candidate(res_global)
if like_5d_ucv(cand_global)<Inf
    res = Optim.optimize(like_5d_ucv, cand_global, BFGS(linesearch=LineSearches.BackTracking()),Optim.Options(show_every = 1, time_limit=2500, show_trace = true,extended_trace=true))
else
    res = Optim.optimize(like_5d_ucv, true_pars+rand(6))
end
println("Replication: ", n_rep, " estimates: ", Optim.minimizer(res))
return Optim.minimizer(res)
end

# n_reps=96
# reps =1:n_reps
reps=80:80
est_pars = pmap(replicate_byseed,reps)

# 195     4.445054e+00     9.244283e-09
# * time: 170.4092879295349
# * step_type: inside contraction
# * centroid: [2.0187411681002927, 5.7763914683949675, 1.7929717166602874, 2.3671704442637713, 4.223155907593449, 3.7166504463718377]

estimation_result = Dict()
push!(estimation_result, "beta_hat" => est_pars)
push!(estimation_result, "beta" => true_pars)


bson("/scratch/ak68/np6par_neldermead/bcv_6par_1500_1000_nm.bson", estimation_result)




Random.seed!(1234)
rand(Normal(0, 4.))

pars1= [0.7953076478659885, 0.82149271566873, 0.009361574194797656, 2.1940877747766976,
        -2.3241075469186385, -1.461997919645698, 1.721928913874204, 2.0533967971812235, 0.929416025826842, -1.6452314584174403]



pars = [0.8037759753995021, 0.5249687865221753, 0.6896115414159297, 3.048717184720271,
        -2.4308400699923487, -1.7819410896738026, 2.3711982914749856, 3.180716938787196, 1.0239774449068106, -1.2399109905587666]
pars[1] - pars[5]

pars[2] - pars[6]

pars[3] - pars[7]
tpar[1]-tpar[5]

tpar[2]-tpar[6]
tpar[3]-tpar[7]
pars1[1]-pars1[5]

pars1[2]-pars1[6]

pars1[3] - pars1[7]
tpar = [1, 1.5, .5, 2,
     -2.5, -2, 1, 1.5,1,-1.5]



parameter: [-0.9250542727930966, 1.119538843061228, 0.5367472195933725, 2.082908347306521, -2.4674090187212814, -1.9556409807850266, 1.3765928960373988, 3.590407877390707, 0.5118564093704369] function value: -0.8851724023995121

pars = [-0.9250542727930966, 1.119538843061228, 0.5367472195933725, 2.082908347306521, -2.4674090187212814, -1.9556409807850266, 1.3765928960373988, 3.590407877390707]
