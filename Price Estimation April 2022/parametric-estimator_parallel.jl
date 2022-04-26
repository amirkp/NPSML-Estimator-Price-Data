
using Distributed
addprocs(23)
# Pkg.add("BenchmarkTools")
using BenchmarkTools
@everywhere begin
    using Optim
    using LinearAlgebra
    using Random
    using Distributions
end

# @everywhere include("data_sim_seed.jl")
# include("data_sim_like.jl")
@everywhere include("data_sim_like_2d_2d_diff.jl")
@everywhere include("data_sim_like_2d_2d_match_only.jl")


@everywhere n_firms=100

# @everywhere function replicate_byseed(n_rep)
@everywhere begin
        bup = [1. 1.5 -1;
               .5 2.5 0;
              0 0  0 ]
        bdown = [2.5 -2 0;
                1  0 0;
                0 0 .5]
        B= bup+bdown

    up_data, down_data, price_data_cf, tmat, bdiff=
        sim_data_like( -1, bup, bdown, [2, 1., 1.], [.5, 3, 1.], n_firms, 20, 2.5)
    mu_price = mean(price_data_cf)
end


tmat
@everywhere tpar = [1, 1.5, .5, 2.5, 2.5, -2, 1, -1, .5]



########################
##################
#### MAIN LIKELIHOOD FUNCTION
#############
######################
###################
function loglikepr(b)
    n_sim=100

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

     solve_draw = x->sim_data_like(up_data[1:2,:],bup, bdown , [2, 1., 1], [.5, 3, 1], n_firms, 1234+x, 2.5)
    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0
    h=[0.1, 0.1, .6]

    for j=1:n_sim
        pconst = mean(sim_dat[j][3])-mu_price
        sim_dat[j][3][:] = sim_dat[j][3] .+ pconst
    end
    n_zeros=0
    begin
        for i =1:n_firms
            like =0.
            for j =1:n_sim
                like+=(
                    pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                    *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                    *pdf(Normal(),((price_data_cf[i] - sim_dat[j][3][i])/h[3]))
                    )
            end
            if like == 0
                # println("Like is zero!!!")
                ll+= -n_firms
                n_zeros += 1
            else
                ll+=log(like/(n_sim*h[1]*h[2]*h[3]))
            end
        end
    end


    # println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    return -ll/n_firms
end

# @benchmark loglikepr(x) setup=(x=tpar+0.01*rand(9))

loglikepr(tpar)

loglikepr([-0.016, 1.602, 0.486, 2.504, 2.099, -1.963, 0.935, 1.746, -1.577])
# foo = [-0.016, 1.602, 0.486, 2.504, 2.099, -1.963, 0.935, 1.746, -1.577]
# using CMAEvolutionStrategy



res_CMAE = CMAEvolutionStrategy.minimize(loglikepr, rand(9), 1.)



Optim.optimize(loglikepr, rand(9))


@benchmark loglikepr(x) setup=(x=tpar+0.01*rand(9))





##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################




altpar = [-0.016, 1.602, 0.486, 2.504, 2.099, -1.963, 0.935, 1.746, -1.577]
altpar=[-3.316, 0.456, 4.643, 1.685, -0.376, -1.52, 1.241, 3.665, -1.824]
# altpar= copy(tpar)
bup1 = [altpar[1] altpar[2] altpar[8];
       altpar[3] altpar[4] 0;
        0 0  0 ]
bdown1 = [altpar[5] altpar[6] 0;
        altpar[7]  0 0;
        0 0 altpar[9]]
B1= bup1+bdown1
up_data1, down_data1, price_data_cf1, tmat1, bdiff1 =
    sim_data_like( -1, bup1, bdown1, [2, 1., 1.], [.5, 3, 1.], n_firms, 20, 2.5)

function loss(altpar)
    bup1 = [altpar[1] altpar[2] altpar[8];
           altpar[3] altpar[4] 0;
            0 0  0 ]
    bdown1 = [altpar[5] altpar[6] 0;
            altpar[7]  0 0;
            0 0 altpar[9]]
    B1= bup1+bdown1
    up_data1, down_data1, price_data_cf1, tmat1, bdiff1 =
        sim_data_like( -1, bup1, bdown1, [2, 1., 1.], [.5, 3, 1.], n_firms, 20, 2.5)

    return norm(down_data-down_data1) +norm(price_data_cf-price_data_cf1)
    #
     # + norm(bdiff- bdiff1)
end



loss(tpar)
res_CMAE = CMAEvolutionStrategy.minimize(loss, -rand(10), 1.)

res = Optim.optimize(loss, vcat(tpar,1.), BFGS(),Optim.Options(g_tol = 1e-12,
                             iterations = 50,
                             store_trace = true,
                             show_trace = true))

res = Optim.optimize(loss, vcat(tpar,1.),Optim.Options(g_tol = 1e-12,
                              iterations = 50,
                              store_trace = true,
                              show_trace = true))
Optim.minimizer(res)
mu_price0 = mean(price_data_cf)
mu_price1 = mean(price_data_cf1)

var(down_data[3,:])
var(down_data1[3,:])
var(price_data_cf)
var(price_data_cf1)
cor(up_data[1,:],down_data[2,:])
cor(up_data1[1,:],down_data1[2,:])

cor(up_data[1,:],down_data[1,:])
cor(up_data1[1,:],down_data1[1,:])

cor(up_data1[2,:],price_data_cf1)
cor(up_data[2,:],price_data_cf)



##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################














loglikepr(tmp)

res_1 = Optim.optimize(loglikepr, tmp)


res_1 = Optim.optimize(loglikepr,tpar )
res_1 = Optim.optimize(loglikepr,rand(9) )



# res_1 = Optim.optimize(loglikepr,randn(10) )

est_pars = Optim.minimizer(res_1)
parameter: [0.9096990084827733, 1.5014328626826445, 0.48271857957354425, 2.4932050323272947, 2.496648174914263, -2.000245205253806, 0.9846901768913681, -1.0723973642542484, 1.57060400252492, 0.5562416984742975] function value: 1.0833060373082983


Value: 1.0833060373082983
Value: 1.0833060428773038
1.0833060424511887
#define function with two variables

fun_2v = x -> loglikepr(vcat(est_pars[1:7], x))
fun_2v(est_pars[8:10])

fun_2v([-1.07, 1.5,1.])


res_2 = Optim.optimize(fun_2v, est_pars[8:10] )
res_2 = Optim.optimize(fun_2v, [-1.07, 1.5,.5] )
res_2 = Optim.optimize(fun_2v, rand(3))

Optim.minimizer(res_2)

# fun_1v = x -> fun_2v(vcat(x, 2))
# res_3 = Optim.optimize(fun_1v, [-1.07, 1.54] )
# res_3 = Optim.optimize(fun_1v, Optim.minimizer(res_3))

fun_tst = x -> loglikepr(vcat(x, 3))
res_4 = Optim.optimize(fun_tst, est_pars[1:9] )
est_pars=Optim.minimizer(res_4)
res_4 = Optim.optimize(fun_tst,[79.165, 3.668, -6.921, 8.779, 119.309, -17.171, 53.594, -57.095, 113.636])
res_4 = Optim.optimize(fun_tst,rand(9))



parameter: [3.228, 1.563, 0.282, 2.682, 5.93, -2.442, 2.532, -2.695, -4.863, -2.0] function value: 1.0563174470528696
parameter: [1.62, 1.519, 0.432, 2.553, 3.546, -2.132, 1.457, -1.557, -2.586, -1.0] function value: 1.05631745963973
parameter: [1.242, 1.516, 0.466, 2.303, 3.45, -2.081, 1.236, -1.241, 2.058, -0.5] function value: 1.2285046417101508
parameter: [0.821, 1.499, 0.491, 2.486, 2.364, -1.983, 0.925, 1.009, 1.443, -0.5] function value: 1.0833060342377498
parameter: [0.035, 1.477, 0.566, 2.423, 1.189, -1.83, 0.395, -0.443, 0.313, 0.0] function value: 1.0833060361311713
parameter: [0.051, 1.477, 0.565, 2.424, 1.212, -1.833, 0.406, -0.455, 0.335, 0.01] function value: 1.0833060399687218
parameter: [0.906, 1.5, 0.498, 2.496, 2.486, -1.995, 0.98, -1.051, -1.574, -0.556] function value: 1.0563174455549786
parameter: [1.067, 1.504, 0.483, 2.509, 2.725, -2.026, 1.088, -1.165, -1.802, -0.656] function value: 1.0563174526285561
parameter: [0.908, 1.501, 0.483, 2.493, 2.495, -2.0, 0.984, -1.072, 1.569, 0.556] function value: 1.0833060578353952
parameter: [1.066, 1.506, 0.468, 2.506, 2.731, -2.031, 1.09, -1.185, 1.796, 0.656] function value: 1.0833060413330948
parameter: [1.224, 1.51, 0.453, 2.518, 2.967, -2.061, 1.196, -1.298, 2.023, 0.756] function value: 1.0833060382103779
parameter: [1.607, 1.521, 0.416, 2.549, 3.541, -2.136, 1.455, -1.574, 2.575, 1.0] function value: 1.0833060341646064
parameter: [7.893, 1.696, -0.184, 3.056, 12.948, -3.358, 5.694, -6.096, 11.62, 5.0] function value: 1.083306040580924
parameter: [8.05, 1.701, -0.199, 3.069, 13.184, -3.389, 5.8, -6.21, 11.847, 5.1] function value: 1.0833060356914954
parameter: [15.75, 1.915, -0.934, 3.69, 24.708, -4.886, 10.993, -11.749, 22.928, 10.0] function value: 1.0833060374699903
parameter: [15.993, 1.921, -0.926, 3.704, 25.025, -4.922, 11.131, -11.879, 23.196, 10.1] function value: 1.0833128630718507
parameter: [79.165, 3.668, -6.921, 8.779, 119.309, -17.171, 53.594, -57.095, 113.636, 50.0] function value: 1.0833095659810787
parameter: [79.096, 3.672, -6.945, 8.783, 119.333, -17.178, 53.613, -57.159, 113.761, 50.1] function value: 1.0833072567393618
parameter: [80.178, 3.713, -7.083, 8.884, 121.141, -17.417, 54.445, -58.103, 115.65, 51.0] function value: 1.083306037600223


x1 = [0.906, 1.5, 0.498, 2.496, 2.486, -1.995, 0.98, -1.051, -1.574, -0.556]
x2 = [1.067, 1.504, 0.483, 2.509, 2.725, -2.026, 1.088, -1.165, -1.802, -0.656]


y1 = [0.908, 1.501, 0.483, 2.493, 2.495, -2.0, 0.984, -1.072, 1.569, 0.556]
y2 = [1.066, 1.506, 0.468, 2.506, 2.731, -2.031, 1.09, -1.185, 1.796, 0.656]
y3 = [1.224, 1.51, 0.453, 2.518, 2.967, -2.061, 1.196, -1.298, 2.023, 0.756]


y4 = [7.893, 1.696, -0.184, 3.056, 12.948, -3.358, 5.694, -6.096, 11.62, 5.0]
y5 = [8.05, 1.701, -0.199, 3.069, 13.184, -3.389, 5.8, -6.21, 11.847, 5.1]


y6 = [15.75, 1.915, -0.934, 3.69, 24.708, -4.886, 10.993, -11.749, 22.928, 10.0]
y7 = [15.993, 1.921, -0.926, 3.704, 25.025, -4.922, 11.131, -11.879, 23.196, 10.1]


y8 = [79.165, 3.668, -6.921, 8.779, 119.309, -17.171, 53.594, -57.095, 113.636, 50.0]
y9 = [79.096, 3.672, -6.945, 8.783, 119.333, -17.178, 53.613, -57.159, 113.761, 50.1]
y10 = [80.178, 3.713, -7.083, 8.884, 121.141, -17.417, 54.445, -58.103, 115.65, 51.0]


println("Gradient: ", round.((y2 - y1)/0.1, digits=3))


println("Gradient: ", round.((y3 - y2)/0.1, digits=3))
println("Gradient: ", round.((y3 - y1)/0.2, digits=3))

println("Gradient: ", round.((z2 - z1)/0.1, digits=3))


println("Gradient: ", round.((y5 - y4)/0.1, digits=3))

println("Gradient: ", round.((y7 - y6)/0.1, digits=3))

println("Gradient: ", round.((y9 - y8)/0.1, digits=3))

println("Gradient: ", round.((y10 - y8)/1, digits=3))


((x2-x1)/-0.1)

Optim.minimizer(res_3)
Optim.minimum(res_3)

res_3 = Optim.optimize(fun_1v, rand(2) )




















#########################################################
############################MATCH ONLY###################
#########################################################
function loglikepr_match(b)
    n_sim=100

    bup = [
        vcat(b[1:2],b[8])';
        vcat(b[3:4], 0.)';
        vcat(b[9], 0, 0)'
    ]


    bdown = [
        vcat(b[5], b[6],0)';
        vcat(b[7], 0, 0)';
        vcat(0 ,0., b[10] )'
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
                )
        end
        ll+=log(like/(n_sim*h[1]*h[2]))
    end
    println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms)
    return -ll/n_firms
end


fun_2v = x -> loglikepr(vcat(est_pars[1:7], x))
fun_2v(est_pars[8:10])

fun_2v([-1.07, 1.5,1.])


res_2 = Optim.optimize(fun_2v, est_pars[8:10] )
res_2 = Optim.optimize(fun_2v, [-1.07, 1.5,.5] )
res_2 = Optim.optimize(fun_2v, rand(3))

Optim.minimizer(res_2)






















println("Pars: ", round.(Optim.minimizer(res_1), digits=3), " error: ", Optim.minimum(res_1))

t2= Optim.minimizer(res_1)
loglikepr(t2)
loglikepr(tpar)
1.512322906068312
parameter: [0.9367709557048315, 1.4996053495338961, 0.4953518306944853, 2.49603019899543, 2.514817957334957, -1.9998389292512435, 0.9841113772121836, 1.582841718095592, 0.801864214906488, -1.0667187511359142] function value: 1.039943389966612
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
