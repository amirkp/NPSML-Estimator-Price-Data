

using Distributed
addprocs(24)
@everywhere using Optim
@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using BlackBoxOptim
    using Plots
    using Assignment
    using Gurobi
    using CMAEvolutionStrategy
    using JuMP
    using BenchmarkTools
    include("JV_DGP-LogNormal.jl")
    include("LP_DGP.jl")
end
# @benchmark sim_data_JV_LogNormal(bup, bdown, sig_up, sig_down, n_firms, 36, false, 0, 0)


#ceo disu for labor 

@everywhere begin
        n_firms=250

        bup = [-2.5 1.5 -3;
               -1.5 -.5 0;
              0 0  0 ]
        bdown = [3.5 2.5 0;
                1.5  0 0;
                0 0 3.]
        B= bup+bdown

    

        sig_up = [0 .1;
                    0 .2;
                    0 .1]
        sig_down = [0 .3;
                    0 .4;
                    0 .1]
    up_data, down_data, price_data, upr, dpr= sim_data_JV_LogNormal(bup, bdown, sig_up, sig_down, n_firms, 28, false, 0, 0,-3.)
    # up1, down1, price1 =sim_data_LP(bup, bdown, sig_up, sig_down, n_firms,36)
    mu_price = mean(price_data)
    # mu_price1 = mean(price1)
    tpar = [-2.5, 1.5, -.5, -.5, 3.5, 1.5, 1.5, 1, 1]
end



scatter(up_data[2,:], down_data[1,:],markersize =3)
# scatter(up_data[1,:], down_data[3,:])
# scatter(up_data[1,:], price_data)
# scatter(up_data[1,:], up1[1,:])
scatter(up_data[1,:], price_data)
scatter(down_data[2,:], dpl)
# scatter(up1[1,:], down1[1,:])
# scatter(up_data[2,:], up1[2,:])
# scatter(down_data[1,:], down1[1,:])
# scatter(price_data, price1)

# scatter(up_data[1,:], down_data[2,:])
# scatter(up1[1,:], down1[1,:],color=:red)
scatter(up_data[1,:], dpr)
scatter(up_data[1,:], upl)

########################
########################
###### BANDWIDTH ######
########################
########################
########################

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
    println("band: ",h," val: ", val)
    return val
end

# bcv2_fun([-.1, 1.0, 1.])


# Optimize over choice of h
res_bcv = Optim.optimize(bcv2_fun, [0.1,0.1,0.1])
# res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)

h = abs.(Optim.minimizer(res_bcv))



#################################
#################################
######### Silverman #############
#################################

n_sim =50
m=3
S=cov(hcat(down_data[1,:], down_data[2,:], price_data))
H_Silverman = (4/(n_sim*(m+2)))^(2/(m+4)) * S

@show h= sqrt.(diag(H_Silverman))
h = [.2,.2,.2]
# h = h/5
########################
##################
#### MAIN LIKELIHOOD FUNCTION
#############
######################
###################

function loglike(b)
    n_sim=25


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

    solve_draw =  x->sim_data_JV_up_obs(bup, bdown , sig_up, sig_down, n_firms, 360+x, true, up_data[1:2,:],b[10])

    sim_dat = pmap(solve_draw, 1:n_sim)
    ll=0.0
    
    
    # h=0.01*[0.2, 0.4, 1.]


    n_zeros = 0
    for i =1:n_firms
        like =0.
        for j =1:n_sim
            like+=(
                pdf(Normal(),((down_data[1,i] - sim_dat[j][2][1,i])/h[1]))
                *pdf(Normal(),((down_data[2,i] - sim_dat[j][2][2,i])/h[2]))
                *pdf(Normal(),((price_data[i] - sim_dat[j][3][i])/h[3]))
                )
            # like+=(
            #     (down_data[1,i] - sim_dat[j][2][1,i])^2
            #     +(down_data[2,i] - sim_dat[j][2][2,i])^2
            #     +(price_data[i] - sim_dat[j][3][i])^2
            #     )
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
    if mod(time(),10)<.1
        println("parameter: ", round.(b, digits=3), " function value: ", -ll/n_firms, " Number of zeros: ", n_zeros)
    end
    
    return -ll/n_firms
end


loglike(vcat(tpar,-3.))
xs = -3:0.05:3.
scatter(xs, [loglike([-3.27262,  1.31947,   -0.164926, 
     -0.784406,   2.86529, 0.0644565,   
      1.45865,    xs[i],    0.497634,    1.6246]) for i =1:length(xs)])
-1.28

loglike([-2.40851 , 0.647752,  -0.995906 , -0.0825694,  2.54985 , -0.0844993  ,1.09125 ,  -0.329535  , -0.0470503  , 1.05985] )
# tpar
# loglike(tpar[1:4])

# @benchmark loglike(rand(3))
xrange = 0.:0.05:9
# scatter(xrange, [loglike(vcat(tpar[1:9], xrange[i])) for i in 1:length(xrange)])
scatter(xrange, [loglike(vcat(tpar[1:7], xrange[i], tpar[9], 3)) for i in 1:length(xrange)])

scatter(xrange, [loglike(vcat(tpar[1:9], xrange[i])) for i in 1:length(xrange)])


stop
# plot(x->loglike(vcat(-2.5, x, -.5)), -2,2 )

Optim.optimize(x->loglike(vcat(x, -.5, -.5)),-5,2)
llike2p = x-> loglike(vcat(tpar[1:7],x[1],x[2],x[3]))
llike2p([-1,-2.])
# @everywhere n_firms =200

bbo_search_range = (-5,5)
bbo_population_size =50
SMM_session =1
bbo_max_time=22000
bbo_ndim = 10

opt_33 = bbsetup(loglike; SearchRange = bbo_search_range, NumDimensions =bbo_ndim, PopulationSize = 50 , Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = bbo_max_time, Workers=[myid()]);
#large bandwidth 
sol33 = bboptimize(opt_33)
println(round.(best_candidate(sol33), digits=3))
println(best_fitness(sol32))

































# Monday April 18, 2022 Log Normal
# N=500 S= 100 parameter: [2.201, -0.744, 2.144, 2.998, 2.766, -0.111, -1.229, -0.347, 0.688] function value: -0.7557211674202852 Number of zeros: 0
# N=500 S=100 parameter: [-0.119, 0.007, 1.138, 3.0, 0.771, -0.598, -0.094, -0.003, 1.545] function value: 1.1840699948728597 Number of zeros: 0
#     (different bandwidth)
# True parameters: 1.1940
# [-0.334, 2.028, 0.525, 2.19, 1.963, -2.234, 1.163, -0.059, 0.777]
                    altpar = [-1.511, 0.841, 3.25, 1.499, -2.584, -2.178, 1.334, 1.006, 0.77]
altpar=[-3.316, 0.456, 4.643, 1.685, -0.376, -1.52, 1.241, 3.665, -1.824]



# altpar= vcat(best_candidate(bbsolution1)[1:4],tpar[5:end])

# altpar = best_candidate(bbsolution1)
bup1 = [altpar[1] altpar[2] altpar[8];
       altpar[3] altpar[4] 0;
        0 0  0 ]
bdown1 = [altpar[5] altpar[6] 0;
        altpar[7]  0 0;
        0 0 altpar[9]]
B1= bup1+bdown1

# sig_up = [0 .1;
#             0 .2;
#             0 .1]
# sig_down = [0 0.1;
#             0 .2;
#             0 .1]
# up_data1, down_data1, price_data1 = sim_data_JV_LogNormal(bup1, bdown1, sig_up, sig_down, n_firms, 36, false, 0, 0)

n_firms = 1000 
up_data, down_data, price_data= sim_data_JV_LogNormal(bup, bdown, sig_up, sig_down, n_firms, 36, false, 0, 0, 1.)

up_data1, down_data1, price_data1 = sim_data_JV_up_obs(bup1, bdown1, sig_up, sig_down, n_firms, 36, true, up_data[1:2,:],altpar[10])
up_data2, down_data2, price_data2 = sim_data_JV_up_obs(bup, bdown, sig_up, sig_down, n_firms, 1360, true, up_data[1:2,:])

mu_price0 = mean(price_data)
mu_price1 = mean(price_data1)
mu_price2 = mean(price_data2)

scatter(up_data[1,:],up_data1[1,:])

# [cor(up_data[1,:],down_data[k,:]) for k =1:2]
#     -[cor(up_data1[1,:],down_data1[k,:]) for k =1:2]


# [cor(up_data[1,:],down_data[k,:]) for k =1:2]
#     -[cor(up_data2[1,:],down_data2[k,:]) for k =1:2]



scatter(up_data1[1,:],up_data[1,:])
scatter(down_data[2,:],down_data1[2,:])
scatter(price_data, price_data1)

cor(vcat(up_data[1:2,:],price_data')', vcat(down_data[1:2,:], price_data')')
cor(vcat(up_data1[1:2,:],price_data1')', vcat(down_data1[1:2,:], price_data1')')
cor(vcat(up_data2[1:2,:],price_data2')', vcat(down_data2[1:2,:], price_data2')')

var(up_data[2,:])
scatter(up_data[1,:], down_data[1,:])
scatter!(up_data1[1,:], down_data1[1,:], color=:red, markersize =2)

scatter(up_data[2,:], down_data[1,:])
scatter!(up_data1[2,:], down_data1[1,:], color=:red, markersize =2)

scatter(up_data[2,:], price_data)
scatter!(up_data1[2,:], price_data1, color=:red, markersize =2)






scatter(up_data[1,:], up_data1[1,:])
scatter(down_data[1,:], down_data1[1,:])

cor(up_data[1,:],down_data[1,:])
cor(up_data1[1,:],down_data1[1,:])

cor(up_data1[2,:],price_data_cf1)
cor(up_data[2,:],price_data_cf)


cor(up_data1[1,:],price_data_cf1)
cor(up_data[1,:],price_data_cf)

scatter(up_data[1,:], down_data[1,:],
        xlims=(0.5,2), ylims=(0.5,2.5))

scatter!(up_data1[1,:], down_data1[1,:],
        xlims=(0.5,2), ylims=(0.5,2.5), color =:red,
        markersize = 2 )

scatter(up_data[2,:], price_data_cf)
scatter!(up_data1[2,:], price_data_cf1,
        color =:red,
        markersize = 2 )

cor(price_data_cf, price_data_cf1)
scatter(price_data_cf, price_data_cf1)

scatter(down_data[1,:], down_data1[1,:])
scatter(down_data[1,:], down_data1[1,:])

scatter(price_data_cf, price_data_cf1)


pdf.(Normal(),((price_data_cf-price_data_cf1)[:]/0.09))
pdf.(Normal(),((up_data[1,:]-up_data1[1,:])[:]/0.09))
scatter((price_data_cf.-[sim_dat[j][3][100] for j=1:n_sim])[:])
scatter((down_data[1,:].-[sim_dat[j][1][3,i] for j=1:n_sim])[:])






solve_draw= x->sim_data_JV_up_obs(bup1, bdown1, sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:])
solve_draw= x->sim_data_JV(bup, bdown, sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:],down_data[1:2,:])

n_sim=500
sim_dat = pmap(solve_draw, 1:n_sim)


#############################################
#############################################
################ Illustration ###########
#############################################
#############################################
#############################################

solve_draw= x->sim_data_JV_up_obs(bup, bdown, sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:])
solve_draw1= x->sim_data_JV_up_obs(bup1*10, bdown1, sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:])
solve_draw= x->sim_data_JV(bup, bdown, sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:],down_data[1:2,:])
solve_draw1= x->sim_data_JV(bup1, bdown1, sig_up, sig_down, n_firms, 1234+x, true, up_data[1:2,:],down_data[1:2,:])

n_sim=500
sim_dat = pmap(solve_draw, 1:n_sim)
sim_dat1 = pmap(solve_draw1, 1:n_sim)

i=20
p1 = scatter(
        ([down_data[1,i] for j=1:n_sim])-([sim_dat[j][2][1,i] for j=1:n_sim])
            , markersize=3, title="down1");
p2= scatter(
        ([down_data[2,i] for j=1:n_sim])-([sim_dat[j][2][2,i] for j=1:n_sim])
            , markersize=3, title = "down2");
p3= scatter(
        ([price_data_cf[i] for j=1:n_sim])-([sim_dat[j][3][i] for j=1:n_sim])
            , markersize=3, title= "price");


####### PLOT WITH ALTERNATIVE VALUES ######



p4 = scatter(
        ([down_data[1,i] for j=1:n_sim])-([sim_dat1[j][2][1,i] for j=1:n_sim])
            , markersize=3, title="down1");
p5= scatter(
        ([down_data[2,i] for j=1:n_sim])-([sim_dat1[j][2][2,i] for j=1:n_sim])
            , markersize=3, title= "down2");
p6= scatter(
        ([price_data_cf[i] for j=1:n_sim])-([sim_dat1[j][3][i] for j=1:n_sim])
            , markersize=3, title = "price");

plot(p1,p2,p3,p4,p5,p6, legends=false)



function bcv2_fun(h)
    h=abs.(h)
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
@benchmark res_bcv = Optim.optimize(bcv2_fun, [.05,.05,.1],LBFGS(),autodiff = :forward)
Optim.minimizer(res_bcv)


function ucv_fun(h)
    h=abs.(h)
    ll = 0.0
    for i = 1:n_firms
        for j=1:n_firms
            if (j!=i)
                expr_1=0.0
                expr_1 += -0.25*((down_data[1,i]-down_data[1,j])/h[1])^2 -0.25*((down_data[2,i]-down_data[2,j])/h[2])^2 -0.25*((price_data_cf[i]-price_data_cf[j])/h[3])^2
                expr_2 =0.0
                expr_2 += -0.5*((down_data[1,i]-down_data[1,j])/h[1])^2 -0.5*((down_data[2,i]-down_data[2,j])/h[2])^2 -.5*((price_data_cf[i]-price_data_cf[j])/h[3])^2
                ll += exp(expr_1)- (2*2^(3/2))*exp(expr_2)
            end
        end
    end
    val = ((2*sqrt(pi))^3 * n_firms *h[1]*h[2]*h[3])^(-1)+ ((2*sqrt(pi))^3 * n_firms^2 *h[1]*h[2]*h[3])^(-1)*ll
    println("band: ",h," val: ", val)
    return val
end
res_ucv = Optim.optimize(ucv_fun, rand(3))
res_bcv = Optim.optimize(bcv2_fun, [0.1,.1,.1])

# h_ucv = Optim.minimizer(res_ucv)
@show h_bcv = Optim.minimizer(res_bcv)
stop































#
# @everywhere using ForwardDiff
# @everywhere g = x-> ForwardDiff.gradient(loglikepr,x);
# g(tpar)




#Tuesday April 19  N=500, S=100
# 1.0610327144220315, 1.1919006904187117, 0.7743018802789254, 2.4655273806134694, 2.518238980780474, -1.7582964994263002, 0.4907513258302425, -0.5106482635628364, 0.6359353989317891





































