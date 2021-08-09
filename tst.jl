
using Optim
using LinearAlgebra
using Random
using Distributions
using BlackBoxOptim
using MLDataUtils
include("data_sim_seed.jl")
include("data_sim_like.jl")
n_firms=1000 #divide sample in two parts for two fold
n_rep=1
up_data, down_data, up_profit_data_cf, down_profit_data_cf, price_data_cf, A_mat, β_diff, β_up, β_down, Σ_up, Σ_down =
 sim_data([2.,5.5, 1.], [1.5, 2., 1.], [3., 1., 1.], [2., 1., 3.], n_firms, 25+n_rep)
Random.seed!(25+n_rep)
price_data_cf = price_data_cf + rand(Normal(0.,4.), n_firms)




# Matrix of data: x, y, p; each row is an observation
data = zeros(n_firms, 3)
for i = 1:n_firms
    data[i,1] = up_data[1,i]
    data[i,2] = down_data[1,i]
    data[i,3] = price_data_cf[i]
end

train, test = splitobs(data, at = 0.5,obsdim=1)

function bcv2_fun(h)
    ll = 0.0
    for i = 1:n_firms
        for j=1:n_firms
            if (j!=i)
                # expr_1 = 0.0
                expr_1 = ((up_data[1,i]-up_data[1,j])/h[1])^2 + ((down_data[1,i]-down_data[1,j])/h[2])^2 + ((price_data_cf[i]-price_data_cf[j])/h[3])^2
                expr_2 = pdf(Normal(),(up_data[1,i]-up_data[1,j])/h[1]) * pdf(Normal(),((down_data[1,i]-down_data[1,j])/h[2])) * pdf(Normal(),((price_data_cf[i]-price_data_cf[j])/h[3]))
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


Optim.minimizer(res_bcv)
# m=2
# yy = data[:,2]
# pp = data[:,3]
# S=cov([yy pp])
# H_Silverman = (4/(n_firms*(m+2)))^(2/(m+4)) * S
# h= sqrt.(diag(H_Silverman))


#step1 estimate on train1:
function loglikep(data,b,h)
    n_firms=size(data,1)
    n_sim=500
    solve_draw = x->sim_data_like(data[:,1],[b[1], b[2], 1.], [b[3], b[4], 1.], [3., 1., 1.], [2., 1.,abs(b[5])], n_firms, abs(b[6]), 12342+x)
    sim_dat = map(solve_draw, 1:n_sim)
    ll=0.0
    for i =1:n_firms
        like =0.
        for j =1:n_sim
            like+=pdf(Normal(),((data[i,2] - sim_dat[j][i,2])/h[1]))*pdf(Normal(),((data[i,3] - sim_dat[j][i,3])/h[2]))
        end
        ll+=log(like/(n_sim*h[1]*h[2]))
    end
    # println("parameter: ", b, " function value: ", -ll/n_firms)
    return -ll/n_firms
end

#bandwidth selection

function likelihood_h_o2(h,sv)
    if (h[1]<0.01 || h[2]<0.01)
        return 10.
    else
        likeh = b->loglikep(train, [2.0,5.5, 1.5,b[1],b[2], 4.0],h)
        res = Optim.optimize(likeh, sv)
        thetahat_h= Optim.minimizer(res)
    end
    println("bandwidth is: ", h)
    return loglikep(train,[2.0,5.5, 1.5,thetahat_h[1],thetahat_h[2],4.0],h), thetahat_h
end

function likelihood_h_o3(h)
    if (h[1]<0.01 || h[2]<0.01)
        return 10.
    else
        likeh = b->loglikep(train, [b[1],b[2],b[3],b[4],b[5],b[6]],h)
        res = Optim.optimize(likeh, ones(6),LBFGS())
        thetahat_h= Optim.minimizer(res)
    end

    # return loglikep(test,[2.,5.5,1.5,thetahat_h[1],thetahat_h[2],4.],[0.44,2.65]), thetahat_h
    println("bandwidth is: ", h)
    return loglikep(test,[thetahat_h[1],thetahat_h[2],thetahat_h[3],thetahat_h[4],3.,thetahat_h[5]],[.35,2.00]), thetahat_h
end

#

fun_o2 = x->likelihood_h_o2(x)[1]
fun_o3 = x->likelihood_h_o3(x)[1]

# res =Optim.optimize(fun_1d,.2,.8)



res_o2 =Optim.optimize(fun_o2,[.3,4.3]), Optim.Options(time_limit=500, show_trace = true,extended_trace=true)



res_o3 =Optim.optimize(fun_o3,[1.,1.], Optim.Options(time_limit=500, show_trace = true,extended_trace=true))

#500 sims:
# [0.0749, 4.84] with 4.466682, and 2.05, 5.478
# [0.189, 1.85 ] with 4.48 , and 2.034, 5.36

h_hat_o2=Optim.minimizer(res_o2)
h_hat_o3=Optim.minimizer(res_o3)

@show likelihood_h_o2(h_hat_o2)
#evaluate at approximate chosen bandwidth by option 2

@show likelihood_h_o2([0.3, 4.3])

@show likelihood_h_o2([0.25, 1.16], [1.,1.])
@show likelihood_h_o2([0.275, 4.32])


#Option 2 parameters by choice of bandwidths h= .3 , 4.3
4.474352387705274: [2.0376907419316828, 5.4692103476237675, 1.5631112690894484, -1.8750800301026171, 0.0802443435498647]

#Option 3 parameters by choice of bandwidths [h = .35, 2.12]
(4.555772386614069, [2.077284762286353, 5.387401718580461, 1.7453802982770854, -1.7395067209442077, 3.115854695787264])



.29476697152565157, 1.178303274829831]
@show likelihood_h_o2(h_hat_o3)

@show likelihood_h_o2([0.3, 1.8])

#100 sims
[1.8727969224578453, 5.808577235858017, 1.7392406280155386, 1.6817652299040744, 3.2839088247142265])




(5.5-5.37)^2  + (3-3.91)^2


(5.49-5.5)^2 + (4.23-3.)^2
# 2.16,5.32 are theta bases on h the same
likelihood_h([.303,.8])

likelihood_h([.203,.8])



fun_solve = x->loglikep(train,[2.0,5.5, 1.5,x[1],3.,x[2]],[0.3, 1.8])
# plot(fun_solve, 0.5, 5)
4.4313
4.41
4.43
4.49
4.4056
4.39

res=Optim.optimize(fun_solve,[.003, 0.0001])
Optim.minimizer(res)
# plot(fun_solve, 0.5, 5)



fun_solve!(1.,[1221])
nlsolve(fun_solve!, [4.])

using NLsolve
