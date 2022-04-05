
using Optim
using LinearAlgebra
using Random
using Distributions
using BlackBoxOptim
using MLDataUtils
using Plots

include("data_sim_like_2d_2d_diff.jl")
n_firms=500 #divide sample in two parts for two fold
n_rep=25



function check_id(b,n_rep)
    bup = [
            vcat(b[1:2],b[8])';
            vcat(b[3:4], 0.)';
            vcat(0, 0, 0)'
        ]


    bdown = [
            vcat(b[5], b[6],0)';
            vcat(b[7], 0, 0)';
            vcat(0 ,0, b[9] )'
         ]

    up_data, down_data, price_data_cf, tmat, bdiff =
     sim_data_like(-1, bup, bdown, [3., 1., 1.], [2., 1.,1], n_firms, 25+n_rep,1.)

# Matrix of data: x, y, p; each row is an observation
    # data = zeros(n_firms, 3)
    # for i = 1:n_firms
    #     data[i,1] = up_data[1,i]
    #     data[i,2] = down_data[1,i]
    #     data[i,3] = price_data_cf[i]
    # end
    # return bdiff*tmat
    # return tmat[1:2,1:2]
    # return bdiff*tmat
    # return down_data[1:2,:]
    return down_data[1:2, :], price_data_cf
end

tpars = [1, 1.5, .5, 2.5, 2.5, -2, 1, -1, 1.5,.5]
#
# check_id(tpars,2)[2]-check_id(est_par,2)[2]
# tpars-est_par
#
#
#
# check_id(Optim.minimizer(res))
#
#
# check_id(tpars)[1] .- check_id(est_par)[1]
# fun1= x-> (norm(check_id([1, 1.5, .5, 2.5, -2.5, -2, 1, -1.5, 1.5])[1] - check_id(x)[1]) +
  # norm(check_id([1, 1.5, .5, 2.5, -2.5, -2, 1, -1.5, 1.5])[2] - check_id(x)[2]))




# The below section does not fix any parameters.

fun1= x-> (norm(check_id(tpars,0)[1] - check_id(x,0)[1]) +
   norm(check_id(tpars,0)[2] - check_id(x,0)[2]))


# The below section does not fix any parameters.

fun1= x-> (norm(check_id(tpars,0)[1] - check_id(x,0)[1]) +
   norm(check_id(tpars,0)[2] - check_id(x,0)[2]))

tpars= [1, 1.5, .5, 2.5, 2.5, -2, 1, 1,  -.5]
tpars1= [1, 1.5, .5, 2.5, 2.5, -2, 1, 1,  -.5]

fun1(tpars1)

res = Optim.optimize(fun1,tpars +rand(10),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))

est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+0.01*randn(10),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))

est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+0.001*rand(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))


est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+0.0001*rand(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))

println("Pars: ", round.(Optim.minimizer(res), digits=3), " error: ", Optim.minimum(res))




fun1= x-> (norm(check_id(tpars,0)[1] - check_id(vcat(x, 3),0)[1]) +
   norm(check_id(tpars,0)[2] - check_id(vcat(x, 3),0)[2]))

fun1(tpars)
res = Optim.optimize(fun1,tpars[1:9]+randn(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))

est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+1*randn(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))

est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+0.01*rand(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))

est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+0.0001*rand(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))

println("Pars: ", round.(Optim.minimizer(res), digits=3), " error: ", Optim.minimum(res))

z1 = copy(est_par)
z2 = copy(est_par)


(z2 - z1)/.1





# Match Only###############


fun2= x-> norm(check_id(tpars,0)[1] - check_id(x,0)[1])


fun2(tpars)
res = Optim.optimize(fun2,tpars[1:9]+randn(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))


est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+1*randn(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))
#
# bup = [
#         vcat(b[1:2],b[8])';
#         vcat(b[3:4], 0.)';
#         vcat(b[9], 0, 0)'
#     ]
#
#
# bdown = [
#         vcat(b[5], b[6],0)';
#         vcat(b[7], 0, 0)';
#         vcat(0 ,0, b[10] )'
#      ]
est_par[1] + est_par[5]
tpars[1] + tpars[5]


est_par[2] + est_par[6]
tpars[2] + tpars[6]

est_par = Optim.minimizer(res)
res = Optim.optimize(fun1,est_par+0.01*rand(9),NelderMead() ,Optim.Options(show_trace = true, iterations=1_0_000))







(z3 - z1)/1













(z4 - z1)/2


z3 = copy(est_par)

z4 = copy(est_par)



der1 = (est_par -tpars[1:9] )/.1

est_par[1]+est_par[5]






Random.seed!(1234)
a= rand(Normal(0,1), 1)
Random.seed!(1234)
b=rand(Normal(0,400), 1)














fun1([1, 1.5, .5, 2, -2.5, -2, 1, 1.5,1])
fun2 = x-> fun1([1, 1.5, .5, 2, -2.5, -2, 1, 1.5,x])
plot(fun2, -2,2)




points = rand(9,10)

bboptimize(fun1; SearchRange = (-3.0, 3.0), NumDimensions = 9, Method=:separable_nes)


res = Optim.optimize(fun1,[1., 1.5, .5, 2, -2.5, -2, 1, 1.2,1.] ,NelderMead() ,Optim.Options(show_trace = true, iterations=1_000_000))

# res = Optim.optimize(fun1,points[1,:] , Optim.Options(show_trace = true, iterations=5_000))
@show pars = Optim.minimizer(res)
res = Optim.optimize(fun1,pars , Optim.Options(show_trace = true, iterations=5000))
@show pars = Optim.minimizer(res)
res = Optim.optimize(fun,pars , Optim.Options(show_trace = true, iterations=5_000))
@show pars = Optim.minimizer(res)


res = Optim.optimize(fun,pars ,BFGS(), Optim.Options(show_trace = true, iterations=100))
@show pars = Optim.minimizer(res)



pars[1] -pars[5]

tpars[1] - tpars[5]


norm(check_id(vcat(pars,-1.5,1.5,1.))[1] - check_id([1, 1.5, .5, 2, -2.5, -2, 1, -1.5, 1.5,1.])[1])






pa






Optim.minimizer(res) = [0.8540771468210884, 1.3045603252699682, 0.959960973935444, 2.6160421095292095, -2.645922853515435, -2.195439673688491, 1.4599609735964998, -0.8839578903846198, 1.5000000048266924, 0.9999999986123719]
