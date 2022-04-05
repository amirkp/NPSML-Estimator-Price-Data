
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

pars[1] -pars[5]

tpars[1] - tpars[5]



pars[2] -pars[6]

tpars[2] - tpars[6]

function check_id(b)
    bup = [b[1] b[2] b[8] ;
            b[3]  b[4]  0;
            0 0  0 ]
    bdown = [b[5]  b[6]  0;
             b[7]   0   0
             0  0  b[9]]
    bsum = bup + bdown

    up_data, down_data, price_data_cf, tmat, bdiff =
     sim_data_like(-1, bup, bdown, [3., 1., 1], [2., 1., 1.], n_firms, 25+n_rep,1.)

# Matrix of data: x, y, p; each row is an observation
    data = zeros(n_firms, 3)
    for i = 1:n_firms
        data[i,1] = up_data[1,i]
        data[i,2] = down_data[1,i]
        data[i,3] = price_data_cf[i]
    end
    return bdiff*tmat, bdiff*tmat +(bdiff*tmat)', tmat, bdiff, bsum, tmat*inv(transpose(bsum))
    # return tmat[1:2,1:2]
    # return bdiff*tmat
    # return down_data[1:2,:]
    return down_data[1:2, :], price_data_cf
end
tpars= [1, 1.5, .5, 2.5, 2.5, -2, 1, -1,  .5]
pars1 = [5.994, 1.888, 0.718, 3.503, 8.105, -4.041, 3.938, -4.431, 5.856, 3.]

pars2 = [7.992, 2.044, 0.805, 3.904, 10.346, -4.857, 5.113, -5.803, 7.598,4.]

(pars1-tpars)/2.5

(pars2-pars1)

tpars= [1, 1.5, .5, 2.5, 2.5, -2, 1, -1,  -.5]
check_id(tpars)





check_id(tpars)


mat1 = check_id(tpars)[5]




mat1*check_id(tpars)[4]




pars1=copy(vcat(est_par,.5))
check_id(pars1)


mat2 = check_id(pars1)[5]


mat2 * check_id(pars1)[4]


check_id(pars2)


mat3 = check_id(pars2)[1]


vec1 = rand(3)
vec2 = rand(3)

vec1'*mat1*vec1

pr1 = vec1'*mat2*vec1

pr2 = vec1'*mat3*vec1





pr1 - pr2

vec2'*mat1*vec2

pr3 = vec2'*mat2*vec2

pr4 = vec2'*mat3*vec2

pr3 - pr4


(mat1+mat1')/2





(mat2+mat2')/2





(mat3+mat3')/2




println(check_id(tpars)[1])
tpars= [1, 1.5, .5, 2.5, -2.5, -2, 1, -1.5, 1.5]
# fun1= x-> (norm(check_id([1, 1.5, .5, 2.5, -2.5, -2, 1, -1.5, 1.5])[1] - check_id(x)[1]) +
  norm(check_id([1, 1.5, .5, 2.5, -2.5, -2, 1, -1.5, 1.5])[2] - check_id(x)[2]))

fun1= x-> (norm(check_id(tpars)[1] - check_id(x)[1]) +
   norm(check_id(tpars)[2] - check_id(x)[2]))

fun1(tpars)
res = Optim.optimize(fun1,[2., 3.5, .5, 2.5, -2.5, -2, 1, 1.2,1.5] ,NelderMead() ,Optim.Options(show_trace = true, iterations=1_000_000))


println("Pars: ", round.(Optim.minimizer(res), digits=3), " error: ", Optim.minimum(res))
























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
