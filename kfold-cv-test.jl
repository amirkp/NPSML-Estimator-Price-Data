#Testing whether choosing tuning parameters by cross-validation works with NPSML method

# test function is given by a scalar outcome
# y = b1 x1 + b2 x1 eps1 + sin(x)*cos(eps1)
using Distributions
using Plots
using Optim

n_firms=1000
b1 = 1.; b2=2.;
Random.seed!(1234)
x = rand(Normal(0,5),n_firms)
eps1 = rand(Normal(1,2), n_firms)
y=zeros(n_firms)
y .= b1*x + b2 * x.*eps1 + sin.(x).*cos.(eps1)

scatter(y,x, markersize=1)
#goal is to estimate some parameters in here using the NPSML

function like_fun(p)
    h_y = 5.
    n_sim = 100
    p[2]=3.
    #simulate the data
    # x_sim=rand(Normal(0,5), n_sim, n_firms)
    eps1_sim=rand(Normal(1,2), n_sim, n_firms)
    y_sim=zeros(n_sim, n_firms)
    for i =1:n_sim
        y_sim[i,:]= p[1]*x + p[2] * x.*eps1_sim[i,:] + sin.(x).*cos.(eps1_sim[i,:])

    end
    like=0.
    for i =1:n_firms
        like_obs=0.
        for j=1:n_sim
            like_obs+=pdf(Normal(0,1), (y_sim[j,i]-y[i])/h_y)
        end
        like+=log(like_obs*(n_sim*h_y)^(-1))
    end
    return -(1/n_firms)*like
end

like_fun([1.,3])

Optim.optimize(like_fun, [1., 3.], LBFGS(), Optim.Options(show_every = 1, time_limit=1000, show_trace = true, extended_trace=true) )
