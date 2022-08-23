### Densities for Log-normal and Uniform distributions. 
using Distributions
using Plots
using Statistics
using LaTeXStrings

ln1 = LogNormal(0, 1 )
ln2 = LogNormal(0, sqrt(0.5) )
ln3 = LogNormal(0, sqrt(0.1) )

N1 = Normal(0,1)
uf1 = Uniform(0,1)

plot(size=(4*600,4*400))

plot(x->pdf(ln1, x), 0, 4, label= L"LN(\mu=0, \sigma^2 =1)", title= "Density of Log-Normal Distributions",
    color = :black)

plot!(x->pdf(ln2, x), 0, 4, label= L"LN (\mu=0, \sigma^2 =0.5)", color=:blue)
plot!(x->pdf(ln3, x), 0, 4, label= L"LN(\mu=0, \sigma^2 =0.1)", color=:red )

plot!(x->pdf(N1, x), 0, 4, label= "Standard Normal", color=:red, 
            linestyle=:dash)


savefig("LN-Densities-large")


