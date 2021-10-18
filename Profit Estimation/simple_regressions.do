import delimited "/Users/amir/github/NPSML-Estimator-Price-Data/Profit Estimation/data_out.csv", clear 


// To rename data after import

rename v1 x 
rename v2 y1
rename v3 y2
rename v4 piu 
rename v5 pid 
rename v6 phi 
rename v7 eps
rename v8 eta



gen xy1 = x*y1
gen xy2 = x*y2
gen xeta = x*eta
gen y1eps = y1*eps
gen epseta = eps*eta

la var x GAI
la var y1 "log emp"
la var y2 "HHI"
la var piu "log (CEO Compensation)"
la var pid "log (Net Income)-Firm"
la var phi "log(CEO Compensation + Net Income)"
la var eps "recovered eps"
la var eta "recovered eta"
la var xy1 "GAI X SCALE"
la var xy2 "GAI X SCOPE"
la var xeta "GAI X ETA"
la var y1eps "SIZE X EPS"
la var epseta "EPS X ETA"



///////////
/////////
///////

cd /Users/amir/Documents/Stata

reg phi c.x#c.y1 c.x#c.y2
outreg2 using reg1.tex, replace ctitle(Model 1)

reg phi xy1 xy2, robust
outreg2 using reg2.tex, replace ctitle(Model 1)

reg phi eps eta, robust
outreg2 using reg1.tex, append ctitle(Model 2)

reg phi xy1 xy2 x y1 y2, robust 
outreg2 using reg1.tex, append ctitle(Model 3)

reg phi xy1 xy2 epseta, robust 
outreg2 using reg1.tex, append ctitle(Model 4)

reg phi x y1 y2 eps eta, robust 
outreg2 using reg1.tex, append ctitle(Model 5)

reg phi xy1 xy2 epseta xeta y1eps , robust 
outreg2 using reg1.tex, append ctitle(Model 6)

reg phi xy1 xy2 epseta xeta y1eps x y1 y2 , robust 
outreg2 using reg1.tex, append ctitle(Model 7)

gen expphi = exp(phi)
gen expemp = exp(y1)









gen phiperemp = expphi/expemp
gen lphiperemp = log(phiperemp)


cd /Users/amir/Documents/Stata

reg lphiperemp xy1 xy2, robust


outreg2 using reg2.tex, replace ctitle(Model 1)

reg lphiperemp eps eta, robust
outreg2 using reg2.tex, append ctitle(Model 2)

reg lphiperemp xy1 xy2 x y1 y2, robust 
outreg2 using reg2.tex, append ctitle(Model 3)

reg lphiperemp xy1 xy2 epseta, robust 
outreg2 using reg2.tex, append ctitle(Model 4)

reg lphiperemp x y1 y2 eps eta, robust 
outreg2 using reg2.tex, append ctitle(Model 5)

reg lphiperemp xy1 xy2 epseta xeta y1eps , robust 
outreg2 using reg2.tex, append ctitle(Model 6)

reg lphiperemp xy1 xy2 epseta xeta y1eps x y1 y2 , robust 
outreg2 using reg2.tex, append ctitle(Model 7)





cd /Users/amir/Documents/Stata

reg phiperemp xy1 xy2, robust
outreg2 using reg3.tex, replace ctitle(Model 1)

reg phiperemp eps eta, robust
outreg2 using reg3.tex, append ctitle(Model 2)

reg phiperemp xy1 xy2 x y1 y2, robust 
outreg2 using reg3.tex, append ctitle(Model 3)

reg phiperemp xy1 xy2 epseta, robust 
outreg2 using reg3.tex, append ctitle(Model 4)

reg phiperemp x y1 y2 eps eta, robust 
outreg2 using reg3.tex, append ctitle(Model 5)

reg phiperemp xy1 xy2 epseta xeta y1eps , robust 
outreg2 using reg3.tex, append ctitle(Model 6)

reg phiperemp xy1 xy2 epseta xeta y1eps x y1 y2 , robust 
outreg2 using reg3.tex, append ctitle(Model 7)



//
//     mfx option can be used to report marginal effects after mfx command
//     has been applied.
//     sysuse auto, clear
//     logit foreign mpg rep78 head
//     mfx compute
//     outreg2 using myfile, replace
//     outreg2 using myfile, mfx ctitle(mfx) see



// Marginal effects
cd /Users/amir/Documents/Stata

reg phi c.x#c.y1 c.x#c.y2, robust
outreg2 using reg_me.tex, replace ctitle(Model 1)
margins, dydx(x y1 y2) post
estimates store xymargins 
outreg2 using reg_me.tex, append ctitle(M/E 1)

reg phi c.x#c.y1 c.x#c.y2 c.x c.y1 c.y2, robust
outreg2 using reg_me.tex, append ctitle(Model 2)
margins, dydx(x y1 y2) post
estimates store xymargins 
outreg2 using reg_me.tex, append ctitle(M/E 2)

reg phi c.x#c.y1 c.x#c.y2 c.eps#c.eta c.y1#c.eps c.x#c.eta c.x c.y1 c.y2, robust
outreg2 using reg_me.tex, append ctitle(Model 3)
margins, dydx(x y1 y2) post
estimates store xymargins 
outreg2 using reg_me.tex, append ctitle(M/E 3)


outreg2 using reg_me.tex, margins  ctitle(mfx)

reg phi c.eps c.eta, robust
outreg2 using reg1.tex, append ctitle(Model 2)

reg phi xy1 xy2 x y1 y2, robust 
outreg2 using reg1.tex, append ctitle(Model 3)

reg phi xy1 xy2 epseta, robust 
outreg2 using reg1.tex, append ctitle(Model 4)

reg phi x y1 y2 eps eta, robust 
outreg2 using reg1.tex, append ctitle(Model 5)

reg phi xy1 xy2 epseta xeta y1eps , robust 
outreg2 using reg1.tex, append ctitle(Model 6)

reg phi xy1 xy2 epseta xeta y1eps x y1 y2 , robust 
outreg2 using reg1.tex, append ctitle(Model 7)




graph matrix x piu pid y1 y2, msize(0.1)
graph matrix phi xy1 xy2 , msize(0.1)
graph matrix lphiperemp xy1 xy2 , msize(0.1)
graph matrix phi eps eta epseta, msize(0.1)
scatter  lphiperemp xy1, msize(0.5)
scatter  lphiperemp eta, msize(0.5)

corr lphiperemp xy1 xy2 
corr phiperemp xy1 xy2 
corr phi xy1 xy2 
reg phi xy1 xy2 



gen exptdc = exp(piu)
gen salprodratio = exptdc/expphi



reg lphiperemp xy1 xy2 x y1 y2 epseta eps eta,  robust
reg lphiperemp xy1 xy2 x y1, robust
reg lphiperemp xy1 xy2 epseta, robust
reg lphiperemp epseta, robust

reg phiperemp epseta, robust
