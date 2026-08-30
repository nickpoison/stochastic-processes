
### Table of Contents
  

  * [Chapter 1](#chapter-1) &nbsp;&nbsp; Preliminaries
  * [Chapter 2](#chapter-2) &nbsp;&nbsp; Markov Chains
  * [Chapter 3](#chapter-3) &nbsp;&nbsp; Markov Chains: Stationarity & MCMC
  * [Chapter 4](#chapter-4) &nbsp;&nbsp; Pure Jump Processes
  * [Chapter 5](#chapter-5) &nbsp;&nbsp; Second Order Processes & Markov Switching Models

<br/>

--- 

> This is a list of the R code in the text. New scripts and data sets were included in the package `astsa` to cover many of the examples. So rather than repeating it... install the package (once) and then load it each time you try something in the text. 

```r
if (!requireNamespace("astsa")) install.packages("astsa")  # install it if not there
library(astsa)   # load it as needed for examples
```
---

<br/>


### Chapter 1

<br/> Figure 1.2

```r
library(astsa)  # in case you scrolled past the note above 

##-- Gambler’s Ruin --##
set.seed(111)
u = sample(c(-1,1), 20, replace=TRUE)
x = ts(c(10, 10+cumsum(u)), start=0)
tsplot(x, type='o', xlab='n', gg=TRUE, pch=19, ylab=bquote(X[~n]), col=4,
main="Gambler's Ruin")

##-- Counting/Poisson Process --##
set.seed(15432)
k = 5; lmbd = 2; N = 0:k
t = cumsum(rexp(k, rate=lmbd))
tsplot(c(0,t), N, type='s', gg=TRUE, ylab=bquote(N[~t]), pch=19, col=4, main="Counting Process")
points(c(0,t), N, col=4, pch=19)
segments(t[k], N[k+1], 8.5, N[k+1], col=4, lty=1)

##-- Unemployment Rate --##
tsplot(UnempRate2, col=4, ylab='Unemployment Rate', gg=TRUE, main="Time Series")

##-- 2d Brownian Motion --##
set.seed(1)
N = 1000
X = cumsum(c(0, rnorm(N-1)))
Y = cumsum(c(0, rnorm(N-1)))
tsplot(X, Y, xlab='X', gg=TRUE, col=4, main="Brownian Motion")
points(c(0, X[N]), c(0, Y[N]), col=3:2, pch=19)
legend('topleft', legend=c('start', 'end'), lty=0, col=3:2, pch=19, bty='n')
```

<br/> RNG

```r
N = 30
x = c(1) # the bad seed (they are all bad)
for (n in 2:N){ x[n] = (12*x[n-1] + 4) %% 2^32 }
x
```

<br/> Figure 1.5 - CLT

```r
set.seed(90210)
par(mfrow=c(1, 3))  
for (n in c(5, 20, 100)){                    # sample sizes
 x = replicate(500, mean(rbinom(n, 10, .2))) # sample means
 hist(x, col='lightblue', breaks=20, prob=TRUE, main=paste('n =', n), xlab=bquote(bar(X)[~n]))
 curve(dnorm(x, 2, sqrt(1.6/n)), 0,4,501, add=TRUE, col=2, lwd=2)
 abline(v=2, col=2, lwd=2)
 }
```

<br/>  Simulation for Problem 1.5 (c)

```r
# probability that one exponential is less than the other
# in the call: ntrials is optional, but means for X and Y are not
PrXltY <- function(ntrials=2*10^5, meanX, meanY){
  ntrials = ntrials                # use the value specified
  X = rexp(ntrials, rate=1/meanX)  # generate X exponentials
  Y = rexp(ntrials, rate=1/meanY)  # generate Y exponentials
  c(Estimate = mean(X < Y), True = meanY/(meanX+meanY)) # display results
} # end
# now run it
PrXltY(meanX=1500, meanY=1000)
```


[<sub>top</sub>](#table-of-contents)

<br/>

### Chapter 2

<br/> 2 State - Joint Probabilities & MLE 

```r
library(markovchain)
set.seed(90210)
# set up the transition matrix
(M = new('markovchain', states=c('0','1'), transitionMatrix = matrix(c(.3,.7,.4,.6), 2, byrow=TRUE)))
x = rmarkovchain(n=100, object=M) # sample from the chain
tsplot(x) # a plot of the generated sequence (not shown)
createSequenceMatrix(x) # count the transitions
createSequenceMatrix(x, toRowProbs=TRUE) # estimated transition probs
```


<br/> Simple Random Walk

```r
set.seed(8675309)
toss = sample(-1:1, 100, replace = TRUE, prob = c(.45,.1,.45))
X = ts(c(0, cumsum(toss)) , start=0)
tsplot(X, ylab=bquote(X[~n]), xlab='n', col=4, gg=TRUE)
```

<br/> Barnsley Fern

```r
plotBFern <- function(n, col="dark green", main=NULL) {
A1 = matrix(c(0,0,0,0.16,0.85,-0.04,0.04,0.85,0.2,0.23,-0.26,
           0.22,-0.15,0.26,0.28,0.24), ncol=4, nrow=4, byrow=TRUE)
A2 = matrix(c(0,0,0,1.6,0,1.6,0,0.44), ncol=2, nrow=4, byrow=TRUE)
P = c(.01,.85,.07,.07)
M1 = vector("list", 4)
M2 = vector("list", 4)
for (i in 1:4) {
M1[[i]] = matrix(c(A1[i, 1:4]), nrow=2)
M2[[i]] = matrix(c(A2[i, 1:2]), nrow=2)
} 
x <- y <-  numeric(n) 
for (i in 1:(n-1)) {
k <- sample(1:4, prob=P, size=1)
M <- as.matrix(M1[[k]])
z <- M%*%c(x[i],y[i]) + M2[[k]]
x[i+1] <- z[1]
y[i+1] <- z[2]
}
plot(x, y, main=main, axes=FALSE, xlab="", ylab="", col=col, cex=0.1)
} 
## Execute it
plotBFern(100000)
```

<br/> Pico & Sepulveda - Rain?

```r
library(astsa)  # you need this
states = 0:1
( P = matrix(c(.9,.6,.1,.4), 2, dimnames=list(states,states) ))
# (i)
round(P %^% 7, 4)
# (ii)
pie = c(.9,.1)
pie %*% (P %^% 7)
```

<br/> Classification of States

```r
states = 0:5
P = matrix(0, nrow=6, ncol=6, dimnames=list(states, states))
P[1,1] = 1
P[2,1:3] = c(1/2, 1/4, 1/4)
P[3,] = c(0, 1/3, 1/3, 1/6, 0, 1/6)
P[4,4:6] = c(1/4, 1/4, 1/2)
P[5,c(4,6)]= c(2/3, 1/3)
P[6,4:6] = c(1/3, 0, 2/3)
round(P, 2)
library(astsa)
P50 = P %^% 50
round(P50, 2)
```

<br/> PageRank

```r
library(igraph) # load the package
# build the matrix of links (edges)
P = diag(0, 9) # 9 x 9 matrix of zeros
# put a 1 for each link - note page 4 doesn’t have links
P[1, c(3:5,7,9)] = 1
P[2, c(4,5,7)]   = 1
P[3, c(5,7)]     = 1
P[5, 7]          = 1
P[6, c(1,4,5,7)] = 1
P[7, c(1,3:6,8)] = 1
P[8, c(4,7)]     = 1
P[9, 4:7]        = 1
# next, make the graph and then plot it
( Pgraph <- graph_from_adjacency_matrix(P, weighted=TRUE) )
plot(Pgraph, asp=0, vertex.size=2.5*degree(Pgraph))  
# finally, get the associated eigenvector (PageRanks)
round(as.matrix(page_rank(Pgraph)$vector), 2)
```

<br/> Stationary Distribution & Eigenvalues / vectors

```r
( P = matrix(c(.8,.3,.2,.7), 2) ) # transition matrix
( u = eigen(t(P)) ) # eigenvalues and vectors of P transpose
( pie = u$vectors[,1]/sum(u$vectors[,1]) ) # apply the constraint
# check 
pie %*% P # is it pie?
```


[<sub>top</sub>](#table-of-contents)

<br/>

### Chapter 3

<br/> Ehrenfest Chain

```r
library(astsa)
d = 30
N = 500
x = rep(1, N) # a vector of 1s; we need x[1]=1 to start
for (i in 2:N) { # simulate
x[i] = ifelse ( (runif(1) < x[i-1]/d), x[i-1]-1, x[i-1]+1 )
} 
# graphic
tsplot(cbind(Body_1=x, Body_2=d-x), col=astsa.col(2*2:3,.7), spag=TRUE, main=paste("Ehrenfest Chain (d =", d, "particles)"), xlab="n", ylab=expression(X[~n]), gg=TRUE, addLegend=TRUE)
abline(h=d/2, col=8, lty=2) # equilibrium
```

<br/> Gibbs - Bivariate Normal

```r
##-- MCMC generation of bivariate normals --##
MCMC <- function(r, nmcmc, x0, burnin){ # r is the correlation
n <- nmcmc + burnin
X <- Y <- rep(0, n)
X[1] <- x0
sr <- sqrt(1-r^2)
for (i in 1:n){
Y[i] <- rnorm(1, r*X[i], sr)
if (i < n) X[i+1] <- rnorm(1, r*Y[i], sr)
}
return(cbind(X, Y)) # includes burnin
} 
##-- run the simulation --##
r = .8; nmcmc = 1000
x0 = 6; burnin = 10
dog = MCMC(r, nmcmc, x0, burnin)

##-- plot the results --##
lred = adjustcolor(2,.4)
plot(dog, pch=19, col=c(rep(1, burnin), rep(lred, nmcmc)))
grid()
lines(dog[1:(burnin+10),])

##-- plot bv normal contours --##
bivariate.normal <- function(x, mu, Sigma) {
  exp(-.5*t(x-mu)%*%solve(Sigma)%*%(x-mu))/sqrt(2*pi*det(Sigma)) }
mu <- c(0,0); Sigma <- matrix(c(1,r,r,1), nrow=2)
x <- y <- seq(-4, 4, len=100)
# evaluate the density for each value of (x, y)
z <- outer(x, y, FUN=function(x, y, ...){ apply(cbind(x,y), 1, bivariate.normal,
...) }, mu=mu, Sigma=Sigma)
# add the contours
contour(x,y,z, drawlabels=FALSE, col=gray(.5), nlevels=20, add=TRUE)  
```

<br/> Gibbs Sampling - Prevalence of Breast Cancer

> To emphasize: This is point-in-time prevalence among asymptomatic women presenting for screening; i.e., the probability that a woman being screened right now has undetected breast cancer at that moment, and not the often-quoted lifetime risk of about 13% (roughly 1 in 8), which is a cumulative probability over a woman’s entire life and is a very different quantity. 

```r
set.seed(91210)
burnin = 1000
nrun = 5000
m = nrun + burnin
alpha = .04       # T=1|D=0
beta = .10        # T=0|D=1
delta = .9996     # D=0|T=0
gamma = .08       # D=1|T=1
D = rep(0, m)     # the Ds (0 or 1)
T = rep(0, m)     # the Ts (0 or 1)
for (n in 2:m){   # initial value is D[1]=0
if (D[n-1]==1) T[n-1] = rbinom(1, 1, 1-beta)
else T[n-1] = rbinom(1, 1, alpha)
if (T[n-1]==1) D[n] = rbinom(1, 1, gamma)
else D[n] = rbinom(1, 1, 1-delta)
}
runprop = cumsum(D)/1:m # running proportion with disease
layout(matrix(1:2, nrow=2), height=c(4,3))
tsplot((-burnin+1):nrun, runprop, lwd=2, col=4, ylim=c(0,.008), xlab="Step",
ylab="Proportion with Disease")
abline(v=0, col=2, lty=4)
# distribution of prevalence
hist(runprop[burnin+1:nrun], breaks="FD", main=NA, col=astsa.col(4,.3), prob=TRUE, xlab='Pr(D=1)')
abline(v = quantile(runprop[burnin+1:nrun], probs = c(0.05, 0.5, 0.95)), col=2, lwd=2)
abline(v=mean(D[(burnin+1):m]), col=4, lwd=2)
```

<br/> Metropolis - Beta


```r
a = 3
b = 10
burnin = 20
nmcmc = 500 + burnin
betax = function(x, a, b) { x^(a-1)*(1-x)^(b-1) }
X = rep(NA, nmcmc)
X[1] = runif(1) # intial value of the chain
for (n in 2:nmcmc){
Y = runif(1) # Y is proposal from the Q-chain
P = betax(Y,a,b) / betax(X[n-1],a,b)
X[n] = X[n-1] + (Y - X[n-1])*(runif(1) < P)
} 
X = ts(X, start=1-burnin)
# Graphics
par(mfrow=c(2,1))
tsplot(X, ylab=expression(X[n]), xlab='n', col=4, lwd=2, gg=TRUE)
lines(window(X, end=0), col=2, lwd=2)
abline(v=0, lty=2)
hist(X[-(1:burnin)], prob=TRUE, main=NA, col='lightblue', xlab=bquote(X[n]))
curve(dbeta(x,3,10), 0, 1, add=TRUE)
```

<br/> Do Kids Like Candy?

```r
n = 5; y = 5 # data
theta0 = .5 # initial value
burnin = 200
nmcmc = 5000 + burnin
prior = function(theta){ dunif(theta) }
likelihood = function(theta){
 if(theta>= 0 & theta <=1) dbinom(y, n, theta)
 else return(0) 
}
theta = c(theta0, rep(NA, nmcmc-1)) # start the collection of thetas
for (i in 2:nmcmc) {
 theta[i] = theta[i-1] # here theta is the current value
 posterior.old = prior(theta[i])*likelihood(theta[i])
 theta.propose = runif(1) # uniform Q
# theta.propose = theta[i] + rnorm(1, sd=.1) # normal random walk
 posterior.new = prior(theta.propose)*likelihood(theta.propose)
 alpha = min(1, posterior.new/posterior.old)
 if (runif(1) < alpha) theta[i] = theta.propose 
}
# graphics
theta = ts(theta, start=1-burnin)
layout(matrix(c(1, 2), nrow = 2), height=c(4,3))
tsplot(theta, main=NA, col=4, xlab='Index', ylab=expression(theta), gg=TRUE)
lines(window(theta, end=0), col=2)
abline(v=0, lty=2)
hist(theta[-(1:burnin)], freq=FALSE, col= astsa.col(5,.2), border=4, main=NA, xlab=expression(theta))
curve(6*x^5, add=TRUE, col=2, lwd=1.5) # actual distribution
```

<br/>  Exponential Waiting Time

```r
# prior
prior = function(lam, a, b) { lam^(a-1) * exp(-b*lam)}
# likelihood - X is sum of data
likelihood = function(X, n, lam){
 if ( lam < 0 ) return(0)
 lam^n * exp(-lam*X)
}
# choose hyperparameters
a0 = 1; b0 = 2
# get some data
set.seed(8675309)
n = 25
( X = sum(rexp(n, rate=.6)) ) # rate is lambda
# start metropolis
burnin = 100
nmcmc = 1000
niter = nmcmc + burnin
lam = rep(1, nmcmc) # start collection, initial value is 1
# Metropolis (normal RW)
for (i in 2:niter) {
lam[i] = lam[i-1] # at this point lambda is the current value
posterior.old = prior(lam[i], a0, b0)*likelihood(X, n, lam[i])
lam.propose = lam[i] + rnorm(1, sd=.2) # normal random walk
posterior.new = prior(lam.propose,a0,b0)*likelihood(X, n, lam.propose)
alpha = min(1, posterior.new/posterior.old) # acceptance prob
if ( runif(1) < alpha ) lam[i] = lam.propose
}
# graph
library(astsa)
par(mfrow=c(2,1))
lam = ts(lam, start=1-burnin)
tsplot(lam, col=4, ylab=expression(lambda), xlab='Index')
lines(window(lam, end=0), col=2)
bayes = mean(lam[-(1:burnin)])
abline(h=c(n/X, bayes), col=2:3)   
hist(lam[-(1:burnin)], prob=TRUE, col='lightblue', xlab='lambdas', main=NA)
abline(v=quantile(lam, probs=c(.05,.50,.95)))
curve(dgamma(x,a0+n,b0+X),0,1, add=TRUE, col=2) # true posterior
```

[<sub>top</sub>](#table-of-contents)

<br/>

### Chapter 4

<br/> Graph of a Pure Jump Markov Chain (not in the text)

```r
par(tcl=-.2, las=1)
N = c(rep(0,7), rep(2,5), rep(1,14), rep(3,4), rep(5,9), rep(3,6))
tsplot(N, type='n', ylab=bquote(X[~t]), axes=FALSE,  col=4, gg=TRUE) 
axis(2, at=0:5, col='white', col.ticks=1)
lines(N, type='s', col=4)
 jmp = N[jmpt <- c(8,13,27,31,40)] 
 bjmp = c(0, lag(N,-1))
 points(jmpt, jmp, col=4, pch=19, lwd=1.5)
 points(jmpt, bjmp[jmpt], col=4, pch=1)
axis(1, line=0, col='white', col.ticks =2, lwd.ticks = 3, cex.axis=1.2, at=jmpt, labels=c(expression(tau[1]), expression(tau[2]), expression(tau[3]), expression(tau[4]), expression(tau[5])))
segments(0,0, 1,0, col=4)
mtext(0, side=1, at=0, cex=.85)
```

<br/> Example of a PP

```r
# generate PP example
libray(astsa)
set.seed(666)
k = 10        # stop after k arrivals
lmbda = 1/2   # rate
t = cumsum(rexp(k, rate=lmbda)) # arrival times
tsplot(c(0,t,t[k]+1), c(0:k,k), type='s', col=4, las=1, xlim=c(0,t[k]), ylab=bquote(X[~t]), main='Poisson Process')
points(c(0,t), 0:k, pch=19, col=4)


##-- or an ugly but simple version (instead of tsplot)
plot(stepfun(t, 0:k), pch=19, ylab='X', xlab='t', main='Poisson Process')
```

<br/> Golden Gate Bridge Suicides

```r
library(astsa)             # the data file is in astsa
( ave = mean(GGBsuicide) ) # average number of days between suicides
# the figure - panel.first might not work on a Mac
plot(ecdf(GGBsuicide), col=4, panel.first={Grid()})
curve(pexp(x, rate = 1/ave), add=TRUE, col=2, lwd=2)
legend('topleft', c('CDF', 'EDF'), col=2*1:2, lty=1, pch=c(NA,20), bg='white')

# CVM test
if (!requireNamespace("gofedf")) install.packages("gofedf")  # install it if you don't have it
gofedf::testExponential(GGBsuicide)      # do it like this and you don't need to load it       

# Correlation tests
if (!requireNamespace("DescTools")) install.packages("DescTools")  
DescTools::VonNeumannTest(GGBsuicide)
DescTools::BartelsRankTest(GGBsuicide)
```

[<sub>top</sub>](#table-of-contents)

<br/>

### Chapter 5

<br/> Example 5.1

```r
library(astsa)
set.seed(90210)
par(mfrow=1:2, las=1)
t = 1:100
(z = rnorm(4))
om = pi/5
x1 = z[1]*cos(om*t) + z[2]*sin(om*t)
x2 = z[3]*cos(om*t) + z[4]*sin(om*t)
tsplot(spline(x1, n=500), ylab=bquote(X[~t]), ylim=c(-2,2), gg=TRUE)
tsplot(spline(x2, n=500), ylab=bquote(X[~t]), ylim=c(-2,2), gg=TRUE)
```

<br/>  Example 5.5

```r
# ?GGBsuicide #  in astsa, but here's the count per quarter:
y = c(6,3,3,4,2,3,4,2,2,3,4,5,6,5,4,3,6,6,4,3,5,5,5,4,7,5,3,6,7,6,7,9,13,6,11,5,2,6,6,7,4,3,9,4)
n = length(y)
( lambda_hat = mean(y) )
var(y)   # should be close to the mean
##-- 95% CIs --##
# Normal approx
se = sqrt(lambda_hat/n)
( ci_norm = lambda_hat + c(-1,1)*qnorm(.975)*se )
[1] 4.402989 5.733375
# Exact
s = sum(y)
( ci_xact = c(qchisq(0.025,2*s)/(2*n), qchisq(0.975,2*(s+1))/(2*n)) )
```

<br/> Marginal Normals that are not Bivariate Normal 

```r
library(astsa)
x = rnorm(1000) 
z = rnorm(1000)
y = ifelse(x*z > 0, z, -z)
scatter.hist(x, y, hist.col=5, pt.col=6)

# and for fun
par(mfrow=1:2)
QQnorm(x)
QQnorm(y)
```

<br/>  Brownian Motion vs Dow Jones

```r
library(astsa)
set.seed(8675309)
num = nrow(djia)
W = ts(cumsum(c(0, rnorm(num-1))), start=0, frequency=num)
par(mfrow=2:1)
tsplot(W, ylab=bquote(W[~t]), col=4, gg=TRUE)
tsplot(timex(djia), djia[,'Close'], ylab='DJIA Close', col=4, gg=TRUE)
```

<br/>  Gaussian Process Regression

```r
library(MASS)    # comes with R, no need to install
library(astsa)

#-- data --#
x = mcycle$accel
t = mcycle$times
n = length(t) 

#-- hyperparameters --#
alpha    = .2
sigma2_f = 2000  # signal variance (mcycle accel range is large)
sigma2_n = 400   # noise variance (a ridge parameter)

#-- estimate signal --#
K     = sigma2_f * exp(-alpha^2 * outer(t, t, "-")^2 / 2)
Krdg  = K + sigma2_n * diag(n)
f.hat = K %*% solve(Krdg, x)

#-- plot it --#
y = cbind(signal=drop(f.hat), observations=x)
tsplot(t, y, spaghetti=TRUE, type='o', pch=c(NA,20), col=2*1:2, gg=TRUE, xlab="Time (ms)", ylab="Acceleration (g)", addLegend=TRUE, lwd=2:1)
```

<br/> Ornstein-Uhlenbeck Process

```r
sim_ou = function(alpha=-1, t_grid=seq(0,10,by=0.01), dt=0.001, T_burn=20){
 stopifnot(alpha < 0)
 # Time grid: from -(T_burn) up to max(t_grid)
 t_start = min(t_grid) - T_burn
 t_end   = max(t_grid)
 u       = seq(t_start, t_end, by = dt)
 n       = length(u)
 # Simulate increments dW_u ~ N(0, dt)
 dW = rnorm(n, mean = 0, sd = sqrt(dt))
 # Model as SDE dX = alpha*X dt + dW.
 rho       = exp(alpha * dt) # regression coefficient
 X_full    = numeric(n)
 X_full[1] = dW[1]
 for (i in 2:n) {
  X_full[i] = rho*X_full[i - 1] + dW[i]
 }
 # Output X at the desired observation times
 idx = vapply(t_grid, function(t) which.min(abs(u - t)), integer(1))
 list(t = t_grid, X = X_full[idx], X_full = X_full, u = u)
} 

# Run it (one time)
set.seed(90210)
alpha = -2
res = sim_ou(alpha=alpha, t_grid=seq(0,10,by=0.01), dt=0.001, T_burn=20)
tsplot(res$t, res$X, col=4, gg=TRUE, xlab="t", ylab=bquote(X[t] == integral(e^{.(alpha)*(t-u)}*dW[u], -infinity, t)), margins=c(0,1.25,0,0)+.25, main="Ornstein-Uhlenbeck Process")
```


<br/>  Ornstein-Uhlenbeck  vs Wiener 

```r
num = 10
alpha = -2
X = diag(0, nrow=1001, ncol=num)
for (i in 1:num){ 
 u = sim_ou(alpha=alpha, t_grid=seq(0,10,by=0.01), dt=0.001, T_burn=20)
 X[,i] = u$X
}
par(mfrow=2:1)
culers = astsa.col(4, wheel=TRUE, num=num)
tsplot(u$t, X, ylab=bquote(X[~t]), col=culers, spaghetti=TRUE, gg=TRUE, main='Ornstein-Uhlenbeck')
W = ts(diag(0, nrow=10000, ncol=num), start=0, frequency=1000)
for (i in 1:num){
 W[,i] = cumsum(c(0, rnorm(10000-1)))
}
tsplot(W, ylab=bquote(W[~t]), col=culers, spaghetti=TRUE, gg=TRUE, main='Wiener Process')
```

<br/> Example 5.17 - White Noise but _not_ Independent 

```r
library(astsa)
layout(matrix(c(1,2,1,3,1,4),2,3), heights=c(.9,1))
r = diff(log(djia[,'Close']))
tsplot(timex(djia)[-1], r, main="DJIA Returns", ylab=bquote(R[~t]), col=4, gg=TRUE)
acf1(r, main=bquote(R[~t]), col=4, gg=TRUE, ylim=c(-.2,.6))
acf1(r^2, main=bquote(R[~t]^2), col=4, gg=TRUE, ylim=c(-.2,.6)) 
acf1(abs(r), main=bquote(abs(' '*R[~t]*' ')), col=4, gg=TRUE, ylim=c(-.2,.6))
```

<br/> It's _not_ Stationary

```r
tsplot(UKgas, col=4, gg=TRUE)
```

<br/> Figure 5.10 - WN and MA

```r
set.seed(10109)
par(mfrow=c(2,2), cex=.9)
Z = rnorm(200)
X = filter(Z, sides=2, filter=rep(1/3,3))
tsplot(Z, main="white noise", col=4, gg=TRUE)
tsplot(X, ylim=c(-3,3), main="moving average", col=6, gg=TRUE)

freq = seq(0, .5, length.out = 1000)
# WN
tsplot(freq, rep(1,1000), gg=TRUE, ylim=c(.7,1.1), col=4, lwd=2, xlab=bquote(omega), ylab=bquote(italic(f)[Z](omega)))
title('White Noise', cex.main=1)
# MA 
spec = ( 3 + 4*cos(2*pi*freq) + 2*cos(4*pi*freq))/9
tsplot(freq, spec, col=6, lwd=2, gg=TRUE, xlab=bquote(omega), ylab=bquote(italic(f)[X](omega)))
title('Moving Average', cex.main=1)
```

<br/> Frequency Responses 

```r
library(astsa)
par(mfrow=c(3,1))
tsplot(ENSO, main='SOI', col=4, ylab=NA )
tsplot(diff(ENSO), col=4, ylab=NA, main='First Difference')
k = kernel("modified.daniell", 6) # the seasonal MA
tsplot(kernapply(ENSO, k), col=4, ylab=NA, main='Seasonal Moving Average')

##-- frequency responses --##
w = seq(0, .5, by=.001)
FRdiff = abs(1-exp(2i*pi*w))^2
par(mfrow=2:1, las=1, mar=c(3,3,2,1)+2)
tsplot(12*w, FRdiff, col=4, ylab=bquote(abs(A(omega))^~2), xlab='frequency (\u00D7 12)', main='First Difference', margins=.5)
u = rowSums(cos(outer(w, 2*pi*1:5)))
FRma = ((1 + cos(12*pi*w) + 2*u)/12)^2
tsplot(12*w, 10*FRma, col=4, ylab=bquote(abs(A(omega))^~2), xlab='frequency (\u00D7 12)', main='Seasonal Moving Average', margins=.5)
```

<br/> ARMA Spectra

```r
par(mfrow=c(2,2), byrow=TRUE)
tsplot(sarima.sim(ma=-.9, n=200), col=4, gg=TRUE, main='Moving Average', ylab=bquote(X[~t]))
tsplot(sarima.sim(ar = c(1,-.9), n=200), col=6, gg=TRUE, main='Autoregression', ylab=bquote(X[~t]))
arma.spec(ma=-.9, main=NA, gg=TRUE, col=4, lwd=2)
arma.spec(ar = c(1,-.9), main=NA, gg=TRUE, col=6, lwd=2)
```

<br/> Periodogram .... bad

```r
library(astsa)
P = mvspec(rnorm(2^10), col=8, main=NA, ylab='periodogram', gg=TRUE)
segments(0,1, .5,1, col=astsa.col(6,.7), lwd=5) # actual spectrum
lines(P$freq, filter(P$spec, filter=rep(1/101,101), circular=TRUE), col=4, lwd=3)
```

<br/> Sunspots 

```r
library(astsa)
sunspots = sqrt(sunspotz)
tsplot(sunspots, col=4, gg=TRUE, ylab=bquote(X[~t]))

par(mfrow=1:2)
sunspots = sqrt(sunspotz)
spec.ic(sunspots, col=5, lwd=1.5, gg=TRUE, xlim=c(0,.5))
abline(v=1/10.5, lty=2, col=6)
mvspec(sunspots, spans=c(3,3,3), col=5, gg=TRUE, taper=.1, xlim=c(0,.5))
abline(v=c(1/11, 1/89), lty=2, col=6)
krnl = kernel('modified.daniell', c(3,3,3))
par(fig = c(.8, 1, .65, 1), new=TRUE, bty="l")
plot(krnl, type="l", xaxt="n", yaxt="n", ann=FALSE)

sunspots = sqrt(sunspotz)
acf2(sunspots)  # ACF/PACF 
ar(sunspots)    # AIC search with Yule-Walker estimates
sarima(sunspots, p=16) # if you want to see the MLEs
```

<br/> HMM - The Kind of Data 

```r
library(astsa)
layout(matrix(1:6,3,2), heights = rep(c(1,.8,.7), 2))
tsplot(EQcount, type='o', col=4, pch=20, gg=TRUE)
acf1(EQcount, col=4, gg=TRUE, main=NA)
hist(EQcount, 20, main=NA, ylim=c(0,.1), col='lightblue', prob=TRUE, xlab=NA)
x = seq(1, 45)
lines(x, dpois(x, mean(EQcount)), col=6, lwd=2)
tsplot(timex(sp500w), sp500w, col=4, gg=TRUE)
acf1(sp500w^2, col=4, gg=TRUE, main=NA)
hist(sp500w,30, main=NA, col='lightblue', prob=TRUE, ylim=c(0,20), xlab=NA)
x = seq(-.15,.15, by=.001)
lines(x, dnorm(x, mean(sp500w), sd(sp500w)), col=6, lwd=2)
mtext("Counts", side=3, line=-1, adj=.25, cex=.9, outer=TRUE, font=2)
mtext("Returns", side=3, line=-1, adj=.78, cex=.9, outer=TRUE, font=2)
```

<br/> Example 5.30 -  EM ... What Could Go Wrong?

```r
# Generate 2-State Normal Mixture Data
set.seed(123)
n_samples = 250
true_mu = c(-2.5, 3.0)
true_sigma = c(1.0, 1.2)
true_P = matrix(c(0.85, 0.15, 0.25, 0.75), nrow=2, byrow=TRUE)
states = numeric(n_samples)
obs = numeric(n_samples)
states[1] = sample(1:2, 1, prob = c(0.5, 0.5))
obs[1] = rnorm(1, mean=true_mu[states[1]], sd=true_sigma[states[1]])
for (t in 2:n_samples) {
 states[t] = sample(1:2, 1, prob = true_P[states[t-1], ])
 obs[t] = rnorm(1, mean=true_mu[states[t]], sd=true_sigma[states[t]])
} 
# plot series
tsplot(obs, ylab=bquote(X[~t]), lwd=1.5, col=8)
points(obs, bg = 6 * states - 2, pch = 21, cex = 1.5)

# good estimation
fitgood <- HmmFit(obs, m=2, family="norm")
round(fitgood$se, 3) # partial output shown
fitgood$loglik

# bad estimation
xhat = mean(obs)
shat = sd(obs)
mu0 = rep(xhat, 2)
sigma0 = rep(shat, 2)
Gamma0 = matrix(.5, 2, 2)
fitbad <- HmmFit(obs, m=2, family="norm", n_perturb=0, start = list(mu0=mu0, sigma0=sigma0, Gamma0=Gamma0))
round(fitbad$se, 3) # partial output shown (and it is bad)
fitbad$loglik
[1] -602.5356

# likelihood surface
mu1_vals <- mu2_vals <- seq(-5, 5, length.out = 50)
ll_grid <- matrix(0, nrow = 50, ncol = 50)
log_lik <- function(m1, m2, x, sigma, P) {
 alpha = c(0.5, 0.5) * dnorm(x[1], mean = c(m1, m2), sd = sigma)
 sc = sum(alpha); alpha = alpha/sc; l_lik = log(sc)
 for (t in 2:length(x)) {
  alpha = (alpha %*% P) * dnorm(x[t], mean = c(m1, m2), sd = sigma)
  sc = sum(alpha); alpha = alpha/sc; l_lik = l_lik + log(sc)
 }
return(l_lik)
}
for (i in 1:50) {
for (j in 1:50) {
ll_grid[i, j] = log_lik(mu1_vals[i], mu2_vals[j], obs, fitgood$sigma, fitgood$Pmatrix)
}
}
par(mar=c(3,3,2,1), cex.main=1, mgp=c(1.6,.5,0), las=1)
culers = rev(astsa.col(4, wheel=TRUE, num=17, v=.7))
contour(mu1_vals, mu2_vals, ll_grid, nlevels=30, labcex=.75, xlab=bquote(mu[~1]), ylab=bquote(mu[~2]), col=culers, main="Log-Likelihood Contour")
# positions of interest
points(true_mu[1], true_mu[2], col=3, pch=19, cex=1.5)
text(true_mu[1], true_mu[2], "True Parameter", adj=c(-0.1,-0.5), col=3, font=2)
points(fitgood$mu[1], fitgood$mu[2], col=4, pch=4, lwd=3, cex=1.5)
text(fitgood$mu[1], fitgood$mu[2], "EM Success Peak", adj=c(-0.1,1.5), col=4, font=2)
points(fitbad$mu[1], fitbad$mu[2], col=2, pch=17, cex=1.5)
text(fitbad$mu[1], fitbad$mu[2], "EM Local Trap", adj=c(0.5,-1.2), col=2, font=2)
```

<br/> Example 5.31 - What Could Go Wrong? Residual Analysis

```r
ts.diag(fitgood$resid, gg=TRUE, col=4, Qstat=FALSE)
ts.diag(fitbad$resid, gg=TRUE, col=6, Qstat=FALSE)
```

<br/> GNP 

```r
library(astsa)
library(MSwM) # install.packages("MSwM") if necessary
##-- data and model
gr = diff(log(gnp)) # quarterly GNP growth rate
mod = lm(gr ~ 1)
fit1 = msmFit(mod, k=2, p=1, sw=c(TRUE, FALSE, TRUE))
summary(fit1)

##-- extract states
smoProb = fit1@Fit@smoProb
statehat = apply(smoProb, 1, which.max)
culers = ifelse(statehat==2, 4, 2)
dates = time(gr)[-1]

##-- graphics
tsplot(dates, gr[-1], ylab="GNP Growth Rate", col=8, gg=TRUE)
points(dates, gr[-1], bg=culers , pch=21)
abline(v = 1984.5, lty=2, col=2)
text(1984.5, max(gr, na.rm=TRUE), "1984:Q3", pos=4, col=2, cex=0.8)

##-- residual diagnostics
res_all_k = msmResid(fit1, regime = 1:2) # raw, per-regime
sig = fit1@std # regime-specific sigmas
res_std_k = sweep(res_all_k, 2, sig, "/") # divide each column by its own sigma
res_std_pooled = rowSums(res_std_k*fit1@Fit@smoProb[-1,]) # then pool
ts.diag(res_std_pooled, col=4, Qstat=FALSE, gg=TRUE)
```


<br/><br/><br/>

[<sub>top</sub>](#table-of-contents)

<br/>