This is a list of the R code in the text. New scripts and data sets were included in the package `astsa` to cover many of the examples. So rather than repeating it... install the package (once) and then load it each time you try something in the text. 

```r
if (!requireNamespace("astsa")){ 
     install.packages("astsa") }  # install astsa if not there
library(astsa)                    # load it as needed for examples
```




### Table of Contents
  

  * [Chapter 1](#chapter-1)
  * [Chapter 2](#chapter-2)



### Chapter 1

RNG

```r
N = 30
x = c(1) # the bad seed (they are all bad)
for (n in 2:N){ x[n] = (12*x[n-1] + 4) %% 2^32 }
x
```

Tutorial 

```r
2+2            # addition (input)
5*5 + 2        # multiplication and addition
5/5 - 3        # division and subtraction
log(exp(pi))   # log, exponential, pi
sin(pi/2)      # sinusoids
2^(-2)         # power
sqrt(8)        # square root
-1:5           # sequences
seq(1, 10, by=2)  # sequences
rep(2, 3)         # repeat 2 three times

x <- 1 + 2 # put 1 + 2 in object x
x = 1 + 2 # same as above with fewer keystrokes
1 + 2 -> x # same
x # view object x
(y = 9 * 3) # put 9 times 3 in y and view the result
(z = rnorm(5)) # put 5 standard normals into z and print z

x = c(1, 2, 3) # numeric vector
y = c("one","two","three") # character vector
z = c(TRUE, TRUE, FALSE) # logical vector
length(y) # length of a vector

( x = c(0, 1, NA) )
2*x
x/0
is.na(x)
sum(is.na(x)) # number of TRUEs
sum(!is.na(x)) # number of FALSEs

par(mfcol=c(2,2), cex=.9) # multifigure plot, 2 rows 2 cols - cex for larger labels

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
X = cumsum(c(0,rnorm(N-1)))
Y = cumsum(c(0,rnorm(N-1)))
tsplot(X, Y, xlab='X', gg=TRUE, col=4, main="Brownian Motion")
points(c(0, X[N]), c(0, Y[N]), col=3:2, pch=19)
legend('topleft', legend=c('start', 'end'), lty=0, col=3:2, pch=19, bty='n')

# functions
XtY <- function(x,y){ x * y } # the script
XtY(20, .5) # now try it

# more functions
PrXltY <- function(ntrials=10000, meanX, meanY){
# the inputs, ntrials is optional, but means for X and Y are not
ntrials = ntrials # use the value specified
X = rexp(ntrials, rate=1/meanX) # generate X exponentials
Y = rexp(ntrials, rate=1/meanY) # generate Y exponentials
c(Estimate = mean(X < Y), True = meanY/(meanX+meanY)) # display results
} # end
PrXltY(meanX=1500, meanY=1000)

# CLT
set.seed(90210)
par(mfrow=c(1, 3))  
for (n in c(5, 20, 100)){                    # sample sizes
 x = replicate(500, mean(rbinom(n, 10, .2))) # sample means
 hist(x, col='lightblue', breaks=20, prob=TRUE, main=paste('n =', n), xlab=bquote(bar(X)[~n]))
 curve(dnorm(x, 2, sqrt(1.6/n)), 0,4,501, add=TRUE, col=2, lwd=2)
 abline(v=2, col=2, lwd=2)
 }
```

[<sub>top</sub>](#table-of-contents)

<br/>

### Chapter 2