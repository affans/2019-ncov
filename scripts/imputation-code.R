rm(list=ls())
library(mice)
library(VIM)
library(lattice)
library(ggplot2)

setwd("/Users/abmlab/OneDrive/Documents/postdoc projects/2019-ncov/")
dat <- fread('data/ncovdata.csv', col.names = c("time", "date", "cum.cases", "cum.cases.intl"))
dat = data.table(dat)

#dat = dat[, .(t = V1, cases=V2)] # rename columns

## grid represents the missing data patterns. 
## the first row represents we have complete information for `cases` for each `t`
## the second row represents the number of observations that are missing for each `t`
gp <- md.pattern(dat)

## Number of observations per patterns for all pairs of variables
p <- md.pairs(dat)
## The pattern rr represents the number of observations where
## both pairs of values are observed. The pattern rm represents the 
## exact opposite, these are the number of observations where both variables 
## are missing values. The pattern mr shows the number of observations 
## where the first variable’s value (e.g. the row variable) is observed 
## and second (or column) variable is missing. The pattern mm is just the opposite.


## The pbox function below wil plot the marginal distribution of a variable 
## within levels or categories of another variable. 
## Here we obtain a plot of the distibution of the variable t against cases. 
pbox(as.matrix(dat),  pos=1)

## impute the data using mice/PMM
## we impute 10 times (5-10 imputations is optimal according to literature). 
## then take the average of the 10 imputations. 
imp1 = mice(dat[, c(1,3)], m = 10, method = "pmm", seed = 5, maxit=5)
print(imp1)

## print the dataset if needed 
print(imp1$imp)

## pool the results of the 10 imputations in long format.
imp_tot2 <- complete(imp1, "long", inc = F)
imp_tot2 <- data.table(imp_tot2)

## I convert it back to wide format (essentially imp1$imp but I like doing it my way)
cases.dt = dcast(data = imp_tot2, formula = time ~ .imp, value.var = "cum.cases" )
cases.dt = data.table(cases.dt)  ## convert to data.table

## when taking the rowMeans, make sure to ignore the first column of time
avgdt = data.table(t=cases.dt$t, avg=rowMeans(cases.dt[, 2:11]))

## plot the imputation
gg = ggplot()
gg = gg + geom_point(data = avgdt, aes(x=t, y=avg), color='red')
gg = gg + geom_line(data = avgdt, aes(x=t, y=avg), color='red')
gg = gg + geom_point(data = dat, aes(x=time, y=cum.cases), color='black')
gg = gg + geom_line(data = dat, aes(x=time, y=cum.cases), color='black')
gg

## save data for Seyed if needed
cases.dt$avg = avgdt$avg
fwrite(x = cases.dt, file = "imputed-data.dat")





