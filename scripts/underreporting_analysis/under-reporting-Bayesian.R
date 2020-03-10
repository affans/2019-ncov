## under-reporting of 2019-nCov cases 
## Affan Shoukat, Feb 2020

## This script computes the level of underreporting using a Bayesian approach.
## The methodology is adapated from Stoner et al. "A Hierarchical Framework for Correcting Under-Reporting in Count Data"
## This script does not rely on reading data from a file. The total cases and suspected cases are hardcoded in. 

rm(list = ls())
library(ggplot2) # For reproducing plots seen in the paper.
library(nimble) # For MCMC computation using NIMBLE.
library(coda) # For manipulation of MCMC results.
library(mgcv)
library(dplyr)
library(viridis)
library(data.table)

## suspected cases and total cases (incidence, all of china)
suspectedcases = c(54, 37, 393, 1072, 1965, 2684, 4794, 6973, 9239, 12167, 15238, 17988, 19544, 21588, 23214, 23260, 24702)
totalcases = c(77, 149, 131, 259, 444, 688, 769, 1771, 1459, 1737, 1982, 2102, 2590, 2829, 3235, 3887, 3694)
#totalcases = c(291, 440, 571, 830, 1287, 1975, 2744, 4515, 5974, 7711, 9692, 11791, 14380, 17205, 20438, 24324, 28018)
#totalcases = c(291, 440, 571, 830, 1287, 1975, 2744, 4515, 5974, 7711, 9692, 11791)
#suspectedcases = c(27, 26, 257, 680, 1118, 1309, 3806, 2077, 3248, 4148, 4812, 5019)

# define the X covariate for data generation
xcov = c(0.58778626,	0.801075269,	0.25,	0.194590533,	0.184308842,	0.204033215,	0.138234765,	0.202538884,	0.136380632,	0.124928078,	0.115098722,	0.104629169, 0.117014548, 0.115861899, 0.122310862, 0.143183409, 0.130088745)
#xcov = c(0.20923913, 0.252971138, 0.186609687, 0.237832874, 0.256499133, 0.258355238, 0.218901224, 
#      0.281737194, 0.196286829, 0.183848434, 0.169778996, 0.151299215, 0.017684009, 0.141209943)
#xcov = c(0.740384615, 0.851428571, 0.337628866, 0.275825346, 0.28425096, 0.344516775,
#         0.168087432, 0.460239085, 0.309963884, 0.295157179, 0.291727995, 0.295183261)
#xcov = c(0.843478261,	0.922431866,	0.592323651,	0.436382755,	0.395756458,	0.42391071,	0.364022287,	0.393018802,	0.392690462,	0.387916289,	0.388768552,	0.395950166)
#wcov = rev(c(0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32))
#wcov = c(0.8,	0.78,	0.76,	0.74,	0.72,	0.7, 0.68,	0.66,	0.64,	0.62,	0.6,	0.58)

# don't need the following 
#wcov = (c(0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42))
#wcov = rep(0, 17)
#wcov = c(0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85)
#wcov = c(0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15)
#wcov = c(0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24)

# center the covariates
xcov = xcov-mean(xcov)
#wcov = wcov-mean(wcov)

#xcov = log(suspectedcases)
#rcov = log(totalcases)

UR_N=length(totalcases) # Number of observations.

## can make into polynomials, but not using them yet.
poly_xcov = poly(xcov)
#poly_wcov = poly(wcov)

# Do we need to center our covariates?

# Model code.
UR_code=nimbleCode({ 
  for(i in 1:n){
    pi[i] <- ilogit(b0 + b1*w[i] + gamma[i])
    lambda[i] <- exp(a0 + a1*x[i])
    z[i] ~ dpois(pi[i]*lambda[i])
    gamma[i]~dnorm(0,sd=epsilon)
  }
  a0 ~ dnorm(-8,sd=1)  ## TB paper: unlikely high case counts over a million
  a1 ~ dnorm(0,sd=10)  ## uninformative
  b0 ~ dnorm(-1.4, sd=0.85)  ## informative, but wide prior
  b1 ~ dnorm(0,sd=10)  ## uninformative
  epsilon ~ T(dnorm(0,1),0,)
})

## we don't use this model
UR_code_fullmodel=nimbleCode({ 
  for(i in 1:n){
    pi[i] <- ilogit(b0 + b1*w[i] + gamma[i])
    lambda[i] <- exp(a0 + a1*x[i])
    y[i] ~ dpois(lambda[i])
    z[i] ~ dbin(prob = pi[i], size = y[i])
    gamma[i]~dnorm(0,sd=epsilon)
  }
  a0 ~ dnorm(-8,sd=1)  ## TB paper: unlikely high case counts over a million
  a1 ~ dnorm(0,sd=10)  ## uninformative
  b0 ~ dnorm(2,sd=0.6)  ## informative prior, why 2 as mean?
  b1 ~ dnorm(0,sd=10)  ## uninformative
  
  epsilon ~ T(dnorm(0,1),0,)
})

UR_code_affan=nimbleCode({ 
  for(i in 1:n){
    #pi[i] <- ilogit(b0 + b1*w[i] + gamma[i])
    pi[i] <- ilogit(tau[i])
    lambda[i] <- exp(a0 + a1*x[i])
    z[i] ~ dpois(pi[i]*lambda[i])
    gamma[i]~dnorm(0,sd=epsilon)
    tau[i] ~ dnorm(0.1, sd=1)
  }
  a0 ~ dnorm(-8,sd=1)  ## TB paper: unlikely high case counts over a million
  a1 ~ dnorm(0,sd=10)  ## uninformative
  #b0 ~ dnorm(2,sd=0.6)  ## informative prior, why 2 as mean?
  #b1 ~ dnorm(0,sd=10)  ## uninformative
  epsilon ~ T(dnorm(0,1),0,)
  #tau ~ dnorm(0.15, sd=1)
})

# Set up data for NIMBLE.
#UR_constants=list(n=UR_N, x=xcov,  w=wcov)
UR_constants=list(n=UR_N, x=xcov)
UR_data=list(z=totalcases)
# Set initial values.
#UR_inits1=list(epsilon=0.25, a0=0, a1=-0.1, b0=0, b1=0.1, gamma=rnorm(UR_N,0,0.25))
#UR_inits2=list(epsilon=0.5, a0=0, a1=-0.1, b0=4, b1=0.1, gamma=rnorm(UR_N,0,0.25))
#UR_inits3=list(epsilon=0.25, a0=0, a1=0.1, b0=6, b1=-0.1, gamma=rnorm(UR_N,0,0.25))
#UR_inits4=list(epsilon=0.5, a0=0, a1=0.1, b0=-2, b1=-0.1, gamma=rnorm(UR_N,0,0.25))

UR_inits1=list(epsilon=0.25, tau=rnorm(UR_N,0,1), a0=0, a1=-0.1, gamma=rnorm(UR_N,0,0.25))
UR_inits2=list(epsilon=0.5, tau=rnorm(UR_N,0.15,1), a0=0, a1=-0.1, gamma=rnorm(UR_N,0,0.25))
UR_inits3=list(epsilon=0.25, tau=rnorm(UR_N,0.85,1), a0=0, a1=0.1, gamma=rnorm(UR_N,0,0.25))
UR_inits4=list(epsilon=0.5, tau=rnorm(UR_N,0.5,1), a0=0, a1=0.1, gamma=rnorm(UR_N,0,0.25))

UR_inits=list(chain1=UR_inits1, chain2=UR_inits2, chain3=UR_inits3, chain4=UR_inits4)

# Build the model.
UR_model <- nimbleModel(UR_code_affan, UR_constants, UR_data, UR_inits)
UR_compiled_model <- compileNimble(UR_model, resetFunctions = TRUE)

# Set up MCMC samplers.
#UR_mcmc_conf <- configureMCMC(UR_model, monitors=c('a0', 'a1', 'b0', 'b1', 'epsilon', 'pi','lambda'),useConjugacy = TRUE)
UR_mcmc_conf <- configureMCMC(UR_model, monitors=c('a0', 'a1', 'tau', 'epsilon', 'pi','lambda'),useConjugacy = TRUE)

UR_mcmc <- buildMCMC(UR_mcmc_conf)
UR_compiled_mcmc <- compileNimble(UR_mcmc, project = UR_model, resetFunctions = TRUE)

# Run the model (a few hours).
UR_samples=runMCMC(UR_compiled_mcmc,inits=UR_inits,
                   nchains = 4, nburnin=50000,niter = 100000, samplesAsCodaMCMC = TRUE,thin=1,
                   summary = FALSE, WAIC = FALSE,setSeed=c(978, 979, 980, 981)) 


# Check chains for convergence.
#plot(UR_samples[,c('a0','a1', 'b0','b1')])
#gelman.diag(UR_samples[,c('a0','a1', 'b0','b1', 'epsilon')])
gelman.diag(UR_samples[,c('a0','a1', 'epsilon')])

### simulate the results
n_sim=10000 # Number of prior samples to simulate.

# Simulate from parameter priors.
prior_alpha=cbind(rnorm(n_sim,-8,1),
                  rnorm(n_sim,0,10))
prior_beta=cbind(rnorm(n_sim,-1.4,0.85),
                 rnorm(n_sim,0,10))
prior_epsilon=qnorm(runif(n_sim,0.5,1),0,1) # 0 truncated normal distribution

# Simulate random effects
prior_gamma=matrix(nrow=n_sim,ncol=UR_N)

for(i in 1:n_sim){
  prior_gamma[i,]=rnorm(UR_N,0,prior_epsilon[i])
} 

# Compute prior distributions for pi and lambda.
prior_pi=expit(prior_beta%*%t(cbind(1,wcov))+prior_gamma)
prior_lambda=exp(prior_alpha%*%t(cbind(1,xcov)))

# Simulate z.
prior_z=t(apply(prior_lambda*prior_pi,1,function(x)rpois(UR_N,x)))
#prior_rate=t(apply(prior_z,1,function(x) x/TBdata$Population))
prior_lmse=apply(prior_z,1,function(x) log(mean((x-totalcases)^2)))

# Combine MCMC chains.
UR_mcmc=do.call('rbind',UR_samples)

# Compute posterior quantities.
posterior_lambda=UR_mcmc[,4:(20)]
posterior_pi=UR_mcmc[, 21:37]

# Simulate z.
posterior_z=t(apply(posterior_pi*posterior_lambda,1,function(x)rpois(UR_N,x)))
posterior_lmse=apply(posterior_z,1,function(x) log(mean((x-totalcases)^2)))

posterior_y=t(apply(posterior_lambda*(1-posterior_pi),1,function(x)rpois(UR_N,x)+totalcases))
#posterior_total_y=t(apply(posterior_y,1,function(x)c(sum(x[1:n_regions]),sum(x[(1:n_regions)+n_regions]),sum(x[(1:n_regions)+2*n_regions]))))
#posterior_total_extra_y=t(apply(posterior_total_y,1,function(x) x-total_obs))

# Ratio of mean log mean squared errors.
mean(exp(posterior_lmse))/mean(exp(prior_lmse))

paste('Coverage of 95% posterior prediction intervals for z ',
      round(mean(totalcases>=apply(posterior_z,2,quantile,0.025)&totalcases<=apply(posterior_z,2,quantile,0.975)),3))

# Predictive quantile plot.
ggplot(data.frame(x=totalcases,l=apply(posterior_z,2,quantile,0.025)-totalcases,
                  u=apply(posterior_z,2,quantile,0.975)-totalcases)) +
  geom_hline(yintercept=0)+
  geom_point(aes(x=x,y=l),col=vp[8]) +
  geom_point(aes(x=x,y=u),col=vp[10]) +
  labs(
    title = "Recorded Tuberculosis Cases",
    y=expression('Predicted '*tilde(z)['t,s']*' - Observed '*z['t,s']),
    x=expression('Observed Number of Cases '*z['t,s'])
  ) +
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA),
    
    plot.title = element_text(size= 17, hjust=0.01, color = "#4e4d47", margin = margin(b = 0.1, t = 0.4, l = 2, unit = "cm")),
    plot.subtitle = element_text(size= 12, hjust=0.01, color = "#4e4d47", margin = margin(b = 0.1, t = 0.43, l = 2, unit = "cm"))
  )+scale_x_sqrt(limits=c(0,10000))

# Produce predictive checking plots.
m1=ggplot(data.frame(prior=log(apply(prior_z,1,mean)),post=log(apply(posterior_z,1,mean))))+
  stat_density(aes(x=prior),adjust=2,alpha=0.5,fill=vp[7])+
  geom_vline(xintercept=log(mean(totalcases)),colour="#22211d")+
  labs(
    y='Prior Density',
    x=expression(log('Sample Mean'))
  )+
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA)
  )

m2=ggplot(data.frame(prior=log(apply(prior_z,1,mean)),post=log(apply(posterior_z,1,mean))))+
  stat_density(aes(x=post),adjust=2,alpha=0.5,fill=vp[7])+
  geom_vline(xintercept=log(mean(totalcases)),colour="#22211d")+
  labs(
    y='Posterior Density',
    x=expression(log('Sample Mean'))
  )+
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA)
  )
v1=ggplot(data.frame(prior=log(apply(prior_z,1,var)),post=log(apply(posterior_z,1,var))))+
  stat_density(aes(x=prior),adjust=2,alpha=0.5,fill=vp[9])+
  geom_vline(xintercept=log(var(totalcases)),colour="#22211d")+
  labs(
    y='Prior Density',
    x=expression(log('Sample Variance'))
  )+
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA)
  )
v2=ggplot(data.frame(prior=log(apply(prior_z,1,var)),post=log(apply(posterior_z,1,var))))+
  stat_density(aes(x=post),adjust=2,alpha=0.5,fill=vp[9])+
  geom_vline(xintercept=log(var(totalcases)),colour="#22211d")+
  labs(
    y='Posterior Density',
    x=expression(log('Sample Variance'))
  )+
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA)
  )


e1=ggplot(data.frame(prior=prior_lmse,post=posterior_lmse))+
  stat_density(aes(x=prior),adjust=2,alpha=0.5,fill=vp[11])+
  labs(
    y='Prior Density',
    x=expression(log('Mean Squared Error'))
  )+
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA)
  )+scale_x_continuous(limits=c(10,35))
e2=ggplot(data.frame(prior=prior_lmse,post=posterior_lmse))+
  stat_density(aes(x=post),adjust=2,alpha=0.5,fill=vp[11])+
  labs(
    y='Posterior Density',
    x=expression(log('Mean Squared Error'))
  )+
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA)
  )


# Fitted values plot. Cant run this without the rate 
ggplot(data.frame(x=100000*TBdata$TB/TBdata$Population,m=100000*apply(posterior_rate,2,mean))) +
  geom_abline(slope=1,intercept=0)+geom_point(aes(x=x,y=m),col=vp[7],alpha=0.75) +
  labs(
    title = "Recorded Tuberculosis Cases",
    subtitle ="per 100,000 People",
    y=expression('Mean Predicted Value'),
    x=expression('Observed Value')
  ) +
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA),
    
    plot.title = element_text(size= 17, hjust=0.01, color = "#4e4d47", margin = margin(b = -0.1, t = 0.4, l = 2, unit = "cm")),
    plot.subtitle = element_text(size= 12, hjust=0.01, color = "#4e4d47", margin = margin(b = 0.1, t = 0.43, l = 2, unit = "cm"))
  )


# Part Four: Results
pi_means <- apply(posterior_pi,2,mean)
#inverse_index=sort(region_index[1:n_regions],index.return=TRUE)$ix
#pi_spatial <- cbind(pi_means[inverse_index],pi_means[inverse_index+557],pi_means[inverse_index+1114]) %>%
#  logit() %>% apply(.,1,mean) %>% expit()
pdf = data.table(means = colMeans(posterior_y), totalcases)
gg = ggplot(pdf) 
gg = gg + geom_point(aes(x = 1:17, y=means), color="Red")
gg = gg + geom_point(aes(x = 1:17, y=totalcases))
gg

#####

# vals1 = c(mean(UR_samples[, c('lambda[1]')]$chain1),
# mean(UR_samples[, c('lambda[2]')]$chain1),
# mean(UR_samples[, c('lambda[3]')]$chain1),
# mean(UR_samples[, c('lambda[4]')]$chain1),
# mean(UR_samples[, c('lambda[5]')]$chain1),
# mean(UR_samples[, c('lambda[6]')]$chain1),
# mean(UR_samples[, c('lambda[7]')]$chain1),
# mean(UR_samples[, c('lambda[8]')]$chain1),
# mean(UR_samples[, c('lambda[9]')]$chain1),
# mean(UR_samples[, c('lambda[10]')]$chain1),
# mean(UR_samples[, c('lambda[11]')]$chain1), 
# mean(UR_samples[, c('lambda[12]')]$chain1))
# 
# 
# vals2 = c(mean(UR_samples[, c('lambda[1]')]$chain2),
#           mean(UR_samples[, c('lambda[2]')]$chain2),
#           mean(UR_samples[, c('lambda[3]')]$chain2),
#           mean(UR_samples[, c('lambda[4]')]$chain2),
#           mean(UR_samples[, c('lambda[5]')]$chain2),
#           mean(UR_samples[, c('lambda[6]')]$chain2),
#           mean(UR_samples[, c('lambda[7]')]$chain2),
#           mean(UR_samples[, c('lambda[8]')]$chain2),
#           mean(UR_samples[, c('lambda[9]')]$chain2),
#           mean(UR_samples[, c('lambda[10]')]$chain2),
#           mean(UR_samples[, c('lambda[11]')]$chain2), 
#           mean(UR_samples[, c('lambda[12]')]$chain2))
# 
# vals3 = c(mean(UR_samples[, c('lambda[1]')]$chain3),
#           mean(UR_samples[, c('lambda[2]')]$chain3),
#           mean(UR_samples[, c('lambda[3]')]$chain3),
#           mean(UR_samples[, c('lambda[4]')]$chain3),
#           mean(UR_samples[, c('lambda[5]')]$chain3),
#           mean(UR_samples[, c('lambda[6]')]$chain3),
#           mean(UR_samples[, c('lambda[7]')]$chain3),
#           mean(UR_samples[, c('lambda[8]')]$chain3),
#           mean(UR_samples[, c('lambda[9]')]$chain3),
#           mean(UR_samples[, c('lambda[10]')]$chain3),
#           mean(UR_samples[, c('lambda[11]')]$chain3), 
#           mean(UR_samples[, c('lambda[12]')]$chain3))
# 
# vals4 = c(mean(UR_samples[, c('lambda[1]')]$chain4),
#           mean(UR_samples[, c('lambda[2]')]$chain4),
#           mean(UR_samples[, c('lambda[3]')]$chain4),
#           mean(UR_samples[, c('lambda[4]')]$chain4),
#           mean(UR_samples[, c('lambda[5]')]$chain4),
#           mean(UR_samples[, c('lambda[6]')]$chain4),
#           mean(UR_samples[, c('lambda[7]')]$chain4),
#           mean(UR_samples[, c('lambda[8]')]$chain4),
#           mean(UR_samples[, c('lambda[9]')]$chain4),
#           mean(UR_samples[, c('lambda[10]')]$chain4),
#           mean(UR_samples[, c('lambda[11]')]$chain4), 
#           mean(UR_samples[, c('lambda[12]')]$chain4))
# 
# 
# df = data.table(data = totalcases, vals1, vals2, vals3, vals4)
# gg = ggplot(df)
# gg = gg + geom_line(aes(x = 1:12, y=vals1), color='red')
# gg = gg + geom_line(aes(x = 1:12, y=vals2), color='blue')
# gg = gg + geom_line(aes(x = 1:12, y=vals3), color='green')
# gg = gg + geom_line(aes(x = 1:12, y=vals4))
# 
# gg = gg + geom_point(aes(x = 1:12, y=totalcases))
# gg

posterior_pi=UR_mcmc[, 21:37]
adf = data.table(posterior_pi)
adf$sim = 1:nrow(adf)
adfm = melt(data = adf, id.vars = c('sim'))
gg = ggplot(adfm) 
gg = gg + hist

gg = adfm %>% 
  filter(variable == 'pi[1]') %>%
  ggplot() 

gg  = gg +  geom_histogram(aes(x = 'value'), stat='count', position="identity", colour="grey40", alpha=0.2, bins = 10)
