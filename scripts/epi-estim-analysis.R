## contact analysis 2019-nCov 
## Affan Shoukat, Feb 2020

## this script uses the EpiEstim R package to calculate temporal R0 values. 
## As input, it requires fitted incidence data from either analysis_one or analysis_two file. 
## It will first use the fitted incidence data to simulate epidemics using negative-binomial distributions.
## It will take each epidemic curve and feed that in the R package. 

rm(list=ls())
library(mice)
library(VIM)
library(lattice)
library(ggplot2)
library(EpiEstim)
library(dplyr)

setwd("/Users/abmlab/OneDrive/Documents/postdoc projects/2019-ncov/")
savstr = "baseline"
dat_est <- fread('data/estimated_incidence_baseline.dat')
dat_est = data.table(dat_est)

# remove NA 
dat_est = dat_est[-which(is.na(dat_est$cv)), ]

# convert the time column to the dates column 
# dat_est$time = factor(dat_est$dates, levels = dat_est$dates) ## will keep the factor order


#### simulate binomial
#' Simulate NegBinomial-distributed incidence 
#' with mean the reported daily incidence.
sim_inc_nbinom <- function(seed, dati) {
  set.seed(seed)
  mean.inc <- dati$cases
  #szv <-  1/(cv.sim^2 - 1/mean.inc)
  szv <- 1/(dati$cv^2 - 1/mean.inc)
  
  # For numerical stability:
  szv[is.infinite(szv)] <- 1e2 
  szv[szv <= 0]         <- 50
  
  # Checks:
  if(0){
    v <- mean.inc + mean.inc^2/szv
    check.cv <- sqrt(v)/ mean.inc # should be close to input `cv.sim`
  }
  
  # Simulate observation process:
  sim.inc <- rnbinom(n  = nrow(dati), 
                     mu = mean.inc, 
                     size = szv)
  
  sim.inc[is.na(sim.inc)] <- 0
  cuminc <- cumsum(sim.inc)
  
  # cv.sim.ts <- cv_window(t = dati$t, 
  #                        y = sim.inc, 
  #                        w = cv.window)
  
  # Checks:
  if(0){
    plot(dati$cv,typ='o')
    abline(h=cv.sim, lty=2)
    lines(cv.sim.ts, col='red')
  }
  
  res <- cbind(dati, 
               sim.inc = sim.inc, 
               seed    = seed, 
               #cv.sim  = c(rep(NA,cv.window+1),cv.sim.ts),
               cuminc = cuminc)
  return(res)
}

simulate_obsprocess <- function(n.mc,  dati) {
  tmp.df <- lapply(1:n.mc, sim_inc_nbinom, 
                   dati = dati)
  dfsim  <- do.call('rbind', tmp.df)
  return(dfsim)
}

# The fitted observation process:
n.mc  <- 500
dfsim <- simulate_obsprocess(n.mc, dat_est)

dfsim.s <- dfsim %>%
  group_by(time) %>%
  summarise(m.cv = mean(cv),
            min.cv = min(cv),
            max.cv = max(cv))

## plotting of cv
# g <- dfsim.s %>%
#   ggplot()+
#   geom_line(aes(x=time, y=m.cv),alpha=0.7)+
#   geom_ribbon(aes(x=time, ymin=min.cv, ymax=max.cv), alpha=0.2)+
#   geom_point(data = dat_est, aes(x=time, y=cv))+
#   ggtitle(paste('Coeff. of Variation of Daily Incidence. CV window =',cv.window), 
#           subtitle = 'Points = data')
# plot(g)

# Check how the fit looks:
g.inc1 <- dfsim %>%
  filter(seed <=9) %>%
  ggplot() + 
  geom_line(aes(x=time, y=sim.inc, group=1),alpha=0.8) + 
  geom_point(data = dat_est, aes(x=time, y=cases, colour="black"), 
             shape=16, alpha=0.6)+
  facet_wrap(~seed, ncol=3)+
  ggtitle('Examples of simulated daily incidence')
plot(g.inc1)


# plot all of them
dfsim_summ <- dfsim %>%
  group_by(time) %>%
  summarise(m = mean(sim.inc, na.rm=T),
            md = median(sim.inc, na.rm=T),
            qlo = quantile(sim.inc, probs = 0.025, na.rm=T),
            qhi = quantile(sim.inc, probs = 0.975, na.rm=T))

g.inc2 <- dfsim_summ %>% ggplot() +
  geom_line(aes(x=time, y=m, group=1))+
  geom_ribbon(aes(x=as.numeric(time), ymin=qlo, ymax=qhi), alpha=.2)+
  geom_point(data = dat_est, aes(x=time, y=cases, colour="red"), 
             shape=16,  alpha=0.8)

plot(g.inc2)
                


# Cumulative incidence ON JANUARY 29, timeindex = 60
jan29.cuminc <- dfsim %>%
  filter(time <= 60) %>%
  group_by(seed) %>%
  summarize(final.size.jan29=max(cuminc))

feb10.cuminc <- dfsim %>%
  group_by(seed) %>%
  summarize(final.size.feb10=max(cuminc))

df.cuminc = data.table(cbind(jan29.cuminc, feb10.cuminc))
df.cuminc[, c(3) := NULL]

df.cuminc.m = melt(data = df.cuminc, id.vars = "seed", measure.vars = c('final.size.jan29', 'final.size.feb10'))

g.finalsize <- df.cuminc.m %>%
  ggplot()+
  geom_histogram(aes(x=value), bins=10) + 
  facet_wrap(~variable, ncol=2,  scales = "free_x") +
  labs(title="Final size Jan29 and Feb growth")
plot(g.finalsize)

# Sliding window size (in days) to estimate R_eff:
r.estim.window <- 7
z <- list()
for (i in unique(dfsim$seed)){  #i=1
  # print(i)
  dfi <- dfsim %>% 
    filter(seed == i ) %>%
    select(time, sim.inc)
  
  # Incidence start with several days at 0
  # in simulations, so remove them in order
  # not to break `estimate_R`:
  idx.pos <- dfi$sim.inc>0
  idx.pos[is.na(idx.pos)] <- FALSE
  t.not.0 <- which(idx.pos)[1]
  idx.pos[t.not.0:length(idx.pos)] <- TRUE
  dfi     <- dfi[idx.pos,]
  
  t_start <- 2:(nrow(dfi)-r.estim.window)  
  t_end   <- t_start + r.estim.window
  
  R.psi <- estimate_R(incid = dfi$sim.inc, 
                      method="uncertain",
                      config = make_config(list(t_start = t_start, 
                                                t_end   = t_end,
                                                mean_si = 7.5, std_mean_si = 0.5,
                                                min_mean_si = 6, max_mean_si = 12,
                                                std_si = 3.5, std_std_si = 0.5,
                                                min_std_si = 2, max_std_si = 5,
                                                n1 = 20, n2 = 20)))
  
  # plot(R.psi)
  tmp    <- R.psi$R
  tmp$mc <- i
  # encode the correct time from the data table
  tmp$t  <- dfi$time[2:tail(t_start, n=1)] #t.not.0 -1 + tmp$t_start
  z[[i]] <- tmp
}


df.R <- do.call('rbind',z)
df.R.s <- df.R %>%
  dplyr::rename(R = `Mean(R)`) %>%
  dplyr::rename(R.lo = `Quantile.0.025(R)`) %>%
  dplyr::rename(R.hi = `Quantile.0.975(R)`) %>%
  group_by(t) %>%
  summarize(R.mean = mean(R, na.rm = TRUE),
            R.lo   = mean(R.lo, na.rm = TRUE),
            R.hi   = mean(R.hi, na.rm = TRUE))

g.R <- df.R.s %>%
  ggplot(aes())+
  geom_line(aes(x=t, y=R.mean, group=1), size=1)+
  geom_ribbon(aes(x=as.numeric(t), ymin=R.lo, ymax=R.hi), alpha=0.2)+
  geom_hline(yintercept = 1, linetype='dotted')+
  ggtitle('Estimate of the Effective Reproduction Number',
          subtitle = paste('Mean and 95CI. Sliding window =',r.estim.window))
plot(g.R)

## save R values for seyed 

aa = data.table(mc = df.R$mc, time = df.R$t, rval = df.R[, names(df.R)[3]])
aa_dcast = dcast(data = aa, formula = mc~time, value.var = 'rval')
#min(aa$time)

# time index matching table
# tdf = data.table(tidx = dat_est$time, tstr = dat_est$date)
# fwrite("time-mapping.csv", x = tdf)
# names(aa_dcast) = c("mc", tdf$tstr[min(aa$time):max(aa$time)])


## save the data for seyed.
fwrite(paste0(savstr, "_incidence.csv"), x = dfsim_summ)  
#fwrite(paste0(savstr, "_finalsize.csv"), x =data.table(df.cuminc))
fwrite(paste0(savstr, "_rvalues.csv"), x = aa_dcast )
fwrite(paste0(savstr, "_rvalues_summary.csv"), x = df.R.s)
