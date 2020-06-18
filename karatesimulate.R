library(ergm)
library(Bergm)
library(abc)
library(ks)
library(matrixcalc)
library(hdrcde)
library(multiplex)
library(igraph)
library(spatstat)
library(plotly)
#simulating from ERGM
karate = read.gml("karate.gml")
network(karate)
plot(karate)
karate = read.graph("karate.gml", format = "gml")
as.matrix(get.adjacency(karate, type = "both"))

k2 = network(as_edgelist(karate), directed = F)
plot(k2)


#fit the ergm 
myergm = ergm(k2 ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T))




#prior mean and var for posterior
priormean = c(-2,0,0)
priorsig = 4*diag(3)
p = 3
####appeoximate exact exchange algorithm

ergmexchange = bergm(k2 ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T), prior.mean = priormean,
                     prior.sigma = priorsig, burn.in = 1000, main.iters = 5000, aux.iters = 10000)
summary(ergmexchange)


##################VI########################


ergmVI <- function(myergm, priormean, priorsig){
  K = 5000
  #myergm: the ergm object u fitted
  #### Initialization ####
  #simulate large number of s(y) from \theta_{ML}
  mu_ML = as.numeric(myergm$coef)
  sumstatpool = simulate(myergm, nsim = K, output = "stats")
  sumstatpool4zfactor = sumstatpool
  logZml = as.numeric(myergm$coef %*% myergm$target.stats - myergm$mle.lik)
  
  y_obs = myergm$target.stats
  
  d = length(y_obs)
  
  #prior mean and cov
  mu_0 = priormean
  Sig_0 = priorsig
  
  
  #initialize
  mu_t = mu_ML
  C_t = 0.6 * diag(d)
  #mu_t = ergmmean
  #C_t = ergmchol
  
  
  #start ADAM
  
  elbostore <- rep(NA,1000)
  elboold = -1e20
  elbonew = -1e10
  iter=0
  
  mugrad1stold = rep(0,d)
  mugrad2ndold = rep(0,d)
  Cgrad1stold = rep(0,d*(d+1)/2)
  Cgrad2ndold = rep(0,d*(d+1)/2)
  
  while (elbonew > elboold){
    for (i in 1:1000){
      
      s_t = rnorm(d)
      
      #CC^T = Sigma the covariance, C is lower triangular with pos. diag. elements
      #mu is the mean vector
      
      theta_t = as.numeric(C_t %*% s_t + mu_t)
      
      #monitoring ESS
      logwtilde = apply(sumstatpool, MARGIN = 1, FUN = function(s){return(s %*% (mu_t - mu_ML))})
      logwtilde = logwtilde - max(logwtilde)
      wtilde = exp(logwtilde)
      wtilde = wtilde/sum(wtilde)
      ESStilde = 1/sum(wtilde^2)
      
      #trigger resampling procedure if ess<K/3
      if (ESStilde < K/3) {indexdraw = sample(1:K, K, prob = wtilde); sumstatpool = sumstatpool[indexdraw,]; mu_ML = mu_t}
      
      #estimating Eytheta using IS
      logw = apply(sumstatpool, MARGIN = 1, FUN = function(s){return(s %*% (theta_t - mu_ML))})
      logw = logw - max(logw)
      w = exp(logw)
      w = w/sum(w)
      
      Eytheta = colSums(w * sumstatpool)
      
      
      #stochastic gradient
      Grad_t = y_obs - Eytheta - as.numeric(solve(Sig_0) %*% (theta_t - mu_0))
      
      mugrad1stnew = 0.9 * mugrad1stold + (1-0.9) * Grad_t
      mugrad2ndnew = 0.999 * mugrad2ndold + (1-0.999) * Grad_t^2
      
      muhat1st = mugrad1stnew/(1-0.9^(iter*1000+i))
      muhat2nd = mugrad2ndnew/(1-0.999^(iter*1000+i))
      
      
      mu_updated = mu_t + 0.0008 * muhat1st/(sqrt(muhat2nd) + 1e-8)
      
      mugrad1stold = mugrad1stnew
      mugrad2ndold = mugrad2ndnew
      
      
      
      C_tprime = C_t
      diag(C_tprime) = log(diag(C_t))
      
      D_Ct = diag(as.numeric(vech(diag(diag(C_t)) + matrix(1,d,d) - diag(d))))
      
      Cgrad1stnew = 0.9 * Cgrad1stold + (1-0.9) * as.numeric(D_Ct %*% vech(Grad_t %*% t(s_t)))
      Cgrad2ndnew = 0.999 * Cgrad2ndold + (1-0.999) * as.numeric(D_Ct %*% vech(Grad_t %*% t(s_t)))^2
      
      Chat1st = Cgrad1stnew/(1-0.9^(iter*1000+i))
      Chat2nd = Cgrad2ndnew/(1-0.999^(iter*1000+i))
      
      vechCprime_updated = vech(C_tprime) + 0.0008 * Chat1st/(sqrt(Chat2nd) + 1e-8)
      
      
      Cgrad1stold = Cgrad1stnew
      Cgrad2ndold = Cgrad2ndnew
      
      Cprime_updated = lower.triangle(invvech(vechCprime_updated))
      
      C_updated = Cprime_updated
      diag(C_updated) = exp(diag(Cprime_updated))
      
      logzfactor = sumstatpool4zfactor %*% (theta_t - as.numeric(myergm$coef))
      logzfactor = log(mean(exp((logzfactor - max(logzfactor))))) + max(logzfactor)
      
      
      elbostore[i] = theta_t%*%y_obs - logZml - logzfactor - 0.5*log(det(Sig_0)) - 
        0.5*t(theta_t-mu_0)%*%solve(Sig_0) %*% (theta_t-mu_0) + log(det(C_t)) + 0.5*t(s_t)%*%s_t
      
      
      mu_t = mu_updated
      C_t = C_updated
    }
    elboold = elbonew
    elbonew = mean(elbostore)
    iter = iter+1
    print(c(iter, elbonew))
    if (iter > 11) {break()}
  }
  
  
  return(list(mu_t, C_t, C_t %*% t(C_t), elbonew))
}

ergmgaus = ergmVI(myergm, priormean, priorsig)




######adj-lkd###################################
ergmadjlkd = bergmC(k2 ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T), prior.mean = priormean, prior.sigma = priorsig, thin = 5, estimate = "MLE")
summary(ergmadjlkd)




#####################ABC#############################

priorsamplingergm <- function(N,p,priormu, priorsig){
  #priormu = mean vector of length p
  #priorsig = p*p diagonal mtrx
  
  priorpar = matrix(rnorm(N*p, mean = priormu, sd = sqrt(diag(priorsig))), nrow = N, ncol = p, byrow = T)
  
  priorpar = priorpar[order(priorpar[,1], decreasing = T),]
  
  networklst = lapply(1:N,function(i){return(i)})
  
  kinit = simulate(myergm, nsim = 1, coef = priorpar[1,], output = "network", control = control.simulate.ergm(MCMC.burnin = 50000, MCMC.interval = 5000))
  
  networklst[[1]] = kinit
  
  priory = matrix(NA, nrow = N, ncol = p)
  
  priory[1,] = summary(networklst[[1]] ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T))
  
  for (i in 2:N){
    if (i %% 100 == 0) {print(paste0(100*i/N, "%"))}
    k22 = networklst[[i-1]]
    networklst[[i]] = simulate(k22 ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T), nsim = 1, coef = priorpar[i,], 
                               output = "network", control = control.simulate.formula(MCMC.burnin = 20000, MCMC.interval = 10000))
    priory[i,] = summary(networklst[[i]] ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T))
  }
  
  return(list(priorpar, priory, networklst))
}






jointpriorergm30000 = priorsamplingergm(30000,3,priormean, priorsig)


#####trimming the joint data########

hist(jointpriorergm30000[[2]][,1])

keepindx1 = which((jointpriorergm30000[[2]][,1] < 300) & (jointpriorergm30000[[2]][,1] != 0))

keepindx = which((jointpriorergm30000[[2]][,1] < 300) & (jointpriorergm30000[[2]][,1] != 0) & (jointpriorergm30000[[2]][,2] != 0))


jointpriorergmtrim2 = list(1,2,3)

jointpriorergmtrim2[[1]] = jointpriorergm30000[[1]][keepindx,]
jointpriorergmtrim2[[2]] = jointpriorergm30000[[2]][keepindx,]
jointpriorergmtrim2[[3]] = jointpriorergm30000[[3]][keepindx]


save(jointpriorergmtrim2, file = "jointergmtrim2.RData")

rm(jointpriorergm30000)




###combinig trim1 and trim2 data (total 14223)
ergmdata = list(1,2,3)
ergmdata[[1]] = rbind(jointpriorergmtrim[[1]], jointpriorergmtrim2[[1]])
ergmdata[[2]] = rbind(jointpriorergmtrim[[2]], jointpriorergmtrim2[[2]])
ergmdata[[3]] = c(jointpriorergmtrim[[3]], jointpriorergmtrim2[[3]])


save(ergmdata, file = "ergmdata.RData")


rm(jointpriorergmtrim)
rm(jointpriorergmtrim2)


#################################
###########plot time#############
#################################
ergmcompare <- function(i){
  plot(density(ergmexchange$Theta[,i]),col=1, main = paste0("dim = ", i))
  lines(density(ergmABC2$adj.values[,i]),col=2)
  lines(density(ergmadjlkd$Theta[,i]),col=3)
  lines(density(rnorm(10000, mean = ergmgaus[[1]][i], sd = sqrt(ergmgaus[[3]][i,i]))),col=4)
  legend("topright", col = c(1,2,3,4), legend = c("exchange", "ABC_reg","adj_lkd", "VI"), lty = 1)
}

ergmcompare(1)
ergmcompare(2)
ergmcompare(3)




######################################################################################################################################
###############################################getting p_hat##########################################################################
######################################################################################################################################




#ergmdata is the dataset

#only take the top 25% nearest neighours as training set (3555 data points)

abcnearestquarter <- abc(myergm$target.stats, ergmdata[[1]],  ergmdata[[2]],tol = 0.01, method = "loclinear")

for (i in 1:3){
  plot(density(ergmABC2$adj.values[,i]))
  lines(density(abcnearestquarter$adj.values[,i]), col = 2)
  lines(density(ehh$Theta[,i]), col = 3)
}



#use ABC to find the nearest neighbours

abcnearestquarter <- abc(myergm$target.stats, ergmdata[[1]],  ergmdata[[2]],tol = 0.25, method = "rejection")
indxnear = which(abcnearestquarter$region)
ergmdistorttrain  = list(1,2,3)
ergmdistorttrain[[1]] = ergmdata[[1]][indxnear,]
ergmdistorttrain[[2]] = ergmdata[[2]][indxnear,]
ergmdistorttrain[[3]] = ergmdata[[3]][indxnear]

save(ergmdistorttrain, file = "ergmdistorttrain.RData")




###################given theta_i, y_i, get phat_i for different methods##############

####Given simulated approx post samples, find phat
getphat <- function(actualtheta, sampledtheta){
  #actualtheta: 1*p vector
  #sampledtheta: n*p mtrx sampled from approx post
  
  #marginal ps
  p = length(actualtheta)
  phatmarginal = sapply(1:p, FUN = function(i){mean(sampledtheta[,i]<=actualtheta[i])}, simplify = T)
  names(phatmarginal) = as.character(1:p)
  #conditionals
  phatcond = numeric()
  for (i in 1:p){
    for (j in 1:p){
      if (i != j){
        myergmckde = cde(x = sampledtheta[,j], y = sampledtheta[,i], x.margin = actualtheta[j], a = 0.05, b=0.05)
        delt = myergmckde$y[2] - myergmckde$y[1]
        indica = myergmckde$y <= actualtheta[i]
        if (!any(indica)) {condiprob = 0} #if observed less than everyone, return 0
        else {condiprob = delt * sum(myergmckde$z[indica])}
        phatcond = c(phatcond, condiprob)
        names(phatcond) = c(names(phatcond)[-length(phatcond)], paste0(i, "|",j))
      }
    }
  }
  return(c(phatmarginal, phatcond))
}












#ergmdistorttrain is the set of data we would like to train using MDN

#but for ABC, the synthetic dataset is still the full ergmdata set


###################ABC#########################

tktk = Sys.time()

p=3

NN = length(indxnear)

phatttunadj = matrix(NA,ncol = p^2, nrow = NN)
phatttadj = matrix(NA,ncol = p^2, nrow = NN)


###for ABC, only calculate those pars in the good region (nonzero, # of edges not greater than 300)

for (i in 1:NN){
  tryCatch({
    if(i%%100 == 0) {print(paste0(100*i/NN, "%"))}
    abcposition = indxnear[i]
    pseudoobs = ergmdistorttrain[[2]][i,]
    pseudopar = ergmdistorttrain[[1]][i,]
    abcergm <- abc(pseudoobs, ergmdata[[1]][-abcposition,], ergmdata[[2]][-abcposition,],tol = 0.01, method = "loclinear")
    
    phatttunadj[i,] = getphat(pseudopar, abcergm$unadj.values)*0.9998 + 1e-5
    phatttadj[i,] = getphat(pseudopar, abcergm$adj.values)*0.9998 + 1e-5}, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}


abcphattrain = cbind(phatttadj, ergmdistorttrain[[2]])
write.csv(abcphattrain, file = "ergmabc_train.csv")

print(Sys.time() - tktk)


#we dont need the full synthetic data anymore
rm(ergmdata)








##############ADJ_LKD#########################

bergmCgetphat <- function(actualpar, simulatednet){
  p = length(actualpar)
  ehhCforcali = tryCatch({
    bergmC(simulatednet ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T), 
           prior.mean = priormean, prior.sigma = priorsig, thin = 5, estimate = "MLE",
           main.iters = 20000)
  }, error=function(e){return(NA)})
  if (is.na(ehhCforcali)) {return(rep(NA,p^2))}
  else {
    phattt = getphat(actualpar,ehhCforcali$Theta)
    phattt = phattt*0.9998 + 1e-5
    return(phattt)
  }
}



library(foreach)
library(doParallel)
cl = makeCluster(10)
registerDoParallel(cl)
getDoParWorkers()

NN = length(indxnear)

tktk = Sys.time()

ergmadjlkd_p_hat3556 = foreach(i = 1:NN, .combine = rbind, .export = c("bergmCgetphat","getphat", "ergmdistorttrain"), .packages = c("ergm", "Bergm", "abc", "ks", "matrixcalc", "hdrcde")) %dopar% bergmCgetphat(ergmdistorttrain[[1]][i,],ergmdistorttrain[[3]][[i]])

stopCluster(cl)

print(Sys.time() - tktk)


ergmadjlkd_train3556 = cbind(ergmadjlkd_p_hat3556, ergmdistorttrain[[2]])

naindx = which(is.na(ergmadjlkd_train3556[,1]))

ergmadjlkd_trainnarm = ergmadjlkd_train3556[-naindx,]

write.csv(ergmadjlkd_trainnarm, file = "ergmadjlkd_trainnarm.csv")





#####################VI############################

bergmVIgetphat <- function(actualpar, simulatednet, priormean, priorsig){
  p = length(actualpar)
  vimu = NA
  viSig = NA
  tryCatch({
    myergm = ergm(simulatednet ~ edges + gwesp(decay = 0.2, fixed = T) + gwdegree(decay = 0.8, fixed = T))
    #VIpar = ergmVI_betterinit(myergm,priormean, priorsig, actualpar)
    VIpar = ergmVI(myergm,priormean, priorsig)
    vimu = VIpar[[1]]
    viSig = VIpar[[3]]
  }, error=function(e){return(NA)})
  if (any(is.na(vimu))) {return(rep(NA,p^2))}
  else {
    print(VIpar)
    phatmarg <- numeric()
    phatcond <- numeric()
    for (i in 1:p){
      phatmarg = c(phatmarg, pnorm(actualpar[i], mean = vimu[i], sd = sqrt(viSig[i,i])))
      for (j in 1:p){
        if (j != i){
          #theta_i given theta_j
          condmean = vimu[i] + (actualpar[j] - vimu[j]) * viSig[i,j]/viSig[j,j]
          condvar = viSig[i,i] - viSig[i,j]^2/viSig[j,j]
          phatcond = c(phatcond, pnorm(actualpar[i], mean = condmean, sd = sqrt(condvar)))
          names(phatcond) = c(names(phatcond)[-length(phatcond)], paste0(i, "|",j))
        }
      }
    }
    names(phatmarg) = as.character(1:p)
    phattt = c(phatmarg, phatcond)
    phattt = phattt*0.9998 + 1e-5
    return(phattt)
  }
}



library(foreach)
library(doParallel)
cl = makeCluster(12)
registerDoParallel(cl)
getDoParWorkers()

tktk = Sys.time()

ergmVIphatnew = foreach(i = 1:NN, .combine = rbind, .export = c(c("bergmVIgetphat", "ergmdistorttrain")), .packages = c("ergm", "Bergm", "abc", "ks", "matrixcalc", "hdrcde")) %dopar% bergmVIgetphat(ergmdistorttrain[[1]][i,],ergmdistorttrain[[3]][[i]], priormean, priorsig)

stopCluster(cl)
print(Sys.time() - tktk)





naVIindxnew = which(is.na(ergmVIphatnew[,1]))
ergmVItrainnew = cbind(ergmVIphatnew, ergmdistorttrain[[2]])

ergmVItrainnew = ergmVItrainnew[-naVIindxnew,]

save(ergmVItrainnew,file = "ergmVItrainnew.RData")
