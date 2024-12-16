################################################### R Code for Real Data Application of spatial latent model###########################################################m
# Load the LaplacesDemon library
library("LaplacesDemon")
# Set seed for reproducibility
set.seed(12345)
# Load the data
data <- read.csv('D:\\Myfiles\\Mythesisresearch\\AnemiaData.csv')
# Attach the data
attach(data)
# Define covariates for each region
X_tig <- as.matrix(subset(data[,2:5], Regions == 1))
X_afa <- as.matrix(subset(data[,2:5], Regions == 2))
X_amh <- as.matrix(subset(data[,2:5], Regions == 3))
X_oro <- as.matrix(subset(data[,2:5], Regions == 4))
X_som <- as.matrix(subset(data[,2:5], Regions == 5))
X_ben <- as.matrix(subset(data[,2:5], Regions == 6))
X_snn <- as.matrix(subset(data[,2:5], Regions == 7))
X_gam <- as.matrix(subset(data[,2:5], Regions == 8))
X_har <- as.matrix(subset(data[,2:5], Regions == 9))
X_add <- as.matrix(subset(data[,2:5], Regions == 10))
X_dir <- as.matrix(subset(data[,2:5], Regions == 11))
# Rescale the covariates
X_tig <- scale(X_tig, center = TRUE, scale = TRUE)
X_afa <- scale(X_afa, center = TRUE, scale = TRUE)
X_amh <- scale(X_amh, center = TRUE, scale = TRUE)
X_oro <- scale(X_oro, center = TRUE, scale = TRUE)
X_som <- scale(X_som, center = TRUE, scale = TRUE)
X_ben <- scale(X_ben, center = TRUE, scale = TRUE)
X_snn <- scale(X_snn, center = TRUE, scale = TRUE)
X_gam <- scale(X_gam, center = TRUE, scale = TRUE)
X_har <- scale(X_har, center = TRUE, scale = TRUE)
X_add <- scale(X_add, center = TRUE, scale = TRUE)
X_dir <- scale(X_dir, center = TRUE, scale = TRUE)
# Set row names for each covariate matrix
rownames(X_tig) <- 1:nrow(X_tig)
rownames(X_afa) <- 1:nrow(X_afa)
rownames(X_amh) <- 1:nrow(X_amh)
rownames(X_oro) <- 1:nrow(X_oro)
rownames(X_som) <- 1:nrow(X_som)
rownames(X_ben) <- 1:nrow(X_ben)
rownames(X_snn) <- 1:nrow(X_snn)
rownames(X_gam) <- 1:nrow(X_gam)
rownames(X_har) <- 1:nrow(X_har)
rownames(X_add) <- 1:nrow(X_add)
rownames(X_dir) <- 1:nrow(X_dir)
# Combine covariates into a list
X <- list(X_tig, X_afa, X_amh, X_oro, X_som, X_ben, X_snn, X_gam, X_har, X_add, X_dir)
# Extract response variables for each region
y_tig <- matrix(subset(data[,1], Regions == 1), nrow(X_tig), 1)
y_afa <- matrix(subset(data[,1], Regions == 2), nrow(X_afa), 1)
y_amh <- matrix(subset(data[,1], Regions == 3), nrow(X_amh), 1)
y_oro <- matrix(subset(data[,1], Regions == 4), nrow(X_oro), 1)
y_som <- matrix(subset(data[,1], Regions == 5), nrow(X_som), 1)
y_ben <- matrix(subset(data[,1], Regions == 6), nrow(X_ben), 1)
y_snn <- matrix(subset(data[,1], Regions == 7), nrow(X_snn), 1)
y_gam <- matrix(subset(data[,1], Regions == 8), nrow(X_gam), 1)
y_har <- matrix(subset(data[,1], Regions == 9), nrow(X_har), 1)
y_add <- matrix(subset(data[,1], Regions == 10), nrow(X_add), 1)
y_dir <- matrix(subset(data[,1], Regions == 11), nrow(X_dir), 1)
# Combine response variables into a list
Y <- list(y_tig, y_afa, y_amh, y_oro, y_som, y_ben, y_snn, y_gam, y_har, y_add, y_dir)
# Initialize values for z
z_tig <- matrix(rep(5, nrow(X_tig)), nrow(X_tig), 1)
z_afa <- matrix(rep(5, nrow(X_afa)), nrow(X_afa), 1)
z_amh <- matrix(rep(5, nrow(X_amh)), nrow(X_amh), 1)
z_oro <- matrix(rep(5, nrow(X_oro)), nrow(X_oro), 1)
z_som <- matrix(rep(5, nrow(X_som)), nrow(X_som), 1)
z_ben <- matrix(rep(5, nrow(X_ben)), nrow(X_ben), 1)
z_snn <- matrix(rep(5, nrow(X_snn)), nrow(X_snn), 1)
z_gam <- matrix(rep(5, nrow(X_gam)), nrow(X_gam), 1)
z_har <- matrix(rep(5, nrow(X_har)), nrow(X_har), 1)
z_add <- matrix(rep(5, nrow(X_add)), nrow(X_add), 1)
z_dir <- matrix(rep(5, nrow(X_dir)), nrow(X_dir), 1)
# Combine z matrices into a list
z <- list(z_tig, z_afa, z_amh, z_oro, z_som, z_ben, z_snn, z_gam, z_har, z_add, z_dir)
# Number of regions
m <- 11
# Number of observations for each region
n_obs <- matrix(c(
  length(z_tig), length(z_afa), length(z_amh), length(z_oro), length(z_som),
  length(z_ben), length(z_snn), length(z_gam), length(z_har), length(z_add), length(z_dir)
), m, 1)
# Prior hyperparameters
lamda2_prior <- 5
psi2_prior <- 10
omega_upsilon_prior <- 10
eta_gamma_prior <- 15
n_r <- 3
r_spatial <- c(0.2, 0.4, 0.1)
mean_spatial <- sum(r_spatial) / n_r
r_spatial_sum <- 0.7
# Initial values for BETA
beta <- matrix(c(-1.2, -0.9, 0.8, 0.9))
BETA <- matrix(NA, ncol(X_tig), 1)
# Required packages
library("geoR")
library("mvtnorm")
library(MCMCpack)
library("msm")
# Number of covariates
p <- ncol(X[[1]])
# Initialize variables for posterior
omega_upsilon_posterior <- NULL  # Posterior degrees of freedom of variance of random effect
lamda2_posterior <- NULL  # Posterior scale parameter of sigma of variance of random effect
eta_gamma_posterior <- NULL  # Posterior degrees of freedom of variance of spatial effect
psi2_posterior <- NULL  # Posterior scale parameter of variance of spatial effect
mean_upsilon_posterior <- var_upsilon_posterior <- NULL  # Post values of mean and variance of random effects
mean_gamma_posterior <- var_gamma_posterior <- NULL  # Post values of mean and variance of spatial random effects
# Number of iterations
n_iter <- 20000
# Initialize variables for posterior
BETA_POST <- gamma_POST <- upsilon_POST <- tausq_gamma_POST <- sigmasq_upsilon_POST <- NULL
upsilon_posterior <- rep(0.1, m)
gamma_posterior <- rep(0.1, m)
sigmasq_upsilon_post <- NULL  # Posterior value of sigmasq of upsilon
tausq_gamma_post <- NULL  # Posterior value of tausq of gamma
# Additional initialization
z_all <- NULL
# Gibbs Sampler
for (i in 1:n_iter) {
  for (j in 1:m) {
    omega_upsilon_posterior[j] = (omega_upsilon_prior + 1) # updating degrees of freedom of variance of latent trait
    lamda2_posterior[j] = (lamda2_prior + ((upsilon_posterior[j])^2) / omega_upsilon_posterior[j]) # update scale parameter of variance of latent trait
    eta_gamma_posterior[j] = (eta_gamma_prior + 1) # updating degrees of freedom of variance of spatial effect
    psi2_posterior[j] = (psi2_prior + (n_r * ((gamma_posterior[j] - mean_spatial)^2)) / eta_gamma_posterior[j]) # update scale parameter of variance of spatial effect
    sigmasq_upsilon_post[j] = rinvchisq(1, omega_upsilon_posterior[j], lamda2_posterior[j]) # variance of random effects
    tausq_gamma_post[j] = rinvchisq(1, eta_gamma_posterior[j], psi2_posterior[j]) # variance of spatial effects
  }
  # Given variance I can sample from upsilon (non-spatial and spatial latent traits)
  mean_upsilon_posterior = list()
  mean_gamma_posterior = list() 
  mean_upsilon_posterior[[1]]=matrix(NA,length(z_tig),1)
  mean_upsilon_posterior[[2]]=matrix(NA,length(z_afa),1)
  mean_upsilon_posterior[[3]]=matrix(NA,length(z_amh),1)
  mean_upsilon_posterior[[4]]=matrix(NA,length(z_oro),1)
  mean_upsilon_posterior[[5]]=matrix(NA,length(z_som),1)
  mean_upsilon_posterior[[6]]=matrix(NA,length(z_ben),1)
  mean_upsilon_posterior[[7]]=matrix(NA,length(z_snn),1)
  mean_upsilon_posterior[[8]]=matrix(NA,length(z_gam),1)
  mean_upsilon_posterior[[9]]=matrix(NA,length(z_har),1)
  mean_upsilon_posterior[[10]]=matrix(NA,length(z_add),1)
  mean_upsilon_posterior[[11]]=matrix(NA,length(z_dir),1)
  ##########
  mean_gamma_posterior[[1]]=matrix(NA,length(z_tig),1)
  mean_gamma_posterior[[2]]=matrix(NA,length(z_afa),1)
  mean_gamma_posterior[[3]]=matrix(NA,length(z_amh),1)
  mean_gamma_posterior[[4]]=matrix(NA,length(z_oro),1)
  mean_gamma_posterior[[5]]=matrix(NA,length(z_som),1)
  mean_gamma_posterior[[6]]=matrix(NA,length(z_ben),1)
  mean_gamma_posterior[[7]]=matrix(NA,length(z_snn),1)
  mean_gamma_posterior[[8]]=matrix(NA,length(z_gam),1)
  mean_gamma_posterior[[9]]=matrix(NA,length(z_har),1)
  mean_gamma_posterior[[10]]=matrix(NA,length(z_add),1)
  mean_gamma_posterior[[11]]=matrix(NA,length(z_dir),1)
  # Calculate mean upsilon and gamma for each region
  for(t in 1:length(z_tig)){
    mean_upsilon_posterior[[1]][t,]=(z[[1]][t,]-X[[1]][t,]%*%beta)
    mean_gamma_posterior[[1]][t,]=(z[[1]][t,]-X[[1]][t,]%*%beta)
  }
  for(t in 1:length(z_afa)){
    mean_upsilon_posterior[[2]][t,]=(z[[2]][t,]-X[[2]][t,]%*%beta)
    mean_gamma_posterior[[2]][t,]=(z[[2]][t,]-X[[2]][t,]%*%beta)
  }
  for(t in 1:length(z_amh)){
    mean_upsilon_posterior[[3]][t,]=(z[[3]][t,]-X[[3]][t,]%*%beta)
    mean_gamma_posterior[[3]][t,]=(z[[3]][t,]-X[[3]][t,]%*%beta)
  }
  for(t in 1:length(z_oro)){
    mean_upsilon_posterior[[4]][t,]=(z[[4]][t,]-X[[4]][t,]%*%beta)
    mean_gamma_posterior[[4]][t,]=(z[[4]][t,]-X[[4]][t,]%*%beta)
  }
  for(t in 1:length(z_som)){
    mean_upsilon_posterior[[5]][t,]=(z[[5]][t,]-X[[5]][t,]%*%beta)
    mean_gamma_posterior[[5]][t,]=(z[[5]][t,]-X[[5]][t,]%*%beta)
  }
  for(t in 1:length(z_ben)){
    mean_upsilon_posterior[[6]][t,]=(z[[6]][t,]-X[[6]][t,]%*%beta)
    mean_gamma_posterior[[6]][t,]=(z[[6]][t,]-X[[6]][t,]%*%beta)
  }
  for(t in 1:length(z_snn)){
    mean_upsilon_posterior[[7]][t,]=(z[[7]][t,]-X[[7]][t,]%*%beta)
    mean_gamma_posterior[[7]][t,]=(z[[7]][t,]-X[[7]][t,]%*%beta)
  }
  for(t in 1:length(z_gam)){
    mean_upsilon_posterior[[8]][t,]=(z[[8]][t,]-X[[8]][t,]%*%beta)
    mean_gamma_posterior[[8]][t,]=(z[[8]][t,]-X[[8]][t,]%*%beta)
  }
  for(t in 1:length(z_har)){
    mean_upsilon_posterior[[9]][t,]=(z[[9]][t,]-X[[9]][t,]%*%beta)
    mean_gamma_posterior[[9]][t,]=(z[[9]][t,]-X[[9]][t,]%*%beta)
  }
  for(t in 1:length(z_add)){
    mean_upsilon_posterior[[10]][t,]=(z[[10]][t,]-X[[10]][t,]%*%beta)
    mean_gamma_posterior[[10]][t,]=(z[[10]][t,]-X[[10]][t,]%*%beta)
  }
  for(t in 1:length(z_dir)){
    mean_upsilon_posterior[[11]][t,]=(z[[11]][t,]-X[[11]][t,]%*%beta)
    mean_gamma_posterior[[11]][t,]=(z[[11]][t,]-X[[11]][t,]%*%beta)
  }
  # Calculate variances
  for (j in 1:m) {
    var_upsilon_posterior[j] = 1 / (n_obs[j, ] + (1 / sigmasq_upsilon_post[j]))
    var_gamma_posterior[j] = 1 / (n_obs + (n_obs / tausq_gamma_post[j]))
  }
  
  # Summing over time
  mean_upsilon_posterior_sum = c()
  mean_gamma_posterior_sum = c()
  
  for (matrix in mean_upsilon_posterior) {
    matrix_sum = sum(matrix)
    mean_upsilon_posterior_sum = c(mean_upsilon_posterior_sum, matrix_sum)
  }
  
  for (matrix in mean_gamma_posterior) {
    matrix_sum =sum(matrix)
    mean_gamma_posterior_sum =c(mean_gamma_posterior_sum, matrix_sum)
  }
  
  # Updating latent traits
  for (j in 1:m) {
    mean_upsilon_posterior_sum[j] = (mean_upsilon_posterior_sum[j] - n_obs[j, ] * gamma_posterior[j]) / (n_obs + 1 / sigmasq_upsilon_post[j])
    upsilon_posterior[j] = rnorm(1, mean_upsilon_posterior_sum[j], sqrt(var_upsilon_posterior[j]))
    mean_gamma_posterior_sum[j] = (r_spatial_sum + tausq_gamma_post[j] * mean_gamma_posterior_sum[j] - n_obs * upsilon_posterior[j]) / (n_obs * tausq_gamma_post[j] + n_obs)
    gamma_posterior[j] = rnorm(1, mean_gamma_posterior_sum[j], sqrt(var_gamma_posterior[j]))
  }
  # As I have only one latent trait upsilon for each region
  upsilon_rep = list()
  gamma_rep = list()
  for (j in 1:m) {
    upsilon_rep[[j]] = rep(upsilon_posterior[j], n_obs[j, ])
    gamma_rep[[j]] = rep(gamma_posterior[j], n_obs[j, ])
  }
  # Convert the list into matrix
  upsilon_rep = lapply(upsilon_rep, function(x) matrix(x, nrow = length(x), ncol = 1))
  gamma_rep = lapply(gamma_rep, function(x) matrix(x, nrow = length(x), ncol = 1))
  a=b=list()
  a[[1]]=matrix(NA,length(z_tig),1)
  a[[2]]=matrix(NA,length(z_afa),1)
  a[[3]]=matrix(NA,length(z_amh),1)
  a[[4]]=matrix(NA,length(z_oro),1)
  a[[5]]=matrix(NA,length(z_som),1)
  a[[6]]=matrix(NA,length(z_ben),1)
  a[[7]]=matrix(NA,length(z_snn),1)
  a[[8]]=matrix(NA,length(z_gam),1)
  a[[9]]=matrix(NA,length(z_har),1)
  a[[10]]=matrix(NA,length(z_add),1)
  a[[11]]=matrix(NA,length(z_dir),1)
  b[[1]]=matrix(NA,length(z_tig),1)
  b[[2]]=matrix(NA,length(z_afa),1)
  b[[3]]=matrix(NA,length(z_amh),1)
  b[[4]]=matrix(NA,length(z_oro),1)
  b[[5]]=matrix(NA,length(z_som),1)
  b[[6]]=matrix(NA,length(z_ben),1)
  b[[7]]=matrix(NA,length(z_snn),1)
  b[[8]]=matrix(NA,length(z_gam),1)
  b[[9]]=matrix(NA,length(z_har),1)
  b[[10]]=matrix(NA,length(z_add),1)
  b[[11]]=matrix(NA,length(z_dir),1)
  for(t in 1:length(z_tig)){
    zt=z[[1]]
    a[[1]][t,]=max(zt[Y[[1]][,1]<Y[[1]][t,]])
    b[[1]][t,]=min(zt[Y[[1]][t,]<Y[[1]][,1]])
    z[[1]][t,]=rtnorm(1,X[[1]][t,]%*%beta+upsilon_rep[[1]][t,]+gamma_rep[[1]][t,],1,a[[1]][t,],b[[1]][t,])
  }
  for(t in 1:length(z_afa)){
    zt=z[[2]]
    a[[2]][t,]=max(zt[Y[[2]][,1]<Y[[2]][t,]])
    b[[2]][t,]=min(zt[Y[[2]][t,]<Y[[2]][,1]])
    z[[2]][t,]=rtnorm(1,X[[2]][t,]%*%beta+upsilon_rep[[2]][t,]+gamma_rep[[2]][t,],1,a[[2]][t,],b[[2]][t,])
  }
  for(t in 1:length(z_amh)){
    zt=z[[3]]
    a[[3]][t,]=max(zt[Y[[3]][,1]<Y[[3]][t,]])
    b[[3]][t,]=min(zt[Y[[3]][t,]<Y[[3]][,1]])
    z[[3]][t,]=rtnorm(1,X[[3]][t,]%*%beta+upsilon_rep[[3]][t,]+gamma_rep[[3]][t,],1,a[[3]][t,],b[[3]][t,])
  }
  for(t in 1:length(z_oro)){
    zt=z[[4]]
    a[[4]][t,]=max(zt[Y[[4]][,1]<Y[[4]][t,]])
    b[[4]][t,]=min(zt[Y[[4]][t,]<Y[[4]][,1]])
    z[[4]][t,]=rtnorm(1,X[[4]][t,]%*%beta+upsilon_rep[[4]][t,]+gamma_rep[[4]][t,],1,a[[4]][t,],b[[4]][t,])
  } 
  for(t in 1:length(z_som)){
    zt=z[[5]]
    a[[5]][t,]=max(zt[Y[[5]][,1]<Y[[5]][t,]])
    b[[5]][t,]=min(zt[Y[[5]][t,]<Y[[5]][,1]])
    z[[5]][t,]=rtnorm(1,X[[5]][t,]%*%beta+upsilon_rep[[5]][t,]+gamma_rep[[5]][t,],1,a[[5]][t,],b[[5]][t,])
  }
  for(t in 1:length(z_ben)){
    zt=z[[6]]
    a[[6]][t,]=max(zt[Y[[6]][,1]<Y[[6]][t,]])
    b[[6]][t,]=min(zt[Y[[6]][t,]<Y[[6]][,1]])
    z[[6]][t,]=rtnorm(1,X[[6]][t,]%*%beta+upsilon_rep[[6]][t,]+gamma_rep[[6]][t,],1,a[[6]][t,],b[[6]][t,])
  }
  for(t in 1:length(z_snn)){
    zt=z[[7]]
    a[[7]][t,]=max(zt[Y[[7]][,1]<Y[[7]][t,]])
    b[[7]][t,]=min(zt[Y[[7]][t,]<Y[[7]][,1]])
    z[[7]][t,]=rtnorm(1,X[[7]][t,]%*%beta+upsilon_rep[[7]][t,]+gamma_rep[[7]][t,],1,a[[7]][t,],b[[7]][t,])
  }
  for(t in 1:length(z_gam)){
    zt=z[[8]]
    a[[8]][t,]=max(zt[Y[[8]][,1]<Y[[8]][t,]])
    b[[8]][t,]=min(zt[Y[[8]][t,]<Y[[8]][,1]])
    z[[8]][t,]=rtnorm(1,X[[8]][t,]%*%beta+upsilon_rep[[8]][t,]+gamma_rep[[8]][t,],1,a[[8]][t,],b[[8]][t,])
  }
  for(t in 1:length(z_har)){
    zt=z[[9]]
    a[[9]][t,]=max(zt[Y[[9]][,1]<Y[[9]][t,]])
    b[[9]][t,]=min(zt[Y[[9]][t,]<Y[[9]][,1]])
    z[[9]][t,]=rtnorm(1,X[[9]][t,]%*%beta+upsilon_rep[[9]][t,]+gamma_rep[[9]][t,],1,a[[9]][t,],b[[9]][t,])
    
  }
  for(t in 1:length(z_add)){
    zt=z[[10]]
    a[[10]][t,]=max(zt[Y[[10]][,1]<Y[[10]][t,]])
    b[[10]][t,]=min(zt[Y[[10]][t,]<Y[[10]][,1]])
    z[[10]][t,]=rtnorm(1,X[[10]][t,]%*%beta+upsilon_rep[[10]][t,]+gamma_rep[[10]][t,],1,a[[10]][t,],b[[10]][t,])
  }
  for(t in 1:length(z_dir)){
    zt=z[[11]]
    a[[11]][t,]=max(zt[Y[[11]][,1]<Y[[11]][t,]])
    b[[11]][t,]=min(zt[Y[[11]][t,]<Y[[11]][,1]])
    z[[11]][t,]=rtnorm(1,X[[11]][t,]%*%beta+upsilon_rep[[11]][t,]+gamma_rep[[11]][t,],1,a[[11]][t,],b[[11]][t,])
  }
  
  muj1 = array(NA, c(p, p, m))
  muj1_sum = matrix(NA, p, p)
  muj2 = array(NA, c(p, 1, m))
  muj2_sum = matrix(NA, p, 1)
  muj_final = matrix(NA, p, 1)
  varj_final = matrix(NA, p, p)
  for (j in 1:m) {
    muj1[, , j] = t(X[[j]]) %*% X[[j]]
    muj1_sum = apply(muj1, c(1:2), sum) # summing over j
    muj2[, , j] = t(X[[j]]) %*% (z[[j]] - upsilon_rep[[j]] - gamma_rep[[j]])
    muj2_sum = apply(muj2, c(1:2), sum) # summing over j
    muj_final = solve(muj1_sum) %*% (muj2_sum)
    varj_final = solve(muj1_sum)
  }
  BETA = mvrnorm(n = 1, mu = muj_final, varj_final) # sampling from a multivariate Normal with updated mean and variance
  BETA_POST = rbind(BETA_POST, BETA)
  upsilon_POST = rbind(upsilon_POST, upsilon_posterior)
  gamma_POST = rbind(gamma_POST, gamma_posterior)
  sigmasq_upsilon_POST = rbind(sigmasq_upsilon_POST, sigmasq_upsilon_post)
  tausq_gamma_POST = rbind(tausq_gamma_POST, tausq_gamma_post)
  print(paste("iteration", i))
  z_all = rbind(z_all, z)
}
# Checking the convergence with burn
library(coda)
library(bayesplot)
library(stableGR)
Beta_gibbis_burn <- BETA_POST[seq(1, nrow(BETA_POST[5001:20000, ]), 5), ]
upsilon_gibbis_burn <- upsilon_POST[seq(1, nrow(upsilon_POST[5001:20000, ]), 5), ]
gamma_gibbis_burn <- gamma_POST[seq(1, nrow(gamma_POST[5001:20000, ]), 5), ]
# Trace plots of Beta
par(mfrow = c(4, 1), mar = c(3, 3, 2, 2))
traceplot(as.mcmc(Beta_gibbis_burn[, 1]), col = "purple", main = expression(paste(beta[1])))
traceplot(as.mcmc(Beta_gibbis_burn[, 2]), col = "purple", main = expression(paste(beta[2])))
traceplot(as.mcmc(Beta_gibbis_burn[, 3]), col = "purple", main = expression(paste(beta[3])))
traceplot(as.mcmc(Beta_gibbis_burn[, 4]), col = "purple", main = expression(paste(beta[4])))
# Density plots of Beta
par(mfrow = c(2, 2), mar = c(7, 3, 3, 1))
densplot(as.mcmc(Beta_gibbis_burn[, 1]), col = "purple", type = "h", main = expression(paste(beta[1])))
densplot(as.mcmc(Beta_gibbis_burn[, 2]), col = "purple", type = "h", main = expression(paste(beta[2])))
densplot(as.mcmc(Beta_gibbis_burn[, 3]), col = "purple", type = "h", main = expression(paste(beta[3])))
densplot(as.mcmc(Beta_gibbis_burn[, 4]), col = "purple", type = "h", main = expression(paste(beta[4])))
# Autocorrelation plots of Beta
par(mfrow = c(2, 2), mar = c(7, 4, 3, 1))
acf(as.mcmc(Beta_gibbis_burn[, 1]), col = "purple", lwd = 2, ylab = "Autocorrelation", ci = FALSE, lag.max = 3000, main = expression(paste(beta[1])))
acf(as.mcmc(Beta_gibbis_burn[, 2]), col = "purple", lwd = 2, ylab = "Autocorrelation", ci = FALSE, lag.max = 3000, main = expression(paste(beta[2])))
acf(as.mcmc(Beta_gibbis_burn[, 3]), col = "purple", lwd = 2, ylab = "Autocorrelation", ci = FALSE, lag.max = 3000, main = expression(paste(beta[3])))
acf(as.mcmc(Beta_gibbis_burn[, 4]), col = "purple", lwd = 2, ylab = "Autocorrelation", ci = FALSE, lag.max = 3000, main = expression(paste(beta[4])))
# posterior Mean of Beta
a <- mean(Beta_gibbis_burn[, 1])
b <- mean(Beta_gibbis_burn[, 2])
c <- mean(Beta_gibbis_burn[, 3])
d <- mean(Beta_gibbis_burn[, 4])
# posterior Mean of Beta
a <- sd(Beta_gibbis_burn[, 1])
b <- sd(Beta_gibbis_burn[, 2])
c <- sd(Beta_gibbis_burn[, 3])
d <- sd(Beta_gibbis_burn[, 4])
# Calculate HPD credible intervals of Beta
Beta_mcmc_object <- as.mcmc(Beta_gibbis_burn)
Beta_credible_interval <- HPDinterval(Beta_mcmc_object, prob = 0.95)
# PSRF test of Beta
PSRF_beta_test <- stable.GR(Beta_gibbis_burn)
# Load necessary libraries
library(reshape2)
library(viridis)
library(ggplot2)
# Load and process the first dataset
new_colnames <- c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
colnames(g) <- new_colnames
g <- data.frame(upsilon_gibbis_burn)
data <- var(g[sapply(g, is.numeric)])
data1 <- melt(data)
# Set region labels
region_labels <- c("Tigray", "Afar", "Amhara", "Oromia", "Somale", "Benishangul", "SNN", "Gambela", "Harari", "Addis Abeba", "Dire Dawa")
# Rename columns for data1
colnames(data1) <- c("Var1", "Var2", "Variations")
# Plot variations
ggplot(data1, aes(x = Var1, y = Var2, fill = Variations)) +
  geom_tile() +
  scale_fill_viridis(discrete = FALSE) +
  labs(
    title = "Regional Variations",
    x = "Regions",
    y = "Regions"
  ) +
  scale_x_discrete(labels = region_labels) +
  scale_y_discrete(labels = region_labels)
# Save processed data to CSV
write.csv(data1, file = "D:\\Myfiles\\Mythesisresearch\\Variation.csv", row.names = FALSE)
# Load and process the second dataset
colnames(g) <- new_colnames
g <- data.frame(gamma_gibbis_burn)
dataa <- cor(g[sapply(g, is.numeric)])
mydata <- melt(dataa)
# Save mydata to CSV
write.csv(mydata, file = "D:\\Myfiles\\Mythesisresearch\\mydata1.csv", row.names = FALSE)
# Load the saved data
data3 <- read.csv('D:\\Myfiles\\Mythesisresearch\\mydata1.csv')
colnames(data3) <- c("Var1", "Var2", "Variations")
# Plot spatial correlations
ggplot(data3, aes(x = Var1, y = Var2, fill = Variations)) +
  geom_tile() +
  scale_fill_viridis(discrete = FALSE) +
  labs(
    title = "Spatial Correlations",
    x = "Regions",
    y = "Regions",
    fill = "Spatial Correlation"
  ) +
  scale_x_discrete(labels = region_labels) +
  scale_y_discrete(labels = region_labels)