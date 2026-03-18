# Load library
library(glmnet) 
library(dplyr)
library(psych)
library(tseries)
library(quantmod)
library(zoo)
library(forecast)
library(ggplot2)
library(reshape2)
# Solving ODE system
library(deSolve) 
library(aTSA)
library(keras)
library(tensorflow)
# GARCH
library(rugarch) 
library(tibble)
# LSTM
library(keras)
library(tidyverse)
# install.packages("bayesreg")
library(bayesreg)
library(devtools)
library(rfBLT)

# --- SmuEIR model ---
sueir_model <- function(time, state, parameters) {
  with(as.list(c(state, parameters)), {
    dS <- -beta * (I + E) * S / N
    dE <- beta * (I + E) * S / N - (sigma * E)
    dI <- mu * sigma * E - gamma * I
    dR <- gamma * I
    return(list(c(dS, dE, dI, dR)))
  })
}

# Set seed
set.seed(456)

# --- Data Simulation ---
# Parameters
beta <- 3/14        # Infection rate
gamma <- 1/14       # Removal rate
sigma <- 0.25       # Latency rate
mu <- 0.75          # Discovery rate

# Initial conditions
S <- 10^6
E <- 0
I <- 1
R <- 0
N <- S + E + I + R # Total population        

initial_state <- c(S = S, E = E, I = I, R = R) # Initial populations
parameters <- c(beta = beta, gamma = gamma, sigma = sigma, mu = mu) # Transmission and recovery rates
times <- seq(0, 180, by = 1) # Time sequence (0 to 180 days)

# Solve the system of ODEs
sueir_data <- ode(y = initial_state, times = times, func = sueir_model, parms = parameters)

# Check the output
print(head(sueir_data))

# Convert to a data frame
sueir_df <- as.data.frame(sueir_data)

# Prepare to add noise to infectious
n_simulation = 100
I_noise_mat = matrix(rep(NA, 181*n_simulation), nrow=n_simulation, ncol=181, byrow=TRUE)

# Calculate proportion
S = sueir_df$S/N
E = sueir_df$E/N
I = sueir_df$I/N
R = sueir_df$R/N

# Add noise into simulation
for(i in (1:n_simulation)){
    noise = rnorm(nrow(sueir_df), mean = 0 , sd = 0.1)
    I_noise_mat[i,] = I + noise*max(abs(I))
}

# Compute 7-day right moving average
I_noise_avg_mat = matrix(rep(NA, 181*n_simulation), nrow=n_simulation, ncol=181, byrow=TRUE)
for(i in (1:n_simulation)){
    I_noise_avg_mat[i,] = rollapply(I_noise_mat[i,], 7, mean, align="right", partial=TRUE)
}

# Visualize data
plot(sueir_df$time, I, type = "l", col = "orange", ylim = c(-0.05, 0.25), 
     xlab = "Time (days)", ylab = "Infectious Proportion", lwd = 2, main = "Proportion of Infectious from S(mu)EIR Model")
for (i in (1:100)){
    lines(sueir_df$time, I_noise_avg_mat[i,], col = "brown", lwd = 2)  # I noise avg
}
abline(v=102, col="black", lty = 2)

# Specify input parameters
pred_size = 7
window_size = 2*4+1

# --- Random Feature Bayesian Lasso without Takens' theorem ---
model_performance_lasso <- function(df,
                                    m,
                                    pred_size,
                                    window_size,
                                    feature_selection = "sqrt",
                                    feature_constant = NULL) {
  
  # y true value vector
  y_true <- df[(m + 1):(m + pred_size)]
  
  ts_bayes <- ts_forecast_bayes_reg_rfm(ts_data = as.vector(df[1:m]),
                                        window_size = window_size,
                                        pred_size = pred_size,
                                        weight_dist = "normal",
                                        weight_params = list(),
                                        bias_dist = "uniform",
                                        bias_params = list(min_val = 0, max_val = 2*pi),
                                        act_func = "fourier",
                                        reg_type = "lasso",
                                        burnin = 1000,
                                        n.samples = 2000,
                                        thin = 5,
                                        n.cores = NULL,
                                        pred_type = "mean",
                                        feature_selection = feature_selection,
                                        feature_constant = feature_constant,
                                        CI = 95)
  
  lasso_bayes_reg_model = ts_bayes$fit_results
  lasso_bayes_fitted_values = ts_bayes$fitted.values
  lasso_posterior_samples = ts_bayes$posterior_samples
  lasso_bayes_future_preds = ts_bayes$future_preds  # matrix
  lasso_bayes_y_pred = ts_bayes$y_pred              # vector
  lasso_bayes.pred.ci = ts_bayes$pred.ci
  lasso_bayes.pred.lower.ci = lasso_bayes.pred.ci[,1]
  lasso_bayes.pred.upper.ci = lasso_bayes.pred.ci[,2]
  lasso_ess_plot = ts_bayes$ess  
  lasso_coef <- ts_bayes$posterior_samples
  lasso_sigma2 <- ts_bayes$posterior_sigma2
    
  error_df <- data.frame(
    bayes_lasso = vec_relative_error(matrix(lasso_bayes_y_pred, nrow = 1, byrow = TRUE), y_true)
  )
  return(list(error_df = error_df,
              y_true = y_true,
              bayes_lasso_fit = list(fit = lasso_bayes_reg_model,
                                     fitted_values = lasso_bayes_fitted_values,
                                     pred = lasso_bayes_y_pred,
                                     coef = lasso_coef,
                                     sigma2 = lasso_sigma2,
                                     lowerCI = lasso_bayes.pred.lower.ci,
                                     upperCI = lasso_bayes.pred.upper.ci,
                                     ess = lasso_ess_plot)))
}

# --- Create empty dfs to save results ---
# True value
y_true_mat <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)

# Bayesian Lasso
bayes_lasso_lowerCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
bayes_lasso_upperCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
bayes_lasso_y_pred <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)

# Bayesian Lasso Taken with Smoothness
bayes_lasso_taken_normal_lowerCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
bayes_lasso_taken_normal_upperCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE) 
bayes_lasso_taken_normal_y_pred <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE) 

# --- RF, rfBLT ---
fixed_point = 102

# --- RF ---
bayes_lasso_ess_list <- list()
bayes_lasso_coef_list <- list()
bayes_lasso_sigma2_list <- list()
bayes_lasso_residual_trend_train_mat <- matrix(NA, nrow=n_simulation, ncol=(fixed_point-window_size))
df_residuals_acf_rfBL <- matrix(NA, nrow=n_simulation, ncol=(fixed_point-window_size))

for (i in 1:n_simulation){
  print(i)
  data_vec = as.vector(I_noise_avg_mat[i,])
  # smoothed_vec = as.vector(fwd_diff_mat[i,])
  res <- model_performance_lasso(
    df = data_vec,
    # smoothed = smoothed_vec,
    m = fixed_point,
    pred_size = pred_size,
    window_size = window_size,
    feature_selection = "factor",
    feature_constant = 0.5
  )
  # True value
  y_true_mat[i,] = as.vector(res$y_true)

  # Model residuals
  train_mat <- embed(data_vec[1:(fixed_point)], (window_size+1))
  residuals <- train_mat[,1] - res$bayes_lasso_fit$fitted_values
  bayes_lasso_residual_trend_train_mat[i, 1:length(residuals)] <- residuals

  residual_acf <- acf(residuals, lag.max=fixed_point-window_size+1, plot=FALSE)
  df_residuals_acf_rfBL[i, ] <- as.vector(residual_acf$acf) 
  
  # Bayesian Lasso
  bayes_lasso_lowerCI[i,] = as.vector(res$bayes_lasso_fit$lowerCI)
  bayes_lasso_upperCI[i,] = as.vector(res$bayes_lasso_fit$upperCI)
  bayes_lasso_y_pred[i,] = as.vector(res$bayes_lasso_fit$pred)
    
  bayes_lasso_ess_list[[i]] <- res$bayes_lasso_fit$ess
  bayes_lasso_coef_list[[i]] <- res$bayes_lasso_fit$coef
  bayes_lasso_sigma2_list[[i]] <- res$bayes_lasso_fit$sigma2
}

# Save results for RF
## ESS plots
pdf("train102_RF_ess.pdf", width=7, height=5)
par(mfrow=c(1, 1))
for (i in 1:n_simulation){
    plot(bayes_lasso_ess_list[[i]])
}
dev.off()

## Residuals
pdf("train102_RF_residuals.pdf", width=10, height=4)
par(mfrow=c(1, 2))
for (i in 1:dim(df_residuals_acf_rfBL)[1]){
  # ACF of residuals
  plot(df_residuals_acf_rfBL[i, ], type="h", ylab="ACF", xlab="Lag", main=paste(i))
  # QQ plot of residuals
  qqnorm(na.omit(bayes_lasso_residual_trend_train_mat[i,]), main=paste(i))
  qqline(na.omit(bayes_lasso_residual_trend_train_mat[i,]))
}
dev.off()

## Coefficients of last iter
pdf("train102_RF_coefs_samples.pdf", width=15, height=20)
par(mfrow=c(6, 3))
for (i in 1:ncol(res$bayes_lasso_fit$coef)){
  plot(res$bayes_lasso_fit$coef[,i], type="l", main=paste0("Tract plot of Beta ", i-1), ylab=paste0("Beta ", i-1))
  hist(res$bayes_lasso_fit$coef[,i], probability = TRUE, main=paste0("Density of Beta ", i-1), xlab=paste0("Beta ", i-1))
  boxplot(res$bayes_lasso_fit$coef[,i], horizontal=TRUE, main=paste0("Box plot of Beta ", i-1))
}
dev.off()

## Sigma2 of last iter
pdf("train102_RF_sigma2_samples.pdf", width=15, height=4)
par(mfrow=c(1, 3))
plot(res$bayes_lasso_fit$sigma2, type="l", ylab="Sigma2", main="Trace plot of Sigma2")
hist(res$bayes_lasso_fit$sigma2, probability = TRUE, main=paste0("Density of Sigma2"), xlab=paste0("Sigma2"))
boxplot(res$bayes_lasso_fit$sigma2, horizontal=TRUE, main="Box plot of Sigma2")
dev.off()

# --- rfBLT ---
bayes_lasso_taken_normal_ess_list <- list()
bayes_lasso_taken_normal_coef_list <- list()
bayes_lasso_taken_normal_sigma2_list <- list()
bayes_lasso_taken_residual_trend_train_mat <- matrix(NA, nrow=n_simulation, ncol=(fixed_point-window_size))
df_residuals_acf_rfBLT <- matrix(NA, nrow=n_simulation, ncol=(fixed_point-window_size))

for (i in 1:n_simulation){
  print(i)
  data_vec = as.vector(I_noise_avg_mat[i,])
  input_vec = data_vec[1:fixed_point]
  # smoothed_vec = as.vector(smoothed_mat[i,])
  res <- ts_forecast_bayes_reg_rfm_taken(
      ts_data = input_vec,
      time = 1:fixed_point,
      smooth_diff = TRUE,
      method = "ma",
      smooth_params = list(window=10),
      window_size = window_size,
      pred_size = pred_size,
      weight_dist = "normal",
      weight_params = list(),
      bias_dist = "uniform",
      bias_params = list(min_val = 0, max_val = 2*pi),
      act_func = "fourier",
      reg_type = "lasso",
      burnin = 1000,
      n.samples = 2000,
      thin = 5,
      n.cores = NULL,
      pred_type = "mean",
      feature_selection = "factor",
      feature_constant = 0.5,
      CI = 95
    )

  # Residuals
  train_mat <- embed(data_vec[1:(fixed_point)], (window_size+1))
  residuals <- train_mat[,1] - res$fitted.values
  bayes_lasso_taken_residual_trend_train_mat[i, 1:length(residuals)] <- residuals

  residual_acf <- acf(residuals, lag.max=fixed_point-window_size+1, plot=FALSE)
  df_residuals_acf_rfBLT[i, ] <- as.vector(residual_acf$acf) 
  
  # Prediction  
  bayes_lasso_taken_normal_lowerCI[i,] = as.vector(res$pred.ci[,1])
  bayes_lasso_taken_normal_upperCI[i,] = as.vector(res$pred.ci[,2])
  bayes_lasso_taken_normal_y_pred[i,] = as.vector(res$y_pred)

  bayes_lasso_taken_normal_ess_list[[i]] <- res$ess
  bayes_lasso_taken_normal_coef_list[[i]] <- res$posterior_samples
  bayes_lasso_taken_normal_sigma2_list[[i]] <- res$posterior_sigma2
}

# Save results for rfBLT
## ESS plots
pdf("train102_rfBLT_ess.pdf", width=7, height=5)
par(mfrow=c(1, 1))
for (i in 1:n_simulation){
    plot(bayes_lasso_taken_normal_ess_list[[i]])
}
dev.off()

## Residuals
pdf("train102_rfBLT_residuals.pdf", width=10, height=4)
par(mfrow=c(1, 2))
for (i in 1:dim(df_residuals_acf_rfBLT)[1]){
  # ACF of residuals
  plot(df_residuals_acf_rfBLT[i, ], type="h", ylab="ACF", xlab="Lag", main=paste(i))
  # QQ plot of residuals
  qqnorm(na.omit(bayes_lasso_taken_residual_trend_train_mat[i,]), main=paste(i))
  qqline(na.omit(bayes_lasso_taken_residual_trend_train_mat[i,]))
}
dev.off()

## Coefficients of last iter
pdf("train102_rfBLT_coefs_samples.pdf", width=15, height=20)
par(mfrow=c(6, 3))
for (i in 1:ncol(res$posterior_samples)){
  plot(res$posterior_samples[,i], type="l", main=paste0("Tract plot of Beta ", i-1), ylab=paste0("Beta ", i-1))
  hist(res$posterior_samples[,i], probability = TRUE, main=paste0("Density of Beta ", i-1), xlab=paste0("Beta ", i-1))
  boxplot(res$posterior_samples[,i], horizontal=TRUE, main=paste0("Box plot of Beta ", i-1))
}
dev.off()

## Sigma2 of last iter
pdf("train102_rfBLT_sigma2_samples.pdf", width=15, height=4)
par(mfrow=c(1, 3))
plot(res$posterior_sigma2, type="l", ylab="Sigma2", main="Trace plot of Sigma2")
hist(res$posterior_sigma2, probability = TRUE, main=paste0("Density of Sigma2"), xlab=paste0("Sigma2"))
boxplot(res$posterior_sigma2, horizontal=TRUE, main="Box plot of Sigma2")
dev.off()

# True matrix
y_true_mat = I_noise_avg_mat[,(fixed_point+1):(fixed_point+pred_size)]

# Bayes Lasso
m102_pred_bayes_lasso = bayes_lasso_y_pred
m102_lowerCI_bayes_lasso = bayes_lasso_lowerCI
m102_upperCI_bayes_lasso = bayes_lasso_upperCI

# Bayes Lasso Takens with Smoothing Derivatives
m102_pred_BLT_normal_error <- bayes_lasso_taken_normal_y_pred
m102_lowerCI_normal_error <- bayes_lasso_taken_normal_lowerCI
m102_upperCI_normal_error <- bayes_lasso_taken_normal_upperCI

# --- ARIMA ---
m102_arima_pred_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m102_arima_lowerCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m102_arima_upperCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m102_arima_avg_noise_aic <- numeric(n_simulation)
m102_arima_p <- numeric(n_simulation)
m102_arima_d <- numeric(n_simulation)
m102_arima_q <- numeric(n_simulation)

# I noise m = 102
for (i in 1:n_simulation){
  arima <- arima_model(I_noise_avg_mat[i,], m=fixed_point, pred_size=pred_size)
  m102_arima_pred_avg_noise[i,] <- arima$pred
  m102_arima_lowerCI_avg_noise[i,] <- arima$lowerCI
  m102_arima_upperCI_avg_noise[i,] <- arima$upperCI
  m102_arima_avg_noise_aic[i] <- arima$aic
  m102_arima_p[i] <- arima$model$arma[1]
  m102_arima_q[i] <- arima$model$arma[2]
  m102_arima_d[i] <- arima$model$arma[6]
}

# --- Holt Exponential Smoothing ---
m102_holt_pred_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m102_holt_lowerCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m102_holt_upperCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m102_holt_avg_noise_aic <- numeric(n_simulation)

# I noise m = 102
for (i in 1:n_simulation){
  data_vec <- ts(I_noise_avg_mat[i,1:fixed_point], frequency=7)
  # For Holt's method (trend, no seasonality)
  fit_holt <- holt(data_vec, h=7)
  
  m102_holt_pred_avg_noise[i,] <- as.vector(fit_holt$mean)
  m102_holt_lowerCI_avg_noise[i,] <- as.vector(fit_holt$lower[,2])
  m102_holt_upperCI_avg_noise[i,] <- as.vector(fit_holt$upper[,2])
  m102_holt_avg_noise_aic[i] <- AIC(fit_holt$model)
}

# Save data for analyzing
save(I_noise_avg_mat,
     y_true_mat,

     # ARIMA
     m102_arima_pred_avg_noise,
     m102_arima_lowerCI_avg_noise,
     m102_arima_upperCI_avg_noise,
     m102_arima_avg_noise_aic,
     
     # Holt
     m102_holt_pred_avg_noise,
     m102_holt_lowerCI_avg_noise,
     m102_holt_upperCI_avg_noise,
     m102_holt_avg_noise_aic,
     
     # rfBL
     m102_pred_bayes_lasso,
     m102_lowerCI_bayes_lasso,
     m102_upperCI_bayes_lasso,

     # rfBLT with smoothness
     m102_pred_BLT_normal_error,
     m102_lowerCI_normal_error,
     m102_upperCI_normal_error,
    
     file = "simulation0.1_m102_July17.RData")
