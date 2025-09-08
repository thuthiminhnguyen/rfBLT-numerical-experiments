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

# Observe the output
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
abline(v=85, col="black", lty = 2)

# Specify input parameters
pred_size = 7
window_size = 2*4+1

# --- Model function ---
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
  lasso_posterior_samples = ts_bayes$posterior_samples
  lasso_bayes_future_preds = ts_bayes$future_preds  # matrix
  lasso_bayes_y_pred = ts_bayes$y_pred              # vector
  lasso_bayes.pred.ci = ts_bayes$pred.ci
  lasso_bayes.pred.lower.ci = lasso_bayes.pred.ci[,1]
  lasso_bayes.pred.upper.ci = lasso_bayes.pred.ci[,2]
  lasso_ess_plot = ts_bayes$ess_plot  
  
  ts_bayes_taken <- ts_forecast_bayes_reg_rfm_taken(
      ts_data = as.vector(df[1:m]),
      time = 1:m,
      smooth_diff = FALSE,
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
      CI = 95
    ) 
  
  # Model output
  lasso_taken_bayes_reg_model = ts_bayes_taken$fit_results
  lasso_taken_posterior_samples = ts_bayes_taken$posterior_samples
  
  lasso_taken_bayes_future_y_preds = ts_bayes_taken$future_y_preds             
  lasso_taken_bayes_y_pred = ts_bayes_taken$y_pred                             
  lasso_taken_bayes.pred.ci = ts_bayes_taken$pred.ci
  lasso_taken_bayes.pred.lower.CI = lasso_taken_bayes.pred.ci[,1]
  lasso_taken_bayes.pred.upper.CI = lasso_taken_bayes.pred.ci[,2]
  
  lasso_taken_ess_plot = ts_bayes_taken$ess_plot
  
  return(list(y_true = y_true,
              bayes_lasso_fit = list(fit = lasso_bayes_reg_model,
                                     pred = lasso_bayes_y_pred,
                                     lowerCI = lasso_bayes.pred.lower.ci,
                                     upperCI = lasso_bayes.pred.upper.ci,
                                     ess = lasso_ess_plot),
              bayes_lasso_taken_fit = list(fit = lasso_taken_bayes_reg_model,
                                           pred = lasso_taken_bayes_y_pred,
                                           lowerCI = lasso_taken_bayes.pred.lower.CI,
                                           upperCI = lasso_taken_bayes.pred.upper.CI,
                                           ess = lasso_taken_ess_plot)))
}

# --- Create empty dfs to save results 
# True value
y_true_mat <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)

# Bayesian Lasso 
bayes_lasso_lowerCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
bayes_lasso_upperCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
bayes_lasso_y_pred <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)

# Bayesian Lasso Taken without Smoothness
bayes_lasso_taken_lowerCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
bayes_lasso_taken_upperCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE) 
bayes_lasso_taken_y_pred <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE) 

# Bayesian Lasso Taken with Smoothness
bayes_lasso_taken_normal_lowerCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
bayes_lasso_taken_normal_upperCI <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE) 
bayes_lasso_taken_normal_y_pred <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE) 

# --- rfBL, rfBLT with and without smoothness derivatives ---
fixed_point = 85 

for (i in 1:n_simulation){
  data_vec = as.vector(I_noise_avg_mat[i,])
  res <- model_performance_lasso(
    df = data_vec,
    m = fixed_point,
    pred_size = pred_size,
    window_size = window_size,
    feature_selection = "factor",
    feature_constant = 0.5
  )
  
  # Bayesian Lasso
  bayes_lasso_lowerCI[i,] = as.vector(res$bayes_lasso_fit$lowerCI)
  bayes_lasso_upperCI[i,] = as.vector(res$bayes_lasso_fit$upperCI)
  bayes_lasso_y_pred[i,] = as.vector(res$bayes_lasso_fit$pred)
  
  # Bayesian Lasso Taken
  bayes_lasso_taken_lowerCI[i,] = as.vector(res$bayes_lasso_taken_fit$lowerCI)
  bayes_lasso_taken_upperCI[i,] = as.vector(res$bayes_lasso_taken_fit$upperCI)
  bayes_lasso_taken_y_pred[i,] = as.vector(res$bayes_lasso_taken_fit$pred)
}

for (i in 1:n_simulation){
  data_vec = as.vector(I_noise_avg_mat[i,])
  input_vec = data_vec[1:fixed_point]
  
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
  
  # Prediction  
  bayes_lasso_taken_normal_lowerCI[i,] = as.vector(res$pred.ci[,1])
  bayes_lasso_taken_normal_upperCI[i,] = as.vector(res$pred.ci[,2])
  bayes_lasso_taken_normal_y_pred[i,] = as.vector(res$y_pred)
}
y_true_mat = I_noise_avg_mat[,(fixed_point+1):(fixed_point+pred_size)]

# Bayes Lasso
m85_pred_bayes_lasso = bayes_lasso_y_pred
m85_lowerCI_bayes_lasso = bayes_lasso_lowerCI
m85_upperCI_bayes_lasso = bayes_lasso_upperCI

# Bayes Lasso Takens without Smoothing Derivatives
m85_pred_bayes_lasso_taken = bayes_lasso_taken_y_pred
m85_lowerCI_bayes_lasso_taken = bayes_lasso_taken_lowerCI
m85_upperCI_bayes_lasso_taken = bayes_lasso_taken_upperCI

# Bayes Lasso Takens with Smoothing Derivatives
m85_pred_BLT_normal_error = bayes_lasso_taken_normal_y_pred
m85_lowerCI_normal_error <- bayes_lasso_taken_normal_lowerCI
m85_upperCI_normal_error <- bayes_lasso_taken_normal_upperCI

# --- ARIMA ---
m85_arima_pred_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m85_arima_lowerCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m85_arima_upperCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m85_arima_avg_noise_aic <- numeric(n_simulation)
m85_arima_p <- numeric(n_simulation)
m85_arima_d <- numeric(n_simulation)
m85_arima_q <- numeric(n_simulation)

y_true_mat = I_noise_avg_mat[,(fixed_point+1):(fixed_point+pred_size)]
for (i in 1:n_simulation){
    arima <- arima_model(I_noise_avg_mat[i,], m=fixed_point, pred_size=pred_size)
    m85_arima_pred_avg_noise[i,] <- arima$pred
    m85_arima_lowerCI_avg_noise[i,] <- arima$lowerCI
    m85_arima_upperCI_avg_noise[i,] <- arima$upperCI
    m85_arima_avg_noise_aic[i] <- arima$aic
    m85_arima_p[i] <- arima$model$arma[1]
    m85_arima_q[i] <- arima$model$arma[2]
    m85_arima_d[i] <- arima$model$arma[6]
}

# --- Holt Exponential Smoothing ---
m85_holt_pred_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m85_holt_lowerCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m85_holt_upperCI_avg_noise <- matrix(rep(NA, (n_simulation*pred_size)), ncol=pred_size, nrow=n_simulation, byrow=TRUE)
m85_holt_avg_noise_aic <- numeric(n_simulation)

# I noise m = 85
y_true_mat = I_noise_avg_mat[,(fixed_point+1):(fixed_point+pred_size)]
for (i in 1:n_simulation){
    data_vec <- ts(I_noise_avg_mat[i,1:fixed_point], frequency=7)
    # For Holt's method (trend, no seasonality)
    fit_holt <- holt(data_vec, h=7)
    
    m85_holt_pred_avg_noise[i,] <- as.vector(fit_holt$mean)
    m85_holt_lowerCI_avg_noise[i,] <- as.vector(fit_holt$lower[,2])
    m85_holt_upperCI_avg_noise[i,] <- as.vector(fit_holt$upper[,2])
    m85_holt_avg_noise_aic[i] <- AIC(fit_holt$model)
}

# Save data for analyzing
save(I_noise_avg_mat,
     y_true_mat,

     # ARIMA
     m85_arima_pred_avg_noise,
     m85_arima_lowerCI_avg_noise,
     m85_arima_upperCI_avg_noise,
     m85_arima_avg_noise_aic,

     # Holt
     m85_holt_pred_avg_noise,
     m85_holt_lowerCI_avg_noise,
     m85_holt_upperCI_avg_noise,
     m85_holt_avg_noise_aic,
     
     # rfBL
     m85_pred_bayes_lasso,
     m85_lowerCI_bayes_lasso,
     m85_upperCI_bayes_lasso,
     
     # rfBLT without smoothing derivatives
     m85_pred_bayes_lasso_taken,
     m85_lowerCI_bayes_lasso_taken,
     m85_upperCI_bayes_lasso_taken,

     # rfBLT with smoothing derivatives
     m85_pred_BLT_normal_error,
     m85_lowerCI_normal_error,
     m85_upperCI_normal_error,
    
     file = "simulation0.1_m85_July17.RData")
