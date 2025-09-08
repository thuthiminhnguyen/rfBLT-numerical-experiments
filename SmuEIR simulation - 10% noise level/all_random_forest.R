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
# Random Forest
library(randomForest)

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
parameters <- c(beta = beta, gamma = gamma, sigma = sigma, mu = mu, N=N) # Transmission and recovery rates
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
    noise = rnorm(nrow(sueir_df), mean = 0, sd = 0.1)
    I_noise_mat[i,] = I + noise*max(abs(I))
}

# Compute 7-day right-moving average
I_noise_avg_mat = matrix(rep(NA, 181*n_simulation), nrow=n_simulation, ncol=181, byrow=TRUE)
for(i in (1:n_simulation)){
    I_noise_avg_mat[i,] = rollapply(I_noise_mat[i,], 7, mean, align="right", partial=TRUE)
}

# Specify input parameters
pred_size = 7
window_size = 9

df_nrow = nrow(I_noise_avg_mat)
df_ncol = ncol(I_noise_avg_mat)

# --- m = 85 ---
m = 85
m85_pred_rforest <- matrix(rep(NA, df_nrow*pred_size), ncol=pred_size, nrow=df_nrow)

for (i in 1:df_nrow){
    index <- m
    input_vec <- as.vector(I_noise_avg_mat[i, 1:index])
    len <- length(input_vec)
    rofc <- (input_vec[-1]-input_vec[-len])/input_vec[-len]
    
    x <- rofc

    train_mat <- embed(x[1:(index-1)], window_size+1)
    y_train_vec <- train_mat[,1]
    x_train_mat <- train_mat[,2:(window_size+1)]

    # Revert data in time order
    x_train_mat <- as.data.frame(x_train_mat[,window_size:1])
    colnames(x_train_mat) <- paste0("t", 1:7)

    x_test_mat <- matrix(rev(train_mat[nrow(train_mat),1:window_size]), nrow=1)
    forecasts_rf <- numeric(pred_size)

    # Fit the model
    fit_rf <- randomForest(x_train_mat, 
                           y_train_vec,
                           type = "regression",
                           ntree = 1000)

    # Make prediction
    forecasts_rf[1] <- predict(fit_rf, x_test_mat)
    for (j in 2:pred_size){
        fit_data <- matrix(tail(c(x_test_mat, forecasts_rf[1:(j-1)]), window_size), nrow=1)
        # Predict using the test set
        forecasts_rf[j] <- predict(fit_rf, fit_data)
    }

    pred_rofc <- forecasts_rf
    y_pred_vec <- numeric(pred_size)
    past <- tail(input_vec, 1)

    # rofc[t] <- x[t]-x[t-1]/x[t-1]
    # x[t] <- rofc[t]*x[t-1]+x[t-1]
    for (j in 1:pred_size){
         y_pred_vec[j] <- pred_rofc[j]*past+past
         past <- y_pred_vec[j]
    }
    m85_pred_rforest[i, ] <- y_pred_vec
}

# --- m = 102 ---
m = 102
m102_pred_rforest <- matrix(rep(NA, df_nrow*pred_size), ncol=pred_size, nrow=df_nrow)

for (i in 1:df_nrow){
    index <- m
    input_vec <- as.vector(I_noise_avg_mat[i, 1:index])
    len <- length(input_vec)
    rofc <- (input_vec[-1]-input_vec[-len])/input_vec[-len]
    
    x <- rofc

    train_mat <- embed(x[1:(index-1)], window_size+1)
    y_train_vec <- train_mat[,1]
    x_train_mat <- train_mat[,2:(window_size+1)]

    # Revert data in time order
    x_train_mat <- as.data.frame(x_train_mat[,window_size:1])
    colnames(x_train_mat) <- paste0("t", 1:7)

    x_test_mat <- matrix(rev(train_mat[nrow(train_mat),1:window_size]), nrow=1)
    forecasts_rf <- numeric(pred_size)

    # Fit the model
    fit_rf <- randomForest(x_train_mat, 
                           y_train_vec,
                           type = "regression",
                           ntree = 1000)

    # Make prediction
    forecasts_rf[1] <- predict(fit_rf, x_test_mat)
    for (j in 2:pred_size){
        fit_data <- matrix(tail(c(x_test_mat, forecasts_rf[1:(j-1)]), window_size), nrow=1)
        # Predict using the test set
        forecasts_rf[j] <- predict(fit_rf, fit_data)
    }

    pred_rofc <- forecasts_rf
    y_pred_vec <- numeric(pred_size)
    past <- tail(input_vec, 1)

    # rofc[t] <- x[t]-x[t-1]/x[t-1]
    # x[t] <- rofc[t]*x[t-1]+x[t-1]
    for (j in 1:pred_size){
         y_pred_vec[j] <- pred_rofc[j]*past+past
         past <- y_pred_vec[j]
    }
    m102_pred_rforest[i, ] <- y_pred_vec
}

# --- m = 108 ---
m = 108
m108_pred_rforest <- matrix(rep(NA, df_nrow*pred_size), ncol=pred_size, nrow=df_nrow)

for (i in 1:df_nrow){
    index <- m
    input_vec <- as.vector(I_noise_avg_mat[i, 1:index])
    len <- length(input_vec)
    rofc <- (input_vec[-1]-input_vec[-len])/input_vec[-len]
    
    x <- rofc

    train_mat <- embed(x[1:(index-1)], window_size+1)
    y_train_vec <- train_mat[,1]
    x_train_mat <- train_mat[,2:(window_size+1)]

    # Revert data in time order
    x_train_mat <- as.data.frame(x_train_mat[,window_size:1])
    colnames(x_train_mat) <- paste0("t", 1:7)

    x_test_mat <- matrix(rev(train_mat[nrow(train_mat),1:window_size]), nrow=1)
    forecasts_rf <- numeric(pred_size)

    # Fit the model
    fit_rf <- randomForest(x_train_mat, 
                           y_train_vec,
                           type = "regression",
                           ntree = 1000)

    # Make prediction
    forecasts_rf[1] <- predict(fit_rf, x_test_mat)

    for (j in 2:pred_size){
        fit_data <- matrix(tail(c(x_test_mat, forecasts_rf[1:(j-1)]), window_size), nrow=1)
        # Predict using the test set
        forecasts_rf[j] <- predict(fit_rf, fit_data)
    }

    pred_rofc <- forecasts_rf
    y_pred_vec <- numeric(pred_size)
    past <- tail(input_vec, 1)

    # rofc[t] <- x[t]-x[t-1]/x[t-1]
    # x[t] <- rofc[t]*x[t-1]+x[t-1]
    for (j in 1:pred_size){
         y_pred_vec[j] <- pred_rofc[j]*past+past
         past <- y_pred_vec[j]
    }
    m108_pred_rforest[i, ] <- y_pred_vec
}

# --- m = 114 ---
m = 114
m114_pred_rforest <- matrix(rep(NA, df_nrow*pred_size), ncol=pred_size, nrow=df_nrow)

for (i in 1:df_nrow){    
    index <- m
    input_vec <- as.vector(I_noise_avg_mat[i, 1:index])
    len <- length(input_vec)
    rofc <- (input_vec[-1]-input_vec[-len])/input_vec[-len]
    
    x <- rofc

    train_mat <- embed(x[1:(index-1)], window_size+1)
    y_train_vec <- train_mat[,1]
    x_train_mat <- train_mat[,2:(window_size+1)]

    # Revert data in time order
    x_train_mat <- as.data.frame(x_train_mat[,window_size:1])
    colnames(x_train_mat) <- paste0("t", 1:7)

    x_test_mat <- matrix(rev(train_mat[nrow(train_mat),1:window_size]), nrow=1)
    forecasts_rf <- numeric(pred_size)

    # Fit the model
    fit_rf <- randomForest(x_train_mat, 
                           y_train_vec,
                           type = "regression",
                           ntree = 1000)

    # Make prediction
    forecasts_rf[1] <- predict(fit_rf, x_test_mat)
    for (j in 2:pred_size){
        fit_data <- matrix(tail(c(x_test_mat, forecasts_rf[1:(j-1)]), window_size), nrow=1)
        # Predict using the test set
        forecasts_rf[j] <- predict(fit_rf, fit_data)
    }

    pred_rofc <- forecasts_rf
    y_pred_vec <- numeric(pred_size)
    past <- tail(input_vec, 1)

    # rofc[t] <- x[t]-x[t-1]/x[t-1]
    # x[t] <- rofc[t]*x[t-1]+x[t-1]
    for (j in 1:pred_size){
         y_pred_vec[j] <- pred_rofc[j]*past+past
         past <- y_pred_vec[j]
    }
    m114_pred_rforest[i, ] <- y_pred_vec
}

# --- m = 125 ---
m = 125
m125_pred_rforest <- matrix(rep(NA, df_nrow*pred_size), ncol=pred_size, nrow=df_nrow)

for (i in 1:df_nrow){    
    index <- m
    input_vec <- as.vector(I_noise_avg_mat[i, 1:index])
    len <- length(input_vec)
    rofc <- (input_vec[-1]-input_vec[-len])/input_vec[-len]
    
    x <- rofc

    train_mat <- embed(x[1:(index-1)], window_size+1)
    y_train_vec <- train_mat[,1]
    x_train_mat <- train_mat[,2:(window_size+1)]

    # Revert data in time order
    x_train_mat <- as.data.frame(x_train_mat[,window_size:1])
    colnames(x_train_mat) <- paste0("t", 1:7)

    x_test_mat <- matrix(rev(train_mat[nrow(train_mat),1:window_size]), nrow=1)
    forecasts_rf <- numeric(pred_size)

    # Fit the model
    fit_rf <- randomForest(x_train_mat, 
                           y_train_vec,
                           type = "regression",
                           ntree = 1000)

    # Make prediction
    forecasts_rf[1] <- predict(fit_rf, x_test_mat)
    for (j in 2:pred_size){
        fit_data <- matrix(tail(c(x_test_mat, forecasts_rf[1:(j-1)]), window_size), nrow=1)
        # Predict using the test set
        forecasts_rf[j] <- predict(fit_rf, fit_data)
    }

    pred_rofc <- forecasts_rf
    y_pred_vec <- numeric(pred_size)
    past <- tail(input_vec, 1)

    # rofc[t] <- x[t]-x[t-1]/x[t-1]
    # x[t] <- rofc[t]*x[t-1]+x[t-1]
    for (j in 1:pred_size){
         y_pred_vec[j] <- pred_rofc[j]*past+past
         past <- y_pred_vec[j]
    }
    m125_pred_rforest[i, ] <- y_pred_vec
}

# Save data for analyzing
save(I_noise_avg_mat,
     m85_pred_rforest,
     m102_pred_rforest,
     m108_pred_rforest,
     m114_pred_rforest,
     m125_pred_rforest,
     file="simulation0.1_rforest_July22.RData")
