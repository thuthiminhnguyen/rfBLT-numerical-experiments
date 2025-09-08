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
  noise = rnorm(nrow(sueir_df), mean = 0, sd = 0.1)
  I_noise_mat[i,] = I + noise*max(abs(I))
}

# Compute 7-day right moving average
I_noise_avg_mat = matrix(rep(NA, 181*n_simulation), nrow=n_simulation, ncol=181, byrow=TRUE)
for(i in (1:n_simulation)){
  I_noise_avg_mat[i,] = rollapply(I_noise_mat[i,], 7, mean, align="right", partial=TRUE)
}

# --- LSTM model ---
lstm_model_fn <- function(df, m, window_size=7, hidden_dim=64, num_epochs=300, batch_size=64, learning_rate=0.0075){
  # Embed the training data
  train_df <- embed(df[1:m], window_size+1)
  x_train_df <- train_df[, 2:ncol(train_df)]
  # Reverse data in order
  x_train_df <- x_train_df[, window_size:1] 
  y_train_df <- train_df[, 1]
  # Embed the testing data
  x_test_df <- embed(df[(m-window_size+1):m], window_size)
  # Reverse data in order
  x_test_df <- x_test_df[, window_size:1]
  
  input_dim <- 1
  lookback <- window_size
  
  x_train <- array_reshape(x_train_df, c(dim(x_train_df)[1], lookback, input_dim))
  y_train <- y_train_df
  
  model <- keras_model_sequential() %>%
    layer_lstm(units = hidden_dim, 
               return_sequences = TRUE, 
               dropout = 0.1,
               recurrent_dropout = 0.14,
               input_shape = c(lookback, input_dim)) %>%
    layer_lstm(units = hidden_dim, 
               return_sequences = TRUE,
               dropout = 0.1,
               recurrent_dropout = 0.14) %>%
    layer_lstm(units = hidden_dim, 
               return_sequences = FALSE,
               dropout = 0.1,
               recurrent_dropout = 0.14) %>%
    layer_dense(units = 1)
  
  # Compile the model using the mean squared error loss and the Adam optimizer
  model %>% compile(loss="mean_squared_error", optimizer=optimizer_adam(learning_rate=learning_rate))
  # Train the model on the training data
  history <- model %>% fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
  
  predictions_lstm <- numeric(pred_size)
  x_input <- array_reshape(x_test_df, c(1, lookback, input_dim))
  predictions_lstm[1] <- model %>% predict(x_input)
  
  for (i in 2:pred_size) {
    x_input <- tail(c(df[1:m], predictions_lstm[1:(i-1)]), window_size)
    x_input <- array_reshape(x_input, c(1, lookback, input_dim))
    predictions_lstm[i] <- model %>% predict(x_input)
  }
  return(list(pred=predictions_lstm, loss=history$metrics$loss))
}

# Define number of epochs
num_epochs = 200

# --- m = 85 ---
m85_pred_lstm_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=pred_size)
m85_loss_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=num_epochs)
for (i in 1:nrow(I_noise_avg_mat)){
  res <- lstm_model_fn(df=I_noise_avg_mat[i,], 
                       m=85, 
                       window_size=9, 
                       hidden_dim=64, 
                       num_epochs=200, 
                       batch_size=64, 
                       learning_rate=0.0075)
  
  m85_pred_lstm_mat[i, ] <- res$pred
  m85_loss_mat[i, ] <- res$loss
}

# --- m = 102 ---
m102_pred_lstm_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=pred_size)
m102_loss_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=num_epochs)
for (i in 1:nrow(I_noise_avg_mat)){
  res <- lstm_model_fn(df=I_noise_avg_mat[i,], 
                       m=102, 
                       window_size=9, 
                       hidden_dim=64, 
                       num_epochs=200, 
                       batch_size=64, 
                       learning_rate=0.0075)
  
  m102_pred_lstm_mat[i, ] <- res$pred
  m102_loss_mat[i, ] <- res$loss
}

# --- m = 108 ---
m108_pred_lstm_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=pred_size)
m108_loss_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=num_epochs)
for (i in 1:nrow(I_noise_avg_mat)){
  res <- lstm_model_fn(df=I_noise_avg_mat[i,],
                       m=108,
                       window_size=9,
                       hidden_dim=64,
                       num_epochs=num_epochs,
                       batch_size=64,
                       learning_rate=0.0075)
  
  m108_pred_lstm_mat[i, ] <- res$pred
  m108_loss_mat[i, ] <- res$loss
}

# --- m = 114 ---
m114_pred_lstm_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=pred_size)
m114_loss_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=num_epochs)
for (i in 1:nrow(I_noise_avg_mat)){
  res <- lstm_model_fn(df=I_noise_avg_mat[i,], 
                       m=114, 
                       window_size=9, 
                       hidden_dim=64, 
                       num_epochs=200, 
                       batch_size=64, 
                       learning_rate=0.0075)
  
  m114_pred_lstm_mat[i, ] <- res$pred
  m114_loss_mat[i, ] <- res$loss
}

# --- m = 125 ---
m125_pred_lstm_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=pred_size)
m125_loss_mat = matrix(NA, nrow=nrow(I_noise_avg_mat), ncol=num_epochs)
for (i in 1:nrow(I_noise_avg_mat)){
  res <- lstm_model_fn(df=I_noise_avg_mat[i,],
                       m=125,
                       window_size=9,
                       hidden_dim=64,
                       num_epochs=num_epochs,
                       batch_size=64,
                       learning_rate=0.0075)
  
  m125_pred_lstm_mat[i, ] <- res$pred
  m125_loss_mat[i, ] <- res$loss
}
