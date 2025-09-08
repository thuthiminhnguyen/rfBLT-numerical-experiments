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

# --- Min-Max scaling ---
# Normalize train data
normalization_fn <- function(x){
  max_x <- max(x)
  min_x <- min(x)
  norm_vec <- as.vector((x-min_x)/(max_x-min_x))
  return(list(norm_vec=norm_vec, max_x=max_x, min_x=min_x))
}
# Revert function
revert_fn <- function(x, max_org, min_org){
  return(as.vector(x*(max_org-min_org)+min_org))
}
# Normalize test data
normalized_test <- function(x, max_org, min_org){
  norm_vec <- as.vector((x-min_org)/(max_org-min_org))
  return(norm_vec)
}

# --- LSTM model function ---
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
               dropout = 0.06,
               recurrent_dropout = 0.14,
               input_shape = c(lookback, input_dim)) %>%
    layer_lstm(units = hidden_dim, 
               return_sequences = FALSE,
               dropout = 0.06,
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

# --- Import data ---
symbol <- "^GSPC"
start_date <- "2022-01-15"
end_date <- "2025-05-31"
# The end of the first training period of the expanding window forecasting
fixed_point <- 753  

# Get stock data for the specified period
getSymbols(symbol, from = start_date, to = end_date)
df = GSPC$GSPC.Close
prices <- as.vector(df$GSPC.Close)

# Visualize data
plot(as.vector(GSPC$GSPC.Close), type="l")
abline(v=fixed_point, lty=2)

# Specify input parameters
window_size <- 20
pred_size <- 7

# --- Create empty data frame to save results ---
start_train_point <- fixed_point
last_train_point <- length(as.vector(prices))-pred_size
# Number of rows 
df_nrow <- length(start_train_point:last_train_point)
# Number of columns
df_ncol = pred_size+1

# Create empty dataframes to save predictions
df_true <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
colnames(df_true) <- c("training_point", paste0("t", 1:pred_size))
# Assign values to the training_point column for pred df
df_true$training_point <- start_train_point:last_train_point

df_pred_lstm <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
colnames(df_pred_lstm) <- c("training_point", paste0("t", 1:pred_size))
df_pred_lstm$training_point <- start_train_point:last_train_point

# --- Execute and add output to empty dfs ---
# True data
for (i in 1:df_nrow){
  vec_dat <- as.vector(prices)
  df_true[i, 2:df_ncol] <- vec_dat[(df_true[i, 1]+1):(df_true[i, 1]+7)]
}

# LSTM
num_epochs = 200
df_loss = data.frame(matrix(rep(NA, df_nrow*(num_epochs+1)), nrow=df_nrow, ncol=num_epochs+1))
colnames(df_loss) <- c("training_point", paste0("Epoch_", 1:(num_epochs)))
df_loss$training_point <- start_train_point:last_train_point

for (i in 1:df_nrow){
  x <- prices
  index <- df_true[i, 1]
  
  normalize_res <- normalization_fn(x[1:index])
  train_vec <- normalize_res$norm_vec
  max_train <- normalize_res$max_x
  min_train <- normalize_res$min_x
  
  res_lstm <- lstm_model_fn(
    df = train_vec,
    m = df_true[i, 1],
    window_size = window_size,
    hidden_dim = 64,
    num_epochs = num_epochs,
    batch_size = 32,
    learning_rate = 0.0075
  )

  
  
  df_pred_lstm[i, 2:df_ncol] <- revert_fn(x=res_lstm$pred, max_org=max_train, min_org=min_train)
  df_loss[i, 2:(num_epochs+1)] <- res_lstm$loss
}

# Save data for analyzing
save(prices, 
     df_true,     
     df_pred_lstm,
     df_loss,
     file = "SP500_LSTM_output_July11.RData")
