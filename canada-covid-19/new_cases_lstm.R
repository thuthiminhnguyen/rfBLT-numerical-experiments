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
                   return_sequences = TRUE,
                   dropout = 0.06,
                   recurrent_dropout = 0.14) %>%
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
raw_df = read.csv("covid19-download.csv")

# Filter the dataframe for 'Canada' and select relevant columns
canada_df <- raw_df %>%
  dplyr::filter(prname == 'Canada') %>%
  select(date, numdeaths, numtotal, numconf, numactive)

# Convert the date column to Date type if it's not already
canada_df$date <- as.Date(canada_df$date)

# Sort the dataframe by date just in case
canada_df <- canada_df %>% arrange(date)

# Calculate the difference in dates and number of deaths
canada_df <- canada_df %>%
  mutate(date_diff = c(NA, diff(date)),
         numconf_diff = c(NA, diff(numconf)),
         numdeaths_diff = c(NA, diff(numdeaths)),
         numtotal_diff = c(NA, diff(numtotal)))
# Omit NA rows
canada_df = na.omit(canada_df)
# Filter the dataframe where there is no missing gap
filtered_df <- canada_df %>%
  dplyr::filter(date >= as.Date("2020-03-12"))
# Calculating right-moving average
filtered_df$numconf_diff_avg <- rollapply(filtered_df$numconf_diff, 7, mean, align="right", partial=TRUE)
filtered_df$index <- 1:nrow(filtered_df)

# --- Split the data into multiple waves ---
m1 = 155
m2 = 362
m3 = 490
m4 = 625
# Plotting the data with vertical lines for the prediction period
plot(filtered_df$date, filtered_df$numconf_diff_avg, col = "black", lwd = 2, type="l",
     xlab = "Time (days)", ylab="COVID19 Daily New Cases in Canada", main = "New cases", cex=0.5)
# Add vertical lines 
abline(v = filtered_df$date[m1], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m2], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m3], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m4], col = "red", lwd = 2, lty = 2)

# --- Extract second wave ---
second_wave_df <- filtered_df %>%
  dplyr::filter(index >= m1 & index < m2)  # Adjust the range as needed
# Add an index column starting from 1 
second_wave_df$index <- 1:nrow(second_wave_df)

# --- Visualize second wave and the start of the expanding prediction period---
m = 100
plot(second_wave_df$date, second_wave_df$numconf_diff_avg, col = "black", lwd = 2, type="l",
     xlab = "Time (days)", ylab="New cases", main = "New cases - COVID19 2nd wave in Canada", cex=0.5)
abline(v = second_wave_df$date[m+1], col = "blue", lwd = 2, lty = 2)

# --- Specify input ---
pred_size = 7 
d = 4 # assumption of number of parameters
window_size = 2*d+1

# --- Create empty dfs to save results ---
start_train_point <- 100
last_train_point <- length(as.vector(second_wave_df$numconf_diff_avg))-pred_size
# Number of rows 
df_nrow <- length(start_train_point:last_train_point)
# Number of columns
df_ncol = 8

# Create empty dataframes to save predictions
df_true <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_pred_lstm <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))

# Assign column names for pred df
colnames(df_true) <- c("training_point", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
colnames(df_pred_lstm) <- c("training_point", "t1", "t2", "t3", "t4", "t5", "t6", "t7")

# Assign values to the training_point column for pred df
df_true$training_point <- start_train_point:last_train_point
df_pred_lstm$training_point <- start_train_point:last_train_point

# --- Execute and add output to empty dfs ---
# True data
for (i in 1:df_nrow){
    vec_dat <- as.vector(second_wave_df$numconf_diff_avg)
    df_true[i, 2:df_ncol] <- vec_dat[(df_true[i, 1]+1):(df_true[i, 1]+7)]
}

# LSTM
num_epochs=300
loss_lstm_mat = matrix(NA, nrow=df_nrow, ncol=num_epochs)
for (i in 1:df_nrow){
    x <- second_wave_df$numconf_diff_avg
    index <- df_pred_lstm[i, 1]
    
    normalize_res <- normalization_fn(x[1:index])
    train_vec <- normalize_res$norm_vec
    max_train <- normalize_res$max_x
    min_train <- normalize_res$min_x
  
    res <- lstm_model_fn(df=train_vec, 
                         m=df_pred_lstm[i, 1], 
                         window_size=9, 
                         hidden_dim=64, 
                         num_epochs=300, 
                         batch_size=64, 
                         learning_rate=0.0075)

    df_pred_lstm[i, 2:df_ncol] <- revert_fn(x=res$pred, max_org=max_train, min_org=min_train) 
    loss_lstm_mat[i, ] <- res$loss
}
