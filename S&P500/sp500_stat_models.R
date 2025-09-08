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

# --- ARIMA ---
arima_model_fn <- function(df, m, pred_size){
  fit_vec <- as.vector(df[1:m])
  arima_fit <- auto.arima(fit_vec)
  
  forecast_result <- forecast::forecast(arima_fit, h = pred_size)
  predictions_arima <- forecast_result$mean
  
  arima_lowerCI = forecast_result$lower[,2]
  arima_upperCI = forecast_result$upper[,2]
  return(list(model=arima_fit, pred=predictions_arima, lowerCI=arima_lowerCI, upperCI=arima_upperCI, aic=arima_fit$aic))
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

df_pred_arima <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_pred_holt <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_pred_bayes_lasso <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_pred_bayes_lasso_taken_normal_error <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))

# Assign column names for pred df
colnames(df_pred_arima) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_pred_holt) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_pred_bayes_lasso) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_pred_bayes_lasso_taken_normal_error) <- c("training_point", paste0("t", 1:pred_size))

# Assign values to the training_point column for pred df
df_pred_arima$training_point <- start_train_point:last_train_point
df_pred_holt$training_point <- start_train_point:last_train_point
df_pred_bayes_lasso$training_point <- start_train_point:last_train_point
df_pred_bayes_lasso_taken_normal_error$training_point <- start_train_point:last_train_point

# Create empty dataframe for saving lower CI
df_lower_arima <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_lower_holt <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_lower_bayes_lasso <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_lower_bayes_lasso_taken_normal_error <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))

# Assign column names for lower CI df
colnames(df_lower_arima) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_lower_holt) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_lower_bayes_lasso) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_lower_bayes_lasso_taken_normal_error) <- c("training_point", paste0("t", 1:pred_size))

# Assign values to the training_point column for lower CI df
df_lower_arima$training_point <- start_train_point:last_train_point
df_lower_holt$training_point <- start_train_point:last_train_point
df_lower_bayes_lasso$training_point <- start_train_point:last_train_point
df_lower_bayes_lasso_taken_normal_error$training_point <- start_train_point:last_train_point

# Create empty dataframe for saving upper CI
df_upper_arima <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_upper_holt <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_upper_bayes_lasso <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_upper_bayes_lasso_taken_normal_error <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))

# Assign column names for lower CI df
colnames(df_upper_arima) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_upper_holt) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_upper_bayes_lasso) <- c("training_point", paste0("t", 1:pred_size))
colnames(df_upper_bayes_lasso_taken_normal_error) <- c("training_point", paste0("t", 1:pred_size))

# Assign values to the training_point column for lower CI df
df_upper_arima$training_point <- start_train_point:last_train_point
df_upper_holt$training_point <- start_train_point:last_train_point
df_upper_bayes_lasso$training_point <- start_train_point:last_train_point
df_upper_bayes_lasso_taken_normal_error$training_point <- start_train_point:last_train_point

# --- Execute and add output to empty dfs ---
# True data
for (i in 1:df_nrow){
  vec_dat <- as.vector(prices)
  df_true[i, 2:df_ncol] <- vec_dat[(df_true[i, 1]+1):(df_true[i, 1]+7)]
}

# ARIMA
for (i in 1:df_nrow){
  x <- prices
  index <- df_true[i, 1]
  
  normalize_res <- normalization_fn(x[1:index])
  train_vec <- normalize_res$norm_vec
  max_train <- normalize_res$max_x
  min_train <- normalize_res$min_x
  
  test_vec <- normalized_test(x[(index+1):(length(x))], max_org=max_train, min_org=min_train)
  combine_vec <- c(train_vec, test_vec)
  
  arima_res <- arima_model_fn(as.vector(combine_vec), m=index, pred_size=pred_size)
  df_pred_arima[i, 2:df_ncol] <- revert_fn(x=arima_res$pred, max_org=max_train, min_org=min_train)
  df_lower_arima[i, 2:df_ncol] <- revert_fn(x=arima_res$lowerCI, max_org=max_train, min_org=min_train)
  df_upper_arima[i, 2:df_ncol] <- revert_fn(x=arima_res$upperCI, max_org=max_train, min_org=min_train)
}

# Holt Exponential Smoothing
for (i in 1:df_nrow){
  x <- prices
  index <- df_true[i,1]
  
  normalize_res <- normalization_fn(x[1:index])
  train_vec <- normalize_res$norm_vec
  max_train <- normalize_res$max_x
  min_train <- normalize_res$min_x
  
  data_vec <- ts(train_vec, frequency=7)
  # For Holt's method (trend, no seasonality)
  fit_holt <- holt(data_vec, h=7)
  df_pred_holt[i, 2:df_ncol] <- revert_fn(x=as.vector(fit_holt$mean), max_org=max_train, min_org=min_train) 
  df_lower_holt[i, 2:df_ncol] <- revert_fn(x=as.vector(fit_holt$lower[,2]), max_org=max_train, min_org=min_train) 
  df_upper_holt[i, 2:df_ncol] <- revert_fn(x=as.vector(fit_holt$upper[,2]), max_org=max_train, min_org=min_train) 
}

# Random Feature Bayesian Lasso (rfBL)
for (i in 1:df_nrow){
  x <- prices
  index <- df_true[i,1]
  
  normalize_res <- normalization_fn(x[1:index])
  train_vec <- normalize_res$norm_vec
  max_train <- normalize_res$max_x
  min_train <- normalize_res$min_x
  
  ts_bayes <- ts_forecast_bayes_reg_rfm(ts_data = as.vector(train_vec),
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
                                        CI = 95)
  # Bayes Lasso
  df_pred_bayes_lasso[i, 2:df_ncol] <- revert_fn(x=as.vector(ts_bayes$y_pred), max_org=max_train, min_org=min_train)
  df_lower_bayes_lasso[i, 2:df_ncol] <- revert_fn(x=as.vector(ts_bayes$pred.ci[,1]), max_org=max_train, min_org=min_train) 
  df_upper_bayes_lasso[i, 2:df_ncol] <- revert_fn(x=as.vector(ts_bayes$pred.ci[,2]), max_org=max_train, min_org=min_train) 
}

# Random Feature Bayesian Lasso Takens (rfBLT)
for (i in 1:df_nrow){
  x <- prices
  index <- df_true[i, 1]
  
  normalize_res <- normalization_fn(x[1:index])
  train_vec <- normalize_res$norm_vec
  max_train <- normalize_res$max_x
  min_train <- normalize_res$min_x
  
  normal_error_res <- ts_forecast_bayes_reg_rfm_taken(
    ts_data = as.vector(train_vec),
    time = 1:length(train_vec),
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
  
  df_pred_bayes_lasso_taken_normal_error[i, 2:df_ncol] <- revert_fn(x=as.vector(normal_error_res$y_pred), max_org=max_train, min_org=min_train)   
  df_lower_bayes_lasso_taken_normal_error[i, 2:df_ncol] <- revert_fn(x=as.vector(normal_error_res$pred.ci[,1]), max_org=max_train, min_org=min_train)   
  df_upper_bayes_lasso_taken_normal_error[i, 2:df_ncol] <- revert_fn(x=as.vector(normal_error_res$pred.ci[,2]), max_org=max_train, min_org=min_train)
}

# Save data for analyzing
save(prices, 
     df_true,

     # ARIMA
     df_pred_arima,
     df_lower_arima,
     df_upper_arima,

     # Holt Exponential Smoothing
     df_pred_holt,
     df_lower_holt,
     df_upper_holt,

     # rfBL
     df_pred_bayes_lasso,
     df_lower_bayes_lasso,
     df_upper_bayes_lasso,

     # rfBLT
     df_pred_bayes_lasso_taken_normal_error,
     df_lower_bayes_lasso_taken_normal_error,
     df_upper_bayes_lasso_taken_normal_error,
     
     file = "SP500_without_LSTM_output_July11.RData")
