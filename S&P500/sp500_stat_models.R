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
  
  data_vec <- ts(train_vec, frequency=252)
  # For Holt's method (trend, no seasonality)
  fit_holt <- holt(data_vec, h=7)
  df_pred_holt[i, 2:df_ncol] <- revert_fn(x=as.vector(fit_holt$mean), max_org=max_train, min_org=min_train) 
  df_lower_holt[i, 2:df_ncol] <- revert_fn(x=as.vector(fit_holt$lower[,2]), max_org=max_train, min_org=min_train) 
  df_upper_holt[i, 2:df_ncol] <- revert_fn(x=as.vector(fit_holt$upper[,2]), max_org=max_train, min_org=min_train) 
}

# Random Feature Bayesian Lasso (RF)
rfBL_residual_trend_train_mat <- matrix(NA, nrow=df_nrow, ncol=nrow(df))
df_residuals_acf_rfBL <- matrix(NA, nrow=df_nrow, ncol=52)
df_residuals_acf_rfBL[,1] <- df_true[,1]
colnames(df_residuals_acf_rfBL) <- c("training_point", paste0(0:(ncol(df_residuals_acf_rfBL)-2)))

# Store samples
rfBL_pred_trend_list <- list()
rfBL_coef_list <- list()
rfBL_sigma2_list <- list()
rfBL_ess_list <- list()

for (i in 1:df_nrow){
  if (i %% 10 == 0){
    print(i)
  }
  x <- prices
  index <- df_true[i,1]
    
  normalize_res <- normalization_fn(x[1:index])
  train_vec <- normalize_res$norm_vec
  # mean_train <- normalize_res$mean_x
  max_train <- normalize_res$max_x
  min_train <- normalize_res$min_x

  test_vec <- normalized_test(x[(index+1):(length(x))], max_org=max_train, min_org=min_train)
    
  vec_dat <- c(train_vec, test_vec)
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
    # Residuals
    train_mat <- embed(train_vec, (window_size+1)) # to get true value
    residuals <- train_mat[,1] - ts_bayes$fitted.values
    rfBL_residual_trend_train_mat[i, 1:length(residuals)] <- residuals

    residual_acf <- acf(residuals, lag.max=50, plot=FALSE)
    df_residuals_acf_rfBL[i, -1] <- as.vector(residual_acf$acf) 

    rfBL_coef_list[[i]] <- ts_bayes$posterior_samples
    rfBL_sigma2_list[[i]] <- ts_bayes$posterior_sigma2
    rfBL_ess_list[[i]] <- ts_bayes$ess
    
    # Bayes Lasso
    rfBL_pred_trend_list[[i]] <- ts_bayes$future_y_preds 
    df_pred_bayes_lasso[i, 2:df_ncol] <- revert_fn(x=as.vector(ts_bayes$y_pred), max_org=max_train, min_org=min_train)
    df_lower_bayes_lasso[i, 2:df_ncol] <- revert_fn(x=as.vector(ts_bayes$pred.ci[,1]), max_org=max_train, min_org=min_train) 
    df_upper_bayes_lasso[i, 2:df_ncol] <- revert_fn(x=as.vector(ts_bayes$pred.ci[,2]), max_org=max_train, min_org=min_train) 
}

## Save results
### ESS plot
pdf("sp500_RF_ess.pdf", width=7, height=5)
par(mfrow=c(1, 1))
for (i in 1:df_nrow){
    plot(rfBL_ess_list[[i]])
}
dev.off()

### Residuals
pdf("sp500_RF_residuals.pdf", width=10, height=4)
par(mfrow=c(1, 2))
for (i in 1:dim(df_residuals_acf_rfBL)[1]){
  # ACF of residuals
  plot(df_residuals_acf_rfBL[i, -1], type="h", ylab="ACF", xlab="Lag", main=paste(i))
  # QQ plot of residuals
  qqnorm(na.omit(rfBL_residual_trend_train_mat[i,]), main=paste(i))
  qqline(na.omit(rfBL_residual_trend_train_mat[i,]))
}
dev.off()

### Coefficients of last iter
pdf("sp500_RF_coefs_samples.pdf", width=15, height=20)
par(mfrow=c(6, 3))
for (i in 1:ncol(ts_bayes$posterior_samples)){
  plot(ts_bayes$posterior_samples[,i], type="l", main=paste0("Tract plot of Beta ", i-1), ylab=paste0("Beta ", i-1))
  hist(ts_bayes$posterior_samples[,i], probability = TRUE, main=paste0("Density of Beta ", i-1), xlab=paste0("Beta ", i-1))
  boxplot(ts_bayes$posterior_samples[,i], horizontal=TRUE, main=paste0("Box plot of Beta ", i-1))
}
dev.off()

### Sigma2 of last iter
pdf("sp500_RF_sigma2_samples.pdf", width=15, height=4)
par(mfrow=c(1, 3))
plot(ts_bayes$posterior_sigma2, type="l", ylab="Sigma2", main="Trace plot of Sigma2")
hist(ts_bayes$posterior_sigma2, probability = TRUE, main=paste0("Density of Sigma2"), xlab=paste0("Sigma2"))
boxplot(ts_bayes$posterior_sigma2, horizontal=TRUE, main="Box plot of Sigma2")
dev.off()

# Random Feature Bayesian Lasso Takens (rfBLT)
rfBLT_residual_trend_train_mat <- matrix(NA, nrow=df_nrow, ncol=nrow(df))
df_residuals_acf_rfBLT <- matrix(NA, nrow=df_nrow, ncol=52)
df_residuals_acf_rfBLT[,1] <- df_true[,1]
colnames(df_residuals_acf_rfBLT) <- c("training_point", paste0(0:(ncol(df_residuals_acf_rfBLT)-2)))

# Store samples 
rfBLT_pred_trend_list <- list()
rfBLT_coef_list <- list()
rfBLT_sigma2_list <- list()
rfBLT_ess_list <- list()

for (i in 1:df_nrow){
  if (i %% 10 == 0){
    print(i)
  }
  x <- prices
  index <- df_true[i, 1]
    
  normalize_res <- normalization_fn(x[1:index])
  train_vec <- normalize_res$norm_vec
  # mean_train <- normalize_res$mean_x
  max_train <- normalize_res$max_x
  min_train <- normalize_res$min_x

  test_vec <- normalized_test(x[(index+1):(length(x))], max_org=max_train, min_org=min_train)
    
  vec_dat <- c(train_vec, test_vec)
  # fwd_diff_df <- na.omit(forward_difference(vec_dat, step=1))
  # smoothed_vec <- rollapply(fwd_diff_df, 10, mean, align="right", partial=TRUE)
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

  # Residuals
  train_mat <- embed(train_vec, (window_size+1)) # to get true value
  residuals <- train_mat[,1] - normal_error_res$fitted.values
  rfBLT_residual_trend_train_mat[i, 1:length(residuals)] <- residuals

  residual_acf <- acf(residuals, lag.max=50, plot=FALSE)
  df_residuals_acf_rfBLT[i, -1] <- as.vector(residual_acf$acf) 

  rfBLT_coef_list[[i]] <- normal_error_res$posterior_samples
  rfBLT_sigma2_list[[i]] <- normal_error_res$posterior_sigma2
  rfBLT_ess_list[[i]] <- normal_error_res$ess
    
  # Prediction
  rfBLT_pred_trend_list[[i]] <- normal_error_res$future_y_preds
  df_pred_bayes_lasso_taken_normal_error[i, 2:df_ncol] <- revert_fn(x=as.vector(normal_error_res$y_pred), max_org=max_train, min_org=min_train)   
  df_lower_bayes_lasso_taken_normal_error[i, 2:df_ncol] <- revert_fn(x=as.vector(normal_error_res$pred.ci[,1]), max_org=max_train, min_org=min_train)   
  df_upper_bayes_lasso_taken_normal_error[i, 2:df_ncol] <- revert_fn(x=as.vector(normal_error_res$pred.ci[,2]), max_org=max_train, min_org=min_train)
}

## Save results
### ESS plot
pdf("sp500_rfBLT_ess.pdf", width=7, height=5)
par(mfrow=c(1, 1))
for (i in 1:df_nrow){
    plot(rfBLT_ess_list[[i]])
}
dev.off()

### Residuals
pdf("sp500_rfBLT_residuals.pdf", width=10, height=4)
par(mfrow=c(1, 2))
for (i in 1:dim(df_residuals_acf_rfBLT)[1]){
  # ACF of residuals
  plot(df_residuals_acf_rfBLT[i, -1], type="h", ylab="ACF", xlab="Lag", main=paste(i))
  # QQ plot of residuals
  qqnorm(na.omit(rfBLT_residual_trend_train_mat[i,]), main=paste(i))
  qqline(na.omit(rfBLT_residual_trend_train_mat[i,]))
}
dev.off()

### Coefficients of last iter
pdf("sp500_rfBLT_coefs_samples.pdf", width=15, height=20)
par(mfrow=c(6, 3))
for (i in 1:ncol(normal_error_res$posterior_samples)){
  plot(normal_error_res$posterior_samples[,i], type="l", main=paste0("Tract plot of Beta ", i-1), ylab=paste0("Beta ", i-1))
  hist(normal_error_res$posterior_samples[,i], probability = TRUE, main=paste0("Density of Beta ", i-1), xlab=paste0("Beta ", i-1))
  boxplot(normal_error_res$posterior_samples[,i], horizontal=TRUE, main=paste0("Box plot of Beta ", i-1))
}
dev.off()

### Sigma2 of last iter
pdf("sp500_rfBLT_sigma2_samples.pdf", width=15, height=4)
par(mfrow=c(1, 3))
plot(normal_error_res$posterior_sigma2, type="l", ylab="Sigma2", main="Trace plot of Sigma2")
hist(normal_error_res$posterior_sigma2, probability = TRUE, main=paste0("Density of Sigma2"), xlab=paste0("Sigma2"))
boxplot(normal_error_res$posterior_sigma2, horizontal=TRUE, main="Box plot of Sigma2")
dev.off()

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
     rfBL_residual_trend_train_mat,
     df_residuals_acf_rfBL, 
     rfBL_pred_trend_list,
     rfBL_coef_list,
     rfBL_sigma2_list,
     rfBL_ess_list,
     df_pred_bayes_lasso,
     df_lower_bayes_lasso,
     df_upper_bayes_lasso,

     # rfBLT
     rfBLT_residual_trend_train_mat,
     df_residuals_acf_rfBLT, 
     rfBLT_pred_trend_list,
     rfBLT_coef_list,
     rfBLT_sigma2_list,
     rfBLT_ess_list,
     df_pred_bayes_lasso_taken_normal_error,
     df_lower_bayes_lasso_taken_normal_error,
     df_upper_bayes_lasso_taken_normal_error,
     
     file = "SP500_without_LSTM_output_July11.RData")
