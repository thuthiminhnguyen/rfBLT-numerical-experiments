# Load library
library(dplyr)
library(zoo)
library(randomForest)
library(forecast)
library(quantmod)

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

df_pred_rforest <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
colnames(df_pred_rforest) <- c("training_point", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
df_pred_rforest$training_point <- start_train_point:last_train_point

# --- Execute and add output to empty dfs ---
# True data
for (i in 1:df_nrow){
    vec_dat <- as.vector(prices)
    df_true[i, 2:df_ncol] <- vec_dat[(df_true[i, 1]+1):(df_true[i, 1]+7)]
}

# Calculate rate of change
n = nrow(df)
df$rofc <- c(NA, (as.numeric(df$GSPC.Close)[-1]-as.numeric(df$GSPC.Close)[-n])/as.numeric(df$GSPC.Close)[-n])
# Visualize rate of change
plot(df$rofc, type="l", xlab="Day", ylab="Log difference")

# Random Forest 
for (i in 1:df_nrow){
    index <- df_pred_rforest[i, 1]
    
    # Change input
    x <- na.omit(df$rofc) # having NA
    train_data <- as.numeric(df$GSPC.Close)[1:index]
    
    res_scale_train <- normalization_fn(x[1:(index-1)])
    scale_train <- res_scale_train$norm_vec
    max_train <- res_scale_train$max_x
    min_train <- res_scale_train$min_x

    train_mat <- embed(scale_train, window_size+1)
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

    pred_rofc <- revert_fn(x=forecasts_rf, max_org=max_train, min_org=min_train)
    y_pred_vec <- numeric(pred_size)
    past <- tail(train_data, 1)

    # rocf[t] <- (x[t]-x[t-1])/x[t-1]
    # x[t] <- x[t-1]*rocf[t]+x[t-1]
    for (j in 1:pred_size){
        y_pred_vec[j] <- past*pred_rofc[j]+past
        past <- y_pred_vec[j]
    }
    df_pred_rforest[i, 2:df_ncol] <- y_pred_vec
}
