# Load library 
library(dplyr)
library(zoo)
library(randomForest)
library(forecast)

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

# --- Specify input ---
pred_size = 7
window_size = 7

# ----- New cases ----- 
m1 = 155
m2 = 362
m3 = 490
m4 = 625

filtered_df$numconf_diff_avg <- rollapply(filtered_df$numconf_diff, 7, mean, align="right", partial=TRUE)
filtered_df$index <- 1:nrow(filtered_df)

# Plotting the data with vertical lines for the prediction period
plot(filtered_df$date, filtered_df$numconf_diff_avg, col = "black", lwd = 2, type="l",
     xlab = "Time (days)", ylab="New cases", main = "COVID19 Daily New Cases in Canada", cex=0.5)
# Add vertical lines
abline(v = filtered_df$date[m1], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m2], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m3], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m4], col = "red", lwd = 2, lty = 2)

# Extract second wave 
second_wave_df <- filtered_df %>%
  dplyr::filter(index >= m1 & index < m2) # Adjust the range as needed
# Add an index column starting from 1 
second_wave_df$index <- 1:nrow(second_wave_df)

# --- Create empty dfs to save results ---
start_train_point <- 100
last_train_point <- length(as.vector(second_wave_df$numconf_diff_avg))-pred_size
# Number of rows 
df_nrow <- length(start_train_point:last_train_point)
# Number of columns
df_ncol = 8

# Create empty dataframe for saving pred
df_true <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_pred_rforest <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))

# Assign column names for pred df
colnames(df_true) <- c("training_point", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
colnames(df_pred_rforest) <- c("training_point", "t1", "t2", "t3", "t4", "t5", "t6", "t7")

# Assign values to the training_point column for pred df
df_true$training_point <- start_train_point:last_train_point
df_pred_rforest$training_point <- start_train_point:last_train_point

# --- Execute and save output ---
# True data
for (i in 1:df_nrow){
    vec_dat <- as.vector(second_wave_df$numconf_diff_avg)
    df_true[i, 2:df_ncol] <- vec_dat[(df_true[i, 1]+1):(df_true[i, 1]+7)]
}

# Compute rof
n = nrow(second_wave_df)
second_wave_df$rofc_new_cases <- c(NA, (second_wave_df$numconf_diff_avg[-1]-second_wave_df$numconf_diff_avg[-n])/second_wave_df$numconf_diff_avg[-n])

for (i in 1:df_nrow){    
    index <- df_pred_rforest[i, 1]
    
    # Change input
    x <- na.omit(second_wave_df$rofc_new_cases) # having NA
    train_data <- second_wave_df$numconf_diff_avg[1:index]
    
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

# Save data
infectious_new_case <- second_wave_df$numconf_diff_avg
rofc_new_cases <- second_wave_df$rofc_new_cases
save(infectious_new_case,
     rofc_new_cases,
     df_true,
     df_pred_rforest,
     file = "COVID19_new_case_output_rforest.RData")

# ----- Deaths ----- 
m1 = 155
m2 = 377
m3 = 500
m4 = 650

# Deaths
filtered_df$numdeaths_diff_avg <- rollapply(filtered_df$numdeaths_diff, 7, mean, align="right", partial=TRUE)

# Plotting the data with vertical lines for the prediction period
plot(filtered_df$date, filtered_df$numdeaths_diff_avg, col = "black", lwd = 2, type="l",
     xlab = "Time (days)", ylab="Deaths", main = "COVID19 Daily New Deaths in Canada", cex=0.5)
# Add vertical lines 
abline(v = filtered_df$date[m1], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m2], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m3], col = "red", lwd = 2, lty = 2)
abline(v = filtered_df$date[m4], col = "red", lwd = 2, lty = 2)

# Extract second wave 
second_wave_df <- filtered_df %>%
  dplyr::filter(index >= m1 & index < m2) # Adjust the range as needed
# Add an index column starting from 1 
second_wave_df$index <- 1:nrow(second_wave_df)

# --- Create empty dfs to save results ---
start_train_point <- 100
last_train_point <- length(as.vector(second_wave_df$numdeaths_diff_avg))-pred_size
# Number of rows 
df_nrow <- length(start_train_point:last_train_point)
# Number of columns
df_ncol = 8

# Create empty dataframe for saving pred
df_true <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))
df_pred_rforest <- data.frame(matrix(ncol = df_ncol, nrow = df_nrow))

# Assign column names for pred df
colnames(df_true) <- c("training_point", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
colnames(df_pred_rforest) <- c("training_point", "t1", "t2", "t3", "t4", "t5", "t6", "t7")

# Assign values to the training_point column for pred df
df_true$training_point <- start_train_point:last_train_point
df_pred_rforest$training_point <- start_train_point:last_train_point

# --- Execute and save output ---
# True data
for (i in 1:df_nrow){
    vec_dat <- as.vector(second_wave_df$numdeaths_diff_avg)
    df_true[i, 2:df_ncol] <- vec_dat[(df_true[i, 1]+1):(df_true[i, 1]+7)]
}

# Compute rofc
n = nrow(second_wave_df)
second_wave_df$rofc_deaths <- c(NA, (second_wave_df$numdeaths_diff_avg[-1]-second_wave_df$numdeaths_diff_avg[-n])/second_wave_df$numdeaths_diff_avg[-n])

for (i in 1:df_nrow){
    index <- df_pred_rforest[i, 1]
    
    # Change input
    x <- na.omit(second_wave_df$rofc_deaths) # having NA
    train_data <- second_wave_df$numdeaths_diff_avg[1:index]
    
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

# Save data
death_cases <- second_wave_df$numdeaths_diff_avg
rofc_deaths <- second_wave_df$rofc_deaths
save(death_cases,
     rofc_deaths,
     df_true,
     df_pred_rforest,
     file = "COVID19_deaths_output_rforest.RData")
