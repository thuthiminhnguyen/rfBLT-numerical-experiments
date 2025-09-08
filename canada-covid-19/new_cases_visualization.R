# Load library
library(dplyr)
library(ggplot2)
library(robustbase)
library(zoo)
library(patchwork)
library(tidyr)

# Load data
load("D:/Dalhousie University/0. Project/Output/COVID19 v3/New cases/COVID19_new_cases_output_July11.RData")

# --- Functions ---
# Compute percentage of coverage and their range
coverage_fn <- function(dat_vec, df_true, df_pred, df_upper, df_lower, start_index, end_index) {
  # Percentage of Coverage
  prop_cover = colMeans(df_upper >= df_true & df_lower <= df_true)
  # Range of Coverage
  cr_range_mat = df_upper - df_lower
  return(list(prop_cover=prop_cover, cr_range_mat = cr_range_mat))
}

# Generate a dataframe for predictions
create_forecast_df <- function(dat_vec, df_true, df_pred, df_lower, df_upper, start_index, end_index) {
  df_nrow <- nrow(df_true)
  df_ncol <- ncol(df_true)
  n_dat <- length(dat_vec)
  
  # Initialize data frame
  data <- data.frame(
    train_time = numeric(),
    forecast_day = numeric(),
    horizon = numeric(),
    predicted_mean = numeric(),
    lower_95 = numeric(),
    upper_95 = numeric(),
    observed = numeric()
  )
  
  # Loop through each row of df_true
  for (i in 1:df_nrow) {
    train_time <- df_true[i, 1]
    for (j in 2:df_ncol) {
      horizon <- j - 1
      forecast_day <- train_time + horizon
      if (forecast_day <= end_index) {
        new_row <- data.frame(
          train_time = train_time,
          forecast_day = forecast_day,
          horizon = horizon,
          predicted_mean = df_pred[i, j],
          lower_95 = df_lower[i, j],
          upper_95 = df_upper[i, j],
          observed = dat_vec[forecast_day]
        )
        data <- rbind(data, new_row)
      }
    }
  }
  
  return(data)
}

# Mean directional average
DA <- function(pred, true, past){
  return((pred-past)*(true-past)>0)
}

MDA <- function(pred_mat, true_mat, past_vec, h){
  if((dim(pred_mat)[1]!=dim(true_mat)[1])|(dim(pred_mat)[2]!=dim(true_mat)[2])) stop("Unmatch dimension of pred_mat and true_mat")
  if(nrow(pred_mat)!=length(past_vec)) stop("Unmatch nrow of pred_mat")
  if(ncol(true_mat) < h) stop("h is greater than ncol of true_mat")
  
  pred_vec <- pred_mat[,h]
  true_vec <- true_mat[,h]
  
  return(mean(DA(pred=pred_vec, true=true_vec, past=past_vec)))
}

# Relative error
relative_error <- function(df_true, df_pred){
  mse <- (df_true - df_pred)^2
  numerator <- rowSums(mse)
  denominator <- rowSums(df_true^2)
  error <- sqrt(numerator/denominator)
  return(error)
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

canada_df = na.omit(canada_df)

# Filter the dataframe
filtered_df <- canada_df %>%
  dplyr::filter(date >= as.Date("2020-03-12"))

filtered_df$numconf_diff_avg <- rollapply(filtered_df$numconf_diff, 7, mean, align="right", partial=TRUE)
filtered_df$index <- 1:nrow(filtered_df)

m1 = 155
m2 = 362
m3 = 490
m4 = 625

# Plotting the data with vertical lines for the prediction period
all_waves <- ggplot(filtered_df, aes(x = date, y = numconf_diff_avg)) +
  geom_line(color = "black", size = 1) +
  geom_vline(xintercept = filtered_df$date[m1], color = "red", linetype = "dashed") +
  geom_vline(xintercept = filtered_df$date[m2], color = "red", linetype = "dashed") +
  geom_vline(xintercept = filtered_df$date[m3], color = "red", linetype = "dashed") +
  geom_vline(xintercept = filtered_df$date[m4], color = "red", linetype = "dashed") +
  labs(x = "Date", y = "Cases") + # title = "COVID-19 Daily New Cases in Canada",
  theme_minimal() +
  theme(plot.title = element_text(size = 11),
        axis.title.x = element_text(size = 9),
        axis.title.y = element_text(size = 9))

all_waves # show visualization
ggsave("all_waves_new_cases_plot.pdf", plot = all_waves, width = 6, height = 3, units = "in")

# --- Visualize the second wave of new cases ---
second_wave_df <- filtered_df %>%
  dplyr::filter(index >= m1 & index < m2)

m <- 100
n <- length(infectious_new_case)

past_vec <- numeric(nrow(df_true))
for (i in 1:nrow(df_true)){
  past_vec[i] <- infectious_new_case[df_true[i,1]]
}

# Data frame for vertical lines
vline_data <- data.frame(
  xintercept = c(m, 112, 120, 135, 151, 189),
  type = c("First Training End", "Uptrend", "Peak", "Trough",
           "Peak", "Trough")
)
vline_data$type <- factor(vline_data$type, levels = c("First Training End", "Uptrend", "Peak", "Trough"))

p_2ndwave <- ggplot(second_wave_df, aes(x = seq_along(numconf_diff_avg), y = numconf_diff_avg)) +
  geom_line(color = "black", size = 1) +
  geom_vline(data = vline_data,
             aes(xintercept = xintercept, color = type, linetype = type),
             size = 0.8, show.legend = TRUE) +
  labs(title = "Second wave - New Deaths",
       x = "Day", y = "Deaths",
       color = NULL, linetype = NULL) +   # <-- removes legend titles
  theme_minimal() +
  theme(
    plot.title = element_text(size = 11),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.title = element_blank(),       # <-- ensures no legend title
    legend.text = element_text(size = 8)
  ) +
  scale_color_manual(values = c("First Training End" = "blue", "Trough" = "gray", "Peak" = "gray", "Uptrend" = "gray")) +
  scale_linetype_manual(values = c("First Training End" = "dashed", "Trough" = "dashed", "Peak" = "solid", "Uptrend" = "dotted")) +
  scale_x_continuous(
    breaks = vline_data$xintercept   # show ticks only at vertical line positions
  ) 

p_2ndwave # show visualization

# --- Random Feature Bayesian Lasso Takens (rfBLT) ---
# Compute coverage percentage and range
res_BLT_normal_error <- coverage_fn(
  dat_vec = infectious_new_case,
  df_true = df_true,
  df_pred = df_pred_bayes_lasso_taken_normal_error,
  df_upper = df_upper_bayes_lasso_taken_normal_error,
  df_lower = df_lower_bayes_lasso_taken_normal_error,
  start_index = 100,
  end_index = n
)

# Compute MDA
MDA_BLT_normal_error <- numeric(ncol(df_true[,-1]))
for (i in 1:ncol(df_true[,-1])){
  MDA_BLT_normal_error[i] <- MDA(pred_mat=df_pred_bayes_lasso_taken_normal_error[,-1], 
                                 true_mat=df_true[, -1], 
                                 past_vec=past_vec, 
                                 h = i)
}

# Create a forecast dataframe
BLT_normal_output_df <- create_forecast_df(
  dat_vec = infectious_new_case, 
  df_true = df_true, 
  df_pred = df_pred_bayes_lasso_taken_normal_error, 
  df_lower = df_lower_bayes_lasso_taken_normal_error, 
  df_upper = df_upper_bayes_lasso_taken_normal_error, 
  start_index = 100, 
  end_index = n)

# Visualize the expanding window forecasting
p1 <- ggplot(BLT_normal_output_df, aes(x = forecast_day)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95, group = train_time, fill = "95% CI"),
              alpha = 0.1) +
  geom_line(aes(y = predicted_mean, group = train_time, color = "Predicted Mean"), 
            alpha = 2) +
  geom_line(aes(y = observed, color = "Observed"), size = 1, alpha = 0.7) +
  labs(title = "Random Feature Bayesian Lasso Taken",
       x = "Day", y = "Cases",
       color = "Legend", fill = "Legend") +
  scale_color_manual(values = c("Predicted Mean" = "red", "Observed" = "black")) +
  scale_fill_manual(values = c("95% CI" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top", 
        legend.title = element_blank(),
        plot.title = element_text(size = 11),
        axis.title.x = element_text(size = 9),
        axis.title.y = element_text(size = 9)) +
  coord_cartesian(ylim = c(2000, 10000))

# --- Random Feature Bayesian Lasso (rfBL) ---
# Compute coverage percentage and range
res_bayes_lasso <- coverage_fn(
  dat_vec = infectious_new_case,
  df_true = df_true,
  df_pred = df_pred_bayes_lasso,
  df_upper = df_upper_bayes_lasso,
  df_lower = df_lower_bayes_lasso,
  start_index = 100,
  end_index = n
)

# Compute MDA 
MDA_bayes_lasso <- numeric(ncol(df_true[,-1]))
for (i in 1:ncol(df_true[,-1])){
  MDA_bayes_lasso[i] <- MDA(pred_mat=df_pred_bayes_lasso[,-1], 
                            true_mat=df_true[, -1], 
                            past_vec=past_vec, 
                            h = i)
}

# Create a forecast dataframe
bayes_lasso_output_df <- create_forecast_df(
  dat_vec = infectious_new_case, 
  df_true = df_true, 
  df_pred = df_pred_bayes_lasso, 
  df_lower = df_lower_bayes_lasso, 
  df_upper = df_upper_bayes_lasso, 
  start_index = 100, 
  end_index = n)

# Visualize the expanding window forecasting
p2 <- ggplot(bayes_lasso_output_df, aes(x = forecast_day)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95, group = train_time, fill = "95% CI"),
              alpha = 0.1) +
  geom_line(aes(y = predicted_mean, group = train_time, color = "Predicted Mean"), 
            alpha = 2) +
  geom_line(aes(y = observed, color = "Observed"), size = 1, alpha = 0.7) +
  labs(title = "Random Feature Bayes Lasso",
       x = "Day", y = "Cases",
       color = "Legend", fill = "Legend") +
  scale_color_manual(values = c("Predicted Mean" = "red", "Observed" = "black")) +
  scale_fill_manual(values = c("95% CI" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top", 
        legend.title = element_blank(),
        plot.title = element_text(size = 11),
        axis.title.x = element_text(size = 9),
        axis.title.y = element_text(size = 9)) +
  coord_cartesian(ylim = c(2000, 10000))

# --- Random Forest ---
# Load outputs
load("D:/Dalhousie University/0. Project/Output/COVID19 v3/New cases/COVID19_new_case_output_rforest_July11.RData")

df_nrow = nrow(df_true)
df_ncol = ncol(df_true)
start_index = 100
end_index = length(infectious_new_case)
pred_mat = matrix(NA, nrow = df_nrow, ncol = (end_index - start_index + 1))

# Compute MDA
MDA_rforest <- numeric(ncol(df_true[,-1]))
for (i in 1:ncol(df_true[,-1])){
  MDA_rforest[i] <- MDA(pred_mat=df_pred_rforest[,-1], 
                        true_mat=df_true[, -1], 
                        past_vec=past_vec, 
                        h = i)
}

# Create a forecast dataframe
rforest_output_df <- create_forecast_df(
  dat_vec = infectious_new_case, 
  df_true = df_true, 
  df_pred = df_pred_rforest, 
  df_lower = matrix(rep(NA, df_nrow*df_ncol), nrow=df_nrow, ncol=df_ncol), 
  df_upper = matrix(rep(NA, df_nrow*df_ncol), nrow=df_nrow, ncol=df_ncol), 
  start_index = 100, 
  end_index = n)

# Visualize the expanding window forecasting
p3 <- ggplot(rforest_output_df, aes(x = forecast_day)) +
  geom_line(aes(y = predicted_mean, group = train_time, color = "Predicted Mean"), 
            alpha = 2) +
  geom_line(aes(y = observed, color = "Observed"), size = 1, alpha = 0.7) +
  labs(title = "Random Forest",
       x = "Day", y = "Cases",
       color = "Legend", fill = "Legend") +
  scale_color_manual(values = c("Predicted Mean" = "red", "Observed" = "black")) +
  theme_minimal() +
  theme(legend.position = "top", 
        legend.title = element_blank(),
        plot.title = element_text(size = 11),
        axis.title.x = element_text(size = 9),
        axis.title.y = element_text(size = 9)) +
  coord_cartesian(ylim = c(2000, 10000))

# --- ARIMA ---
# Compute coverage percentage and range
res_arima <- coverage_fn(
  dat_vec = infectious_new_case,
  df_true = df_true,
  df_pred = df_pred_arima,
  df_upper = df_upper_arima,
  df_lower = df_lower_arima,
  start_index = 100,
  end_index = n
)

# Compute MDA
MDA_arima <- numeric(ncol(df_true[,-1]))
for (i in 1:ncol(df_true[,-1])){
  MDA_arima[i] <- MDA(pred_mat=df_pred_arima[,-1], 
                      true_mat=df_true[, -1], 
                      past_vec=past_vec, 
                      h = i)
}

# Create a forecast dataframe
arima_output_df <- create_forecast_df(
  dat_vec = infectious_new_case, 
  df_true = df_true, 
  df_pred = df_pred_arima, 
  df_lower = df_lower_arima, 
  df_upper = df_upper_arima, 
  start_index = 100, 
  end_index = n)

# Visualize the expanding window forecasting
p4 <- ggplot(arima_output_df, aes(x = forecast_day)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95, group = train_time, fill = "95% CI"),
              alpha = 0.1) +
  geom_line(aes(y = predicted_mean, group = train_time, color = "Predicted Mean"), 
            alpha = 2) +
  geom_line(aes(y = observed, color = "Observed"), size = 1, alpha = 0.7) +
  labs(title = "ARIMA",
       x = "Date", y = "Cases",
       color = "Legend", fill = "Legend") +
  scale_color_manual(values = c("Predicted Mean" = "red", "Observed" = "black")) +
  scale_fill_manual(values = c("95% CI" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top", 
        legend.title = element_blank(),
        plot.title = element_text(size = 11),
        axis.title.x = element_text(size = 9),
        axis.title.y = element_text(size = 9)) +
  coord_cartesian(ylim = c(2000, 10000))

# --- Holt ---
# Compute coverage percentage and range
res_holt <- coverage_fn(
  dat_vec = infectious_new_case,
  df_true = df_true,
  df_pred = df_pred_holt,
  df_upper = df_upper_holt,
  df_lower = df_lower_holt,
  start_index = 100,
  end_index = n
)

# Compute MDA
MDA_holt <- numeric(ncol(df_true[,-1]))
for (i in 1:ncol(df_true[,-1])){
  MDA_holt[i] <- MDA(pred_mat=df_pred_holt[,-1], 
                     true_mat=df_true[, -1], 
                     past_vec=past_vec, 
                     h = i)
}

# Create a forecast dataframe
holt_output_df <- create_forecast_df(
  dat_vec = infectious_new_case, 
  df_true = df_true, 
  df_pred = df_pred_holt, 
  df_lower = df_lower_holt, 
  df_upper = df_upper_holt, 
  start_index = 100, 
  end_index = n)

# Visualize the expanding window forecasting
p5 <- ggplot(holt_output_df, aes(x = forecast_day)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95, group = train_time, fill = "95% CI"),
              alpha = 0.1) +
  geom_line(aes(y = predicted_mean, group = train_time, color = "Predicted Mean"), 
            alpha = 2) +
  geom_line(aes(y = observed, color = "Observed"), size = 1, alpha = 0.7) +
  labs(title = "Holt's Linear Trend",
       x = "Day", y = "Cases",
       color = "Legend", fill = "Legend") +
  scale_color_manual(values = c("Predicted Mean" = "red", "Observed" = "black")) +
  scale_fill_manual(values = c("95% CI" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top", 
        legend.title = element_blank(),
        plot.title = element_text(size = 11),
        axis.title.x = element_text(size = 9),
        axis.title.y = element_text(size = 9)) +
  coord_cartesian(ylim = c(2000, 10000))

# --- LSTM ---
# Load outputs
load("D:/Dalhousie University/0. Project/Output/COVID19 v3/New cases/COVID19_new_case_output_lstm_July11.RData")

df_nrow = nrow(df_true)
df_ncol = ncol(df_true)
start_index = 100
end_index = length(infectious_new_case)
pred_mat = matrix(NA, nrow = df_nrow, ncol = (end_index - start_index + 1))

# Compute MDA
MDA_lstm <- numeric(ncol(df_true[,-1]))
for (i in 1:ncol(df_true[,-1])){
  MDA_lstm[i] <- MDA(pred_mat=df_pred_lstm[,-1], 
                     true_mat=df_true[, -1], 
                     past_vec=past_vec, 
                     h = i)
}

# Create a forecast dataframe
lstm_output_df <- create_forecast_df(
  dat_vec = infectious_new_case, 
  df_true = df_true, 
  df_pred = df_pred_lstm, 
  df_lower = matrix(rep(NA, df_nrow*df_ncol), nrow=df_nrow, ncol=df_ncol), 
  df_upper = matrix(rep(NA, df_nrow*df_ncol), nrow=df_nrow, ncol=df_ncol), 
  start_index = 100, 
  end_index = n)

# Visualize training loss
for (i in 1:nrow(loss_lstm_mat)){
  if (i == 1){
    plot(loss_lstm_mat[i,], type="l", xlab="epoch", ylab="loss", 
         ylim=c(min(loss_lstm_mat), max(loss_lstm_mat)), 
         main="LSTM model train loss")
  }
  else {
    lines(loss_lstm_mat[i,])
  }
}

# Visualize the expanding window forecasting
p6 <- ggplot(lstm_output_df, aes(x = forecast_day)) +
  geom_line(aes(y = predicted_mean, group = train_time, color = "Predicted Mean"), 
            alpha = 2) +
  geom_line(aes(y = observed, color = "Observed"), size = 1, alpha = 0.7) +
  labs(title = "LSTM",
       x = "Day", y = "Cases",
       color = "Legend", fill = "Legend") +
  scale_color_manual(values = c("Predicted Mean" = "red", "Observed" = "black")) +
  theme_minimal() +
  theme(legend.position = "top", 
        legend.title = element_blank(),
        plot.title = element_text(size = 11),
        axis.title.x = element_text(size = 9),
        axis.title.y = element_text(size = 9)) +
  coord_cartesian(ylim = c(2000, 10000))

# --- Visualize all predictions and observations ---
final_plot <- p_2ndwave/(p1|p2|p3)/(p4|p5|p6)
pred_plot <- final_plot + 
  plot_annotation(
    # title = "Expanding Window 7-Day Ahead Forecasts with Confidence/Credible Intervals",
    theme = theme(plot.title = element_text(size = 20, hjust = 0.5))
  )
pred_plot # show the plot
ggsave("new_cases_pred_plot.pdf", plot = pred_plot, width = 12, height = 10, units = "in")

# --- Visualize MDA for all methods ---
MDA_df <- data.frame(
  Normal_error = MDA_BLT_normal_error,
  Bayes_Lasso = MDA_bayes_lasso,
  ARIMA = MDA_arima,
  Holt = MDA_holt,
  RForest = MDA_rforest,
  LSTM = MDA_lstm
)
# Observe MDA values
t(round(MDA_df, 4))

# ARIMA
mda_arima_df <- as.data.frame(t(MDA_arima))
colnames(mda_arima_df) <- paste0("t", 1:7)
arima_mda_long <- mda_arima_df %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "MDA") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# Holt
mda_holt_df <- as.data.frame(t(MDA_holt))
colnames(mda_holt_df) <- paste0("t", 1:7)
holt_mda_long <- mda_holt_df %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "MDA") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# rfBL
mda_bayes_lasso_df <- as.data.frame(t(MDA_bayes_lasso))
colnames(mda_bayes_lasso_df) <- paste0("t", 1:7)
bayes_lasso_mda_long <- mda_bayes_lasso_df %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "MDA") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# rfBLT
mda_BLT_normal_error_df <- as.data.frame(t(MDA_BLT_normal_error))
colnames(mda_BLT_normal_error_df) <- paste0("t", 1:7)
BLT_normal_error_mda_long <- mda_BLT_normal_error_df %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "MDA") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# Random Forest
mda_rforest_df <- as.data.frame(t(MDA_rforest))
colnames(mda_rforest_df) <- paste0("t", 1:7)
rforest_mda_long <- mda_rforest_df %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "MDA") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "RForest")

# LSTM
mda_lstm_df <- as.data.frame(t(MDA_lstm))
colnames(mda_lstm_df) <- paste0("t", 1:7)
lstm_mda_long <- mda_lstm_df %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "MDA") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "LSTM")

# Combine all methods
combine_mda_df <- rbind(
  BLT_normal_error_mda_long,
  bayes_lasso_mda_long,
  arima_mda_long,
  holt_mda_long,
  rforest_mda_long,
  lstm_mda_long
)

combine_mda_df$Method <- factor(combine_mda_df$Method, levels = unique(combine_mda_df$Method))

MDA_plot <- ggplot(combine_mda_df, aes(x = Day, y = MDA, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = "Prediction horizon, h (days)",
    y = "MDA"
  ) +
  theme_minimal(base_size = 12) +
  theme(# legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2")

MDA_plot # show the plot
ggsave("new_cases_MDA_plot.pdf", plot = MDA_plot, width = 9, height = 5, units = "in")

# --- Visualize relative error for all methods ---
# Compute relative error
arima_relative_error <- relative_error(df_true=df_true[,-1], df_pred=df_pred_arima[,-1])
holt_relative_error <- relative_error(df_true=df_true[,-1], df_pred=df_pred_holt[,-1])
lstm_relative_error <- relative_error(df_true=df_true[,-1], df_pred=df_pred_lstm[,-1])
BL_relative_error <- relative_error(df_true=df_true[,-1], df_pred=df_pred_bayes_lasso[,-1])
rforest_relative_error <- relative_error(df_true=df_true[,-1], df_pred=df_pred_rforest[,-1])
BLT_normal_relative_error <- relative_error(df_true=df_true[,-1], df_pred=df_pred_bayes_lasso_taken_normal_error[,-1])

# Create a dataframe
relative_error_df <- data.frame(
  Normal_error = BLT_normal_relative_error,
  Bayes_Lasso = BL_relative_error,
  ARIMA = arima_relative_error,
  Holt = holt_relative_error,
  RForest = rforest_relative_error,
  LSTM = lstm_relative_error
)

# Reshape to long format
relative_error_long <- pivot_longer(relative_error_df, 
                               cols = everything(),
                               names_to = "Method",
                               values_to = "relative_error")

# Rename methods to match the desired labels
method_labels <- c("rfBLT", "rfBL", "ARIMA", "Holt", "RForest", "LSTM")
names(method_labels) <- names(relative_error_df)
relative_error_long$Method <- method_labels[relative_error_long$Method]
relative_error_long$Method <- factor(relative_error_long$Method, levels = unique(relative_error_long$Method))

# Create ggplot
relative_error_plot <- ggplot(relative_error_long, aes(x = Method, y = relative_error, fill = Method)) +
  geom_boxplot() +  
  labs(x = "Methods",
       y = "Relative error") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12)) + 
  scale_fill_brewer(palette = "Set2") 

relative_error_plot # show the plot
ggsave("new_cases_relative_error_plot.pdf", plot = relative_error_plot, width = 5, height = 5, units = "in")

# --- Coverage of statistical models ---
# - Percentage -
cover_percent_df <- data.frame(
  rfBLT = res_BLT_normal_error$prop_cover[-1],
  rfBL = res_bayes_lasso$prop_cover[-1],
  ARIMA = res_arima$prop_cover[-1],
  Holt = res_holt$prop_cover[-1]
)
t(round(cover_percent_df*100, 2))

# - Median Range -
cover_median_range_df <- data.frame(
  rfBLT = colMedians(as.matrix(res_BLT_normal_error$cr_range_mat[, 2:df_ncol])),
  rfBL = colMedians(as.matrix(res_bayes_lasso$cr_range_mat[, 2:df_ncol])),
  ARIMA = colMedians(as.matrix(res_arima$cr_range_mat[, 2:df_ncol])),
  Holt = colMedians(as.matrix(res_holt$cr_range_mat[, 2:df_ncol]))
)
t(round(cover_median_range_df, 1))
