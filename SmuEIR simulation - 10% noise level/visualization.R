# --- Load library ---
library(robustbase)
library(zoo)
library(stats)
library(forecast)
library(ggplot2)
library(RColorBrewer)
library(tidyr) 
library(reshape2)
library(matrixStats)
library(dplyr)
library(patchwork)

# --- Functions ---
# Relative error
relative_error <- function(df_true, df_pred){
  mse <- (df_true - df_pred)^2
  numerator <- rowSums(mse)
  denominator <- rowSums(df_true^2)
  error <- sqrt(numerator/denominator)
  return(error)
}

# Visualize all training points in the same plot
all_pred_plot_simulation <- function(points = c(85, 102, 108, 114, 125),
                                     title = NULL,
                                     true_mat,
                                     pred_mat = list(),
                                     upperCI_mat = list(),
                                     lowerCI_mat = list(),
                                     start_index,
                                     end_index) {
  
  pred_size <- 7
  n <- length(colMedians(true_mat))
  
  # Prepare true data for plotting
  true_data <- data.frame(
    day = (start_index:end_index)-1,
    infectious_proportion = colMedians(true_mat)[start_index:end_index]
  )
  
  # Initialize ggplot object
  p <- ggplot() +
    geom_line(data = true_data, aes(x = day, y = infectious_proportion), color = "black") +
    labs(title = title, x = "Day", y = "Proportion") +
    ylim(0.02, 0.34) +
    theme_minimal() +
    theme(legend.position = "top", 
          legend.title = element_blank(),
          plot.title = element_text(size = 11),
          axis.title.x = element_text(size = 9),
          axis.title.y = element_text(size = 9)) 
  
  # Prepare prediction data
  pred_data_list <- list()
  for (i in seq_along(points)) {
    pred_start <- points[i] + 1
    pred_end <- points[i] + pred_size
    x_range <- pred_start:pred_end
    
    pred_data <- data.frame(
      day = x_range-1,
      pred = colMedians(pred_mat[[i]]),
      group = i  # To distinguish different prediction segments
    )
    
    if (!is.null(upperCI_mat) && !is.null(lowerCI_mat)) {
      pred_data$upper <- colMedians(upperCI_mat[[i]])
      pred_data$lower <- colMedians(lowerCI_mat[[i]])
    }
    
    pred_data_list[[i]] <- pred_data
  }
  
  # Combine all prediction data
  pred_data_combined <- do.call(rbind, pred_data_list)
  
  # Add prediction lines
  p <- p + geom_line(data = pred_data_combined, aes(x = day, y = pred, group = group), 
                     color = "blue", size = 1)
  
  # Add confidence intervals if available
  if (!is.null(upperCI_mat) && !is.null(lowerCI_mat)) {
    p <- p + geom_ribbon(data = pred_data_combined, 
                         aes(x = day, ymin = lower, ymax = upper, group = group), 
                         fill = "blue", alpha = 0.2)
  }
  
  return(p)
}

# --- Load outputs ---
# LSTM
load("D:/Dalhousie University/0. Project/Output/Simulation 0.1 v4/simulation0.1_lstm_July17.RData")
# Random Forest
load("D:/Dalhousie University/0. Project/Output/Simulation 0.1 v4/simulation0.1_rforest_July22.RData")

# --- Visualize data ---
n_days <- ncol(I_noise_avg_mat)   

# Convert matrix to long-format data frame
I_noise_long <- melt(I_noise_avg_mat, varnames = c("Simulation", "Day"), value.name = "Infectious")
I_noise_long$Day <- as.numeric(gsub("X", "", I_noise_long$Day-1)) 
I_noise_long$Simulation <- factor(I_noise_long$Simulation)

colors_hcl <- hcl(h = seq(0, 360, length.out = 100), c = 70, l = 60)  
colors_spectral <- colorRampPalette(brewer.pal(11, "Spectral"))(100)

# Create ggplot2 plot
simulation_plot <- ggplot(I_noise_long, aes(x = Day, y = Infectious, group = Simulation, color = Simulation)) +
  geom_line(size = 0.8) +  
  labs(x = "Day",
       y = "Proportion") +
  scale_color_manual(values = colors_spectral, guide = "none") +
  geom_vline(xintercept = c(85, 102, 108, 114, 125), color = "blue", linetype = "dashed") + 
  theme_minimal() +
  theme(plot.title = element_text(size = 11),
        axis.title = element_text(size = 9),
        panel.background = element_rect(fill = "white", color = NA))+  # White panel background
  scale_x_continuous(
    breaks = c(85, 102, 108, 114, 125)   # show ticks only at vertical line positions
  ) 

simulation_plot # show plot
ggsave("simulation0.1_plot.pdf", plot = simulation_plot, width = 9, height = 5, units = "in")

# --- Visualize training loss of LSTM ---
# m = 85
for (i in 1:nrow(m85_loss_mat)){
    if (i==1){
        plot(m85_loss_mat[i, ], type="l", xlab="Epoch", ylab="Loss", main="m = 85")
    } else {
        lines(m85_loss_mat[i, ])
    }
}

# m = 102
for (i in 1:nrow(m102_loss_mat)){
    if (i==1){
        plot(m102_loss_mat[i, ], type="l", xlab="Epoch", ylab="Loss", main="m = 102")
    } else {
        lines(m102_loss_mat[i, ])
    }
}

# m = 108
for (i in 1:nrow(m108_loss_mat)){
    if (i==1){
        plot(m108_loss_mat[i, ], type="l", xlab="Epoch", ylab="Loss", main="m = 108")
    } else {
        lines(m108_loss_mat[i, ])
    }
}

# m = 114
for (i in 1:nrow(m114_loss_mat)){
    if (i==1){
        plot(m114_loss_mat[i, ], type="l", xlab="Epoch", ylab="Loss", main="m = 114")
    } else {
        lines(m114_loss_mat[i, ])
    }
}

# m = 125
for (i in 1:nrow(m125_loss_mat)){
    if (i==1){
        plot(m125_loss_mat[i, ], type="l", xlab="Epoch", ylab="Loss", main="m = 125")
    } else {
        lines(m125_loss_mat[i, ])
    }
}

# --- Tranining days = 85 days ---
# ---- Load model outputs ----
load("D:/Dalhousie University/0. Project/Output/Simulation 0.1 v4/simulation0.1_m85_July17.RData")
m85_y_true_mat_test <- y_true_mat

# ---- Compute relative error ----
arima_relative_error <- relative_error(df_true=y_true_mat, df_pred=m85_arima_pred_avg_noise)
holt_relative_error <- relative_error(df_true=y_true_mat, df_pred=m85_holt_pred_avg_noise)
lstm_relative_error <- relative_error(df_true=y_true_mat, df_pred=m85_pred_lstm_mat)
BL_relative_error <- relative_error(df_true=y_true_mat, df_pred=m85_pred_bayes_lasso)
rforest_relative_error <- relative_error(df_true=y_true_mat, df_pred=m85_pred_rforest)
BLT_normal_relative_error <- relative_error(df_true=y_true_mat, df_pred=m85_pred_BLT_normal_error)

# Create relative_error_df
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
m85_relative_error_plot <- ggplot(relative_error_long, aes(x = Method, y = relative_error, fill = Method)) +
  geom_boxplot() +  
  labs(title = "Days 0–84",
       x = "Methods",
       y = "Relative error") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12)) + 
  scale_fill_brewer(palette = "Set2") +
  coord_cartesian(ylim = c(0, 5))

# ---- Coverage proportion ----
# ----- ARIMA -----
arima_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m85_arima_lowerCI_avg_noise
                                              & m85_arima_upperCI_avg_noise>=y_true_mat))*100))
colnames(arima_prop_cover) <- paste0("t", 1:7)
arima_cover_long <- as.data.frame(arima_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m85_holt_lowerCI_avg_noise
                                             & m85_holt_upperCI_avg_noise>=y_true_mat))*100))
colnames(holt_prop_cover) <- paste0("t", 1:7)
holt_cover_long <- as.data.frame(holt_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m85_lowerCI_bayes_lasso
                                                    & m85_upperCI_bayes_lasso>=y_true_mat))*100))
colnames(bayes_lasso_prop_cover) <- paste0("t", 1:7)
bayes_lasso_cover_long <- as.data.frame(bayes_lasso_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m85_lowerCI_normal_error
                                                     & m85_upperCI_normal_error>=y_true_mat))*100))
colnames(normal_error_prop_cover) <- paste0("t", 1:7)
BLT_normal_error_cover_long <- as.data.frame(normal_error_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cover_df <- rbind(BLT_normal_error_cover_long,
                          bayes_lasso_cover_long, 
                          arima_cover_long, 
                          holt_cover_long)

combine_cover_df$Method <- factor(combine_cover_df$Method, levels = unique(combine_cover_df$Method))

cover_plot_m85 <- ggplot(combine_cover_df, aes(x = Day, y = Coverage, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–84",
    x = "Day",
    y = "Coverage (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# ---- Coverage median range ----
# ----- ARIMA -----
arima_range = as.data.frame(t(as.matrix(colMedians(m85_arima_upperCI_avg_noise-m85_arima_lowerCI_avg_noise))))
colnames(arima_range) <- paste0("t", 1:7)

arima_cr_range_long <- arima_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_range = as.data.frame(t(as.matrix(colMedians(m85_holt_upperCI_avg_noise-m85_holt_lowerCI_avg_noise))))
colnames(holt_range) <- paste0("t", 1:7)

holt_cr_range_long <- holt_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_range = as.data.frame(t(as.matrix(colMedians(m85_upperCI_bayes_lasso-m85_lowerCI_bayes_lasso))))
colnames(bayes_lasso_range) <- paste0("t", 1:7)

bayes_lasso_cr_range_long <- bayes_lasso_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_range = as.data.frame(t(as.matrix(colMedians(m85_upperCI_normal_error-m85_lowerCI_normal_error))))
colnames(normal_error_range) <- paste0("t", 1:7)

BLT_normal_error_cr_range_long <- normal_error_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cr_range_df <- rbind(
  BLT_normal_error_cr_range_long,
  bayes_lasso_cr_range_long,
  arima_cr_range_long,
  holt_cr_range_long
)

combine_cr_range_df$Method <- factor(combine_cr_range_df$Method, levels = unique(combine_cr_range_df$Method))

range_plot_m85 <- ggplot(combine_cr_range_df, aes(x = Day, y = Range, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–84",
    x = "Day",
    y = "Range"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") + 
  guides(color = guide_legend(nrow = 2))

# --- Tranining days = 102 days ---
# ---- Load model outputs ----
load("D:/Dalhousie University/0. Project/Output/Simulation 0.1 v4/simulation0.1_m102_July17.RData")
m102_y_true_mat_test <- y_true_mat

# ---- Compute relative error ----
arima_relative_error <- relative_error(df_true=y_true_mat, df_pred=m102_arima_pred_avg_noise)
holt_relative_error <- relative_error(df_true=y_true_mat, df_pred=m102_holt_pred_avg_noise)
lstm_relative_error <- relative_error(df_true=y_true_mat, df_pred=m102_pred_lstm_mat)
BL_relative_error <- relative_error(df_true=y_true_mat, df_pred=m102_pred_bayes_lasso)
rforest_relative_error <- relative_error(df_true=y_true_mat, df_pred=m102_pred_rforest)
BLT_normal_relative_error <- relative_error(df_true=y_true_mat, df_pred=m102_pred_BLT_normal_error)

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
m102_relative_error_plot <- ggplot(relative_error_long, aes(x = Method, y = relative_error, fill = Method)) +
  geom_boxplot() +  
  labs(title = "Days 0–101",
       x = "Methods",
       y = "Relative error") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12)) + 
  scale_fill_brewer(palette = "Set2") +
  coord_cartesian(ylim = c(0, 1))

# ---- Coverage proportion ----
# ----- ARIMA -----
arima_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m102_arima_lowerCI_avg_noise
                                              & m102_arima_upperCI_avg_noise>=y_true_mat))*100))
colnames(arima_prop_cover) <- paste0("t", 1:7)
arima_cover_long <- as.data.frame(arima_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m102_holt_lowerCI_avg_noise
                                             & m102_holt_upperCI_avg_noise>=y_true_mat))*100))
colnames(holt_prop_cover) <- paste0("t", 1:7)
holt_cover_long <- as.data.frame(holt_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m102_lowerCI_bayes_lasso
                                                    & m102_upperCI_bayes_lasso>=y_true_mat))*100))
colnames(bayes_lasso_prop_cover) <- paste0("t", 1:7)
bayes_lasso_cover_long <- as.data.frame(bayes_lasso_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m102_lowerCI_normal_error
                                                     & m102_upperCI_normal_error>=y_true_mat))*100))
colnames(normal_error_prop_cover) <- paste0("t", 1:7)
BLT_normal_error_cover_long <- as.data.frame(normal_error_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cover_df <- rbind(BLT_normal_error_cover_long,
                          bayes_lasso_cover_long, 
                          arima_cover_long, 
                          holt_cover_long)

combine_cover_df$Method <- factor(combine_cover_df$Method, levels = unique(combine_cover_df$Method))

cover_plot_m102 <- ggplot(combine_cover_df, aes(x = Day, y = Coverage, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–101",
    x = "Day",
    y = "Coverage (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") + 
  guides(color = guide_legend(nrow = 2))

# ---- Coverage median range ----
# ----- ARIMA -----
arima_range = as.data.frame(t(as.matrix(colMedians(m102_arima_upperCI_avg_noise-m102_arima_lowerCI_avg_noise))))
colnames(arima_range) <- paste0("t", 1:7)

arima_cr_range_long <- arima_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_range = as.data.frame(t(as.matrix(colMedians(m102_holt_upperCI_avg_noise-m102_holt_lowerCI_avg_noise))))
colnames(holt_range) <- paste0("t", 1:7)

holt_cr_range_long <- holt_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_range = as.data.frame(t(as.matrix(colMedians(m102_upperCI_bayes_lasso-m102_lowerCI_bayes_lasso))))
colnames(bayes_lasso_range) <- paste0("t", 1:7)

bayes_lasso_cr_range_long <- bayes_lasso_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_range = as.data.frame(t(as.matrix(colMedians(m102_upperCI_normal_error-m102_lowerCI_normal_error))))
colnames(normal_error_range) <- paste0("t", 1:7)

BLT_normal_error_cr_range_long <- normal_error_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cr_range_df <- rbind(
  BLT_normal_error_cr_range_long,
  bayes_lasso_cr_range_long,
  arima_cr_range_long,
  holt_cr_range_long
)

combine_cr_range_df$Method <- factor(combine_cr_range_df$Method, levels = unique(combine_cr_range_df$Method))

range_plot_m102 <- ggplot(combine_cr_range_df, aes(x = Day, y = Range, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–101",
    x = "Day",
    y = "Range"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# --- Tranining days = 108 days ---
# ---- Load model outputs ----
load("D:/Dalhousie University/0. Project/Output/Simulation 0.1 v4/simulation0.1_m108_July17.RData")
m108_y_true_mat_test <- y_true_mat

# ---- Compute relative error ----
arima_relative_error <- relative_error(df_true=y_true_mat, df_pred=m108_arima_pred_avg_noise)
holt_relative_error <- relative_error(df_true=y_true_mat, df_pred=m108_holt_pred_avg_noise)
lstm_relative_error <- relative_error(df_true=y_true_mat, df_pred=m108_pred_lstm_mat)
BL_relative_error <- relative_error(df_true=y_true_mat, df_pred=m108_pred_bayes_lasso)
rforest_relative_error <- relative_error(df_true=y_true_mat, df_pred=m108_pred_rforest)
BLT_normal_relative_error <- relative_error(df_true=y_true_mat, df_pred=m108_pred_BLT_normal_error)

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
m108_relative_error_plot <- ggplot(relative_error_long, aes(x = Method, y = relative_error, fill = Method)) +
  geom_boxplot() +  
  labs(title = "Days 0–107",
       x = "Methods",
       y = "Relative error") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12)) + 
  scale_fill_brewer(palette = "Set2") +
  coord_cartesian(ylim = c(0, 1))

# ---- Coverage proportion ----
# ----- ARIMA -----
arima_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m108_arima_lowerCI_avg_noise
                                              & m108_arima_upperCI_avg_noise>=y_true_mat))*100))
colnames(arima_prop_cover) <- paste0("t", 1:7)
arima_cover_long <- as.data.frame(arima_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m108_holt_lowerCI_avg_noise
                                             & m108_holt_upperCI_avg_noise>=y_true_mat))*100))
colnames(holt_prop_cover) <- paste0("t", 1:7)
holt_cover_long <- as.data.frame(holt_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m108_lowerCI_bayes_lasso
                                                    & m108_upperCI_bayes_lasso>=y_true_mat))*100))
colnames(bayes_lasso_prop_cover) <- paste0("t", 1:7)
bayes_lasso_cover_long <- as.data.frame(bayes_lasso_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m108_lowerCI_normal_error
                                                     & m108_upperCI_normal_error>=y_true_mat))*100))
colnames(normal_error_prop_cover) <- paste0("t", 1:7)
BLT_normal_error_cover_long <- as.data.frame(normal_error_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

combine_cover_df <- rbind(BLT_normal_error_cover_long,
                          bayes_lasso_cover_long, 
                          arima_cover_long, 
                          holt_cover_long)

combine_cover_df$Method <- factor(combine_cover_df$Method, levels = unique(combine_cover_df$Method))

cover_plot_m108 <- ggplot(combine_cover_df, aes(x = Day, y = Coverage, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–107",
    x = "Day",
    y = "Coverage (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# ---- Coverage median range ----
# ----- ARIMA -----
arima_range = as.data.frame(t(as.matrix(colMedians(m108_arima_upperCI_avg_noise-m108_arima_lowerCI_avg_noise))))
colnames(arima_range) <- paste0("t", 1:7)

arima_cr_range_long <- arima_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_range = as.data.frame(t(as.matrix(colMedians(m108_holt_upperCI_avg_noise-m108_holt_lowerCI_avg_noise))))
colnames(holt_range) <- paste0("t", 1:7)

holt_cr_range_long <- holt_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_range = as.data.frame(t(as.matrix(colMedians(m108_upperCI_bayes_lasso-m108_lowerCI_bayes_lasso))))
colnames(bayes_lasso_range) <- paste0("t", 1:7)

bayes_lasso_cr_range_long <- bayes_lasso_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_range = as.data.frame(t(as.matrix(colMedians(m108_upperCI_normal_error-m108_lowerCI_normal_error))))
colnames(normal_error_range) <- paste0("t", 1:7)

BLT_normal_error_cr_range_long <- normal_error_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cr_range_df <- rbind(
  BLT_normal_error_cr_range_long,
  bayes_lasso_cr_range_long,
  arima_cr_range_long,
  holt_cr_range_long
)

combine_cr_range_df$Method <- factor(combine_cr_range_df$Method, levels = unique(combine_cr_range_df$Method))

range_plot_m108 <- ggplot(combine_cr_range_df, aes(x = Day, y = Range, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–107",
    x = "Day",
    y = "Range"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# --- Tranining days = 114 days ---
# ---- Load model outputs ----
load("D:/Dalhousie University/0. Project/Output/Simulation 0.1 v4/simulation0.1_m114_July17.RData")
m114_y_true_mat_test <- y_true_mat

# ---- Compute relative error ----
arima_relative_error <- relative_error(df_true=y_true_mat, df_pred=m114_arima_pred_avg_noise)
holt_relative_error <- relative_error(df_true=y_true_mat, df_pred=m114_holt_pred_avg_noise)
lstm_relative_error <- relative_error(df_true=y_true_mat, df_pred=m114_pred_lstm_mat)
BL_relative_error <- relative_error(df_true=y_true_mat, df_pred=m114_pred_bayes_lasso)
rforest_relative_error <- relative_error(df_true=y_true_mat, df_pred=m114_pred_rforest)
BLT_normal_relative_error <- relative_error(df_true=y_true_mat, df_pred=m114_pred_BLT_normal_error)

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
m114_relative_error_plot <- ggplot(relative_error_long, aes(x = Method, y = relative_error, fill = Method)) +
  geom_boxplot() +  
  labs(title = "Days 0–113",
       x = "Methods",
       y = "Relative error") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12)) + 
  scale_fill_brewer(palette = "Set2") +
  coord_cartesian(ylim = c(0, 1))

# ---- Coverage proportion ----
# ----- ARIMA -----
arima_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m114_arima_lowerCI_avg_noise
                                              & m114_arima_upperCI_avg_noise>=y_true_mat))*100))
colnames(arima_prop_cover) <- paste0("t", 1:7)
arima_cover_long <- as.data.frame(arima_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m114_holt_lowerCI_avg_noise
                                             & m114_holt_upperCI_avg_noise>=y_true_mat))*100))
colnames(holt_prop_cover) <- paste0("t", 1:7)
holt_cover_long <- as.data.frame(holt_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m114_lowerCI_bayes_lasso
                                                    & m114_upperCI_bayes_lasso>=y_true_mat))*100))
colnames(bayes_lasso_prop_cover) <- paste0("t", 1:7)
bayes_lasso_cover_long <- as.data.frame(bayes_lasso_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT ----- 
normal_error_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m114_lowerCI_normal_error
                                                     & m114_upperCI_normal_error>=y_true_mat))*100))
colnames(normal_error_prop_cover) <- paste0("t", 1:7)
BLT_normal_error_cover_long <- as.data.frame(normal_error_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cover_df <- rbind(BLT_normal_error_cover_long,
                          bayes_lasso_cover_long, 
                          arima_cover_long, 
                          holt_cover_long)

combine_cover_df$Method <- factor(combine_cover_df$Method, levels = unique(combine_cover_df$Method))

cover_plot_m114 <- ggplot(combine_cover_df, aes(x = Day, y = Coverage, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–113",
    x = "Day",
    y = "Coverage (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# ---- Coverage median range ----
# ----- ARIMA -----
arima_range = as.data.frame(t(as.matrix(colMedians(m114_arima_upperCI_avg_noise-m114_arima_lowerCI_avg_noise))))
colnames(arima_range) <- paste0("t", 1:7)

arima_cr_range_long <- arima_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_range = as.data.frame(t(as.matrix(colMedians(m114_holt_upperCI_avg_noise-m114_holt_lowerCI_avg_noise))))
colnames(holt_range) <- paste0("t", 1:7)

holt_cr_range_long <- holt_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_range = as.data.frame(t(as.matrix(colMedians(m114_upperCI_bayes_lasso-m114_lowerCI_bayes_lasso))))
colnames(bayes_lasso_range) <- paste0("t", 1:7)

bayes_lasso_cr_range_long <- bayes_lasso_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_range = as.data.frame(t(as.matrix(colMedians(m114_upperCI_normal_error-m114_lowerCI_normal_error))))
colnames(normal_error_range) <- paste0("t", 1:7)

BLT_normal_error_cr_range_long <- normal_error_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cr_range_df <- rbind(
  BLT_normal_error_cr_range_long,
  bayes_lasso_cr_range_long,
  arima_cr_range_long,
  holt_cr_range_long
)

combine_cr_range_df$Method <- factor(combine_cr_range_df$Method, levels = unique(combine_cr_range_df$Method))

range_plot_m114 <- ggplot(combine_cr_range_df, aes(x = Day, y = Range, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–113",
    x = "Day",
    y = "Range"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# --- Tranining days = 125 days ---
# ---- Load model outputs ----
load("D:/Dalhousie University/0. Project/Output/Simulation 0.1 v4/simulation0.1_m125_July17.RData")
m125_y_true_mat_test <- y_true_mat

# ---- Compute relative error ----
arima_relative_error <- relative_error(df_true=y_true_mat, df_pred=m125_arima_pred_avg_noise)
holt_relative_error <- relative_error(df_true=y_true_mat, df_pred=m125_holt_pred_avg_noise)
lstm_relative_error <- relative_error(df_true=y_true_mat, df_pred=m125_pred_lstm_mat)
BL_relative_error <- relative_error(df_true=y_true_mat, df_pred=m125_pred_bayes_lasso)
rforest_relative_error <- relative_error(df_true=y_true_mat, df_pred=m125_pred_rforest)
BLT_normal_relative_error <- relative_error(df_true=y_true_mat, df_pred=m125_pred_BLT_normal_error)

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
m125_relative_error_plot <- ggplot(relative_error_long, aes(x = Method, y = relative_error, fill = Method)) +
  geom_boxplot() +  
  labs(title = "Days 0–124",
       x = "Methods",
       y = "Relative error") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12)) + 
  scale_fill_brewer(palette = "Set2") +
  coord_cartesian(ylim = c(0, 1))

# ---- Coverage proportion ----
# ----- ARIMA -----
arima_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m125_arima_lowerCI_avg_noise
                                              & m125_arima_upperCI_avg_noise>=y_true_mat))*100))
colnames(arima_prop_cover) <- paste0("t", 1:7)
arima_cover_long <- as.data.frame(arima_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m125_holt_lowerCI_avg_noise
                                             & m125_holt_upperCI_avg_noise>=y_true_mat))*100))
colnames(holt_prop_cover) <- paste0("t", 1:7)
holt_cover_long <- as.data.frame(holt_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m125_lowerCI_bayes_lasso
                                                    & m125_upperCI_bayes_lasso>=y_true_mat))*100))
colnames(bayes_lasso_prop_cover) <- paste0("t", 1:7)
bayes_lasso_cover_long <- as.data.frame(bayes_lasso_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_prop_cover <- as.data.frame(t(as.matrix(colMeans(y_true_mat>=m125_lowerCI_normal_error
                                                     & m125_upperCI_normal_error>=y_true_mat))*100))
colnames(normal_error_prop_cover) <- paste0("t", 1:7)
BLT_normal_error_cover_long <- as.data.frame(normal_error_prop_cover) %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Coverage") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cover_df <- rbind(BLT_normal_error_cover_long,
                          bayes_lasso_cover_long, 
                          arima_cover_long, 
                          holt_cover_long)

combine_cover_df$Method <- factor(combine_cover_df$Method, levels = unique(combine_cover_df$Method))

cover_plot_m125 <- ggplot(combine_cover_df, aes(x = Day, y = Coverage, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–124",
    x = "Day",
    y = "Coverage (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# ---- Coverage median range ----
# ----- ARIMA -----
arima_range = as.data.frame(t(as.matrix(colMedians(m125_arima_upperCI_avg_noise-m125_arima_lowerCI_avg_noise))))
colnames(arima_range) <- paste0("t", 1:7)

arima_cr_range_long <- arima_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "ARIMA")

# ----- Holt -----
holt_range = as.data.frame(t(as.matrix(colMedians(m125_holt_upperCI_avg_noise-m125_holt_lowerCI_avg_noise))))
colnames(holt_range) <- paste0("t", 1:7)

holt_cr_range_long <- holt_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "Holt")

# ----- rfBL -----
bayes_lasso_range = as.data.frame(t(as.matrix(colMedians(m125_upperCI_bayes_lasso-m125_lowerCI_bayes_lasso))))
colnames(bayes_lasso_range) <- paste0("t", 1:7)

bayes_lasso_cr_range_long <- bayes_lasso_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBL")

# ----- rfBLT -----
normal_error_range = as.data.frame(t(as.matrix(colMedians(m125_upperCI_normal_error-m125_lowerCI_normal_error))))
colnames(normal_error_range) <- paste0("t", 1:7)

BLT_normal_error_cr_range_long <- normal_error_range %>%
  pivot_longer(cols = starts_with("t"),
               names_to = "Day",
               values_to = "Range") %>%
  mutate(Day = as.numeric(gsub("t", "", Day)),
         Method = "rfBLT")

# ----- Combine all 4 statistical methods -----
combine_cr_range_df <- rbind(
  BLT_normal_error_cr_range_long,
  bayes_lasso_cr_range_long,
  arima_cr_range_long,
  holt_cr_range_long
)

combine_cr_range_df$Method <- factor(combine_cr_range_df$Method, levels = unique(combine_cr_range_df$Method))

range_plot_m125 <- ggplot(combine_cr_range_df, aes(x = Day, y = Range, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Days 0–124",
    x = "Day",
    y = "Range"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        legend.title = element_blank()) + 
  scale_fill_brewer(palette = "Set2") +
  guides(color = guide_legend(nrow = 2))

# --- Visualize all training points of 6 methods ---
# ---- Predictions ----
# ----- ARIMA -----
arima_all_pred_plot = all_pred_plot_simulation(
  points = c(85, 102, 108, 114, 125),
  title = "ARIMA",
  true_mat = I_noise_avg_mat,
  pred_mat = list(
    m85_arima_pred_avg_noise,
    m102_arima_pred_avg_noise,
    m108_arima_pred_avg_noise,
    m114_arima_pred_avg_noise,
    m125_arima_pred_avg_noise
  ),
  upperCI_mat = list(
    m85_arima_upperCI_avg_noise,
    m102_arima_upperCI_avg_noise,
    m108_arima_upperCI_avg_noise,
    m114_arima_upperCI_avg_noise,
    m125_arima_upperCI_avg_noise
  ),
  lowerCI_mat = list(
    m85_arima_lowerCI_avg_noise,
    m102_arima_lowerCI_avg_noise,
    m108_arima_lowerCI_avg_noise,
    m114_arima_lowerCI_avg_noise,
    m125_arima_lowerCI_avg_noise
  ),
  start_index = 81,
  end_index = 136
)

# ----- Holt -----
holt_all_pred_plot = all_pred_plot_simulation(
  points = c(85, 102, 108, 114, 125),
  title = "Holt's Linear Trend",
  true_mat = I_noise_avg_mat,
  pred_mat = list(
    m85_holt_pred_avg_noise,
    m102_holt_pred_avg_noise,
    m108_holt_pred_avg_noise,
    m114_holt_pred_avg_noise,
    m125_holt_pred_avg_noise
  ),
  upperCI_mat = list(
    m85_holt_upperCI_avg_noise,
    m102_holt_upperCI_avg_noise,
    m108_holt_upperCI_avg_noise,
    m114_holt_upperCI_avg_noise,
    m125_holt_upperCI_avg_noise
  ),
  lowerCI_mat = list(
    m85_holt_lowerCI_avg_noise,
    m102_holt_lowerCI_avg_noise,
    m108_holt_lowerCI_avg_noise,
    m114_holt_lowerCI_avg_noise,
    m125_holt_lowerCI_avg_noise
  ),
  start_index = 81,
  end_index = 136
)

# ----- LSTM -----
lstm_all_pred_plot = all_pred_plot_simulation(
  points = c(85, 102, 108, 114, 125),
  title = "LSTM",
  true_mat = I_noise_avg_mat,
  pred_mat = list(
    m85_pred_lstm_mat,
    m102_pred_lstm_mat,
    m108_pred_lstm_mat,
    m114_pred_lstm_mat,
    m125_pred_lstm_mat
  ),
  upperCI_mat = NULL,
  lowerCI_mat = NULL,
  start_index = 81,
  end_index = 136
)

# ----- rfBL -----
bayes_lasso_all_pred_plot = all_pred_plot_simulation(
  points = c(85, 102, 108, 114, 125),
  title = "rfBL",
  true_mat = I_noise_avg_mat,
  pred_mat = list(
    m85_pred_bayes_lasso,
    m102_pred_bayes_lasso,
    m108_pred_bayes_lasso,
    m114_pred_bayes_lasso,
    m125_pred_bayes_lasso
  ),
  upperCI_mat = list(
    m85_upperCI_bayes_lasso,
    m102_upperCI_bayes_lasso,
    m108_upperCI_bayes_lasso,
    m114_upperCI_bayes_lasso,
    m125_upperCI_bayes_lasso
  ),
  lowerCI_mat = list(
    m85_lowerCI_bayes_lasso,
    m102_lowerCI_bayes_lasso,
    m108_lowerCI_bayes_lasso,
    m114_lowerCI_bayes_lasso,
    m125_lowerCI_bayes_lasso
  ),
  start_index = 81,
  end_index = 136
)

# ----- Random Forest -----
rforest_all_pred_plot = all_pred_plot_simulation(
  points = c(85, 102, 108, 114, 125),
  title = "Random Forest",
  true_mat = I_noise_avg_mat,
  pred_mat = list(
    m85_pred_rforest,
    m102_pred_rforest,
    m108_pred_rforest,
    m114_pred_rforest,
    m125_pred_rforest
  ),
  upperCI_mat = NULL,
  lowerCI_mat = NULL,
  start_index = 81,
  end_index = 136
)

# ----- rfBLT -----
normal_error_all_pred_plot = all_pred_plot_simulation(
  points = c(85, 102, 108, 114, 125),
  title = "rfBLT",
  true_mat = I_noise_avg_mat,
  pred_mat = list(
    m85_pred_BLT_normal_error,
    m102_pred_BLT_normal_error,
    m108_pred_BLT_normal_error,
    m114_pred_BLT_normal_error,
    m125_pred_BLT_normal_error
  ),
  upperCI_mat = list(
    m85_upperCI_normal_error,
    m102_upperCI_normal_error,
    m108_upperCI_normal_error,
    m114_upperCI_normal_error,
    m125_upperCI_normal_error
  ),
  lowerCI_mat = list(
    m85_lowerCI_normal_error,
    m102_lowerCI_normal_error,
    m108_lowerCI_normal_error,
    m114_lowerCI_normal_error,
    m125_lowerCI_normal_error
  ),
  start_index = 81,
  end_index = 136
)

# ----- Put everything in the same plot -----
pred_plot <- (normal_error_all_pred_plot | bayes_lasso_all_pred_plot | rforest_all_pred_plot) / (arima_all_pred_plot | holt_all_pred_plot | lstm_all_pred_plot)

final_pred_plot <- pred_plot + 
  plot_annotation(
    theme = theme(plot.title = element_text(size = 20, hjust = 0.5))
  )

final_pred_plot # show plot
ggsave("simulation0.1_pred_plot.pdf", plot = pred_plot, width = 9, height = 7, units = "in")

# ---- Coverage Proportion ----
cover_all_plot <- (cover_plot_m85 | cover_plot_m102 | cover_plot_m108 | cover_plot_m114 | cover_plot_m125)
cover_all_plot <- cover_all_plot + 
  plot_annotation(
    theme = theme(plot.title = element_text(size = 20, hjust = 0.5))
  )
cover_all_plot # show plot

# ---- Coverage Median Range ----
range_all_plot <- (range_plot_m85 | range_plot_m102 | range_plot_m108 | range_plot_m114 | range_plot_m125)
range_all_plot <- range_all_plot + 
  plot_annotation(
    theme = theme(plot.title = element_text(size = 20, hjust = 0.5))
  )
range_all_plot # show plot
