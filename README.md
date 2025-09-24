# rfBLT-numerical-experiments
This repository provides the R code to reproduce the experiments in the paper of *rfBLT: Random Feature Bayesian Lasso Takens Model for time series forecasting*. 

## Installation
Please install the `rfBLT` package from this repository [link](https://github.com/thuthiminhnguyen/rfBLT). 

## Code Structure
- `S&P500/`: Experiments on the S&P500 index from January 15, 2022, to May 31, 2025.
  - `sp500_lstm.R`: Implement LSTM.
  - `sp500_random_forest.R`: Implement Random Forest.
  - `sp500_stat_models.R`: Implement statistical models (ARIMA, Holt, rfBL, rfBLT).
  - `sp500_visualization.R`: Visualize the results of all models using the closing value of S&P500 index.
- `SmuEIR simulation - 10% noise level/`: Experiments on 100 simulations from the SmuEIR model at 5 distinct training periods.
  - `all_lstm.R`: Implement LSTM.
  - `all_random_forest.R`: Implement Random Forest.
  - `training_period_102.R`: Implement statistical models (ARIMA, Holt, rfBL, rfBLT) at the training period of 102 days.
  - `training_period_108.R`: Implement statistical models at the training period of 108 days.
  - `training_period_114.R`: Implement statistical models at the training period of 114 days.
  - `training_period_125.R`: Implement statistical models at the training period of 125 days.
  - `training_period_85.R`: Implement statistical models at the training period of 85 days.
  - `visualization.R`: Visualize the results at all training periods. 
- `canada-covid-19/`: Contains Canada COVID-19 data and experiments for new cases and deaths in Canada.
  -  `covid19-download.csv`: Daily data from March 2020 to January 2022.
  -  `new_cases_lstm.R`: Implement LSTM for new cases.
  -  `new_cases_stat_models.R`: Implement statistical models (ARIMA, Holt, rfBL, rfBLT) for new cases.
  -  `new_cases_visualization.R`: Visualize the results of all models using new cases data.
  -  `new_deaths_lstm.R`: Implement LSTM for new fatalities.
  -  `new_deaths_stat_models.R`: Implement statistical models (ARIMA, Holt, rfBL, rfBLT) for new fatalities.
  -  `new_deaths_visualization.R`: Visualize the results of all models using new deaths data.
  -  `random_forests.R`: Implement Random Forest for new cases and fatalities. 
