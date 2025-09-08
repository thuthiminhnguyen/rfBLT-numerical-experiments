# rfBLT-numerical-experiments
This repository provides the R code to reproduce the experiments in the paper of *rfBLT: Random Feature Bayesian Lasso Takens Model for time series forecasting*. 

## Installation
Please install the `rfBLT` package from this repository [link](https://github.com/thuthiminhnguyen/rfBLT). 

## Code Structure
- `S&P500/`: Experiments on S&P500 index.
  - `sp500_lstm.R`: Implement LSTM.
  - `sp500_random_forest.R`: Implement Random Forest.
  - `sp500_stat_models.R`: Implement statistical models (ARIMA, Holt, rfBL, rfBLT).
- `canada-covid-19/`: Contains Canada COVID-19 data and experiments for new cases and deaths in Canada.
  -  `covid19-download.csv`: Daily data from March 2020 to January 2022.
  -  `new_cases_lstm.R`: Implement LSTM for new cases.
  -  `new_cases_stat_models.R`: Implement statistical models (ARIMA, Holt, rfBL, rfBLT) for new cases.
  -  `new_deaths_lstm.R`: Implement LSTM for new fatalities.
  -  `new_deaths_stat_models.R`: Implement statistical models (ARIMA, Holt, rfBL, rfBLT) for new fatalities.
  -  `random_forests.R`: Implement Random Forest for new cases and fatalities. 
