# Project title: Stonks

### Project Description:

This project explore the use of machine learning to predict short term stock movements and evaluate whether a model based approach can outperform the s&p 500. Given the fact that short term prediction is typically very noisy and by testing with a model based approach we can evaluate the usefulness of predictive features in financial data.

## Goal:

- Develop a predictive trading model that can achieve higher total returns compared to the S&P 500.
- Successfully be able to predict whether or not a stock will increase or decrease for the next day

## Required Data

- Historical trading data
    - Sources
        - Download data from yahoo finance
    - Fields
        - Open, high, low, close ,volume levels
    - Frequency
        - Daily data from 2000-2026
- Calendar effects
    - Day of week
    - Month of year / Seasons

## Data modeling

- Using a always up predictor
    - Used as a sanity check on to see if our xgboost even works correctly
- Using xgboost to learn and predict the next day price
    - Using trees to learn the error of past trees and develops a pattern spotting

## Data Visualization

- Line chart
    - Show difference in asset price between modal and S&P
- Scatter plot
    - Predicted and actual returns for the model itself
- Heatmap
    - Correlations for classifications and to show relation between inputs

## Test plan

- Strategy
    - Each day, select top-3 predicted “up” stocks and buy them
    - Sell top 3 stocks that are predicted to go down
- Train on historical data
- Benchmarks
    - Compare to buy-and-hold SPY
- Backtest
    - Testing using more recent years (2020-2026)
    - Include transaction cost
    - Only allowed to see past data when testing and not current