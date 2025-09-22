# Project title: Stonks

### Project Description:

This project explores the use of machine learning to predict short term stock movements and evaluate whether a model based approach can outperform the s&p 500. Because financial markets is very noisy especially in the short term time frame, our goal is to test whether or not incorporating technical signals and calendar effects can improve next-day prediction. By including these features we can evaluate the usefulness of predictive pattern in financial data.

The initial analysis will start off with a subset of 10 companies from the fortune 500 list. This set will be later expanded to include additional stocks for greater diversity (i.e cover all different business sectors). We plan to use supervised learning methods (i.e classifications and regression) to predict whether a stock will move up or down on the next day and make buy and sell decisions based on the prediction. 

## Goal:

- Develop a predictive trading model that can achieve higher total returns compared to the S&P 500.
- Successfully be able to predict whether or not a stock will increase or decrease for the next day
- Identify which technical patterns are most predictive of short-term price movements

## Required Data

- Historical trading data
    - Sources
        - Download data from yahoo finance
    - Fields
        - Open, high, low, close ,volume levels
    - Frequency
        - Daily data from 2000-2026
    - Universe
        - 10 companies from the fortune 500
        - s&p500 (SPY/VOO)
- Calendar effects
    - Day of week
    - Month of year / Seasons

## Data modeling

- Using a always up predictor
    - Used as a sanity check on to see if our xgboost even works correctly
- Using xgboost to learn and predict the next day direction
    - Using trees to learn the error of past trees and develops a spotting

## Data Visualization

- Line chart
    - Show difference in asset price between model and S&P
- Scatter plot
    - Predicted and actual returns for the model itself
- Heatmap
    - Correlations for classifications and to show relation between inputs (i.e RSI, volume, calendar effects)
- Confusion matrix
    - shows the model performance (i.e how often it correctly predicts if the stock is going to go up or down)

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