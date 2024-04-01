# Store Sales - Time Series Forecasting📈
EDA, Feature Engineering, and Time Series Forecasting on Store Sales Data in Ecuador.

Kaggle Competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting

## Table of Contents
1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Feature Engineering](#feature-engineering)
4. [Time Series Forecasting](#time-series-forecasting)
5. [Conclusion](#conclusion)

## Introduction
This project aims to forecast store sales in Ecuador using **Time Series** forecasting techniques. The transformations are automated using **Data Pipelines**, and the models experimented were tracked with **MLflow**.

## Data Description

Datasets used in this project are:
- **train.csv**: Historical sales data from 2013 to 2017.
- **test.csv**: Sales data to predict.
- **stores.csv**: Stores information.
- **holidays.csv**: Holidays in Ecuador
- **oil.csv**: Oil prices in Ecuador

## Feature Engineering
Applied joins, aggregations, and transformations to create new features, as well as some techniques specific to Time Series data like:
- **Lag Features**: Shifted sales data to create lag features.
- **Rolling Window Statistics**: Calculated rolling mean, median, and standard deviation.

## Time Series Forecasting

Used an XGRegressor model to forecast sales. Hyperparameter tuning was made with TimeSeriesSplit and GridSearchCV. The models were tracked using MLflow.

## Conclusion
The model achieved a **Root Mean Square Error** of **~250 usd** on the test set, which means that the model can predict store sales with an error of +-250 usd on average (a great success, considering sales range from 0 to 10000 usd in some cases).

When submitting to kaggle, the model performed just as well, achieving the same RMSE.
