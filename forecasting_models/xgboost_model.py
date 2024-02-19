## Xgboost Model

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import STL
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Stop logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Custom Functions
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return result[1]  # p-value

def check_seasonality_acf(series, nlags=40):
    autocorr = acf(series, nlags=nlags)
    return max(autocorr[1:nlags]) > 0.5

def create_features(data):
    """
    Create time series features based on time series index.
    """
    data = data.copy()
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['quarter'] = data.index.quarter
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofyear'] = data.index.dayofyear
    data['dayofmonth'] = data.index.day
    data['weekofyear'] = data.index.isocalendar().week
    return data

def add_lags(data):
    """
    Add lag features to the first column of the DataFrame.
    
    """   
    data['lag1'] = data['y'].shift(12)
    data['lag2'] = data['y'].shift(24)
    data['lag3'] = data['y'].shift(36)
    data['lag4'] = data['y'].shift(48)
    data['lag5'] = data['y'].shift(60)
    data['lag6'] = data['y'].shift(72)
    data['lag7'] = data['y'].shift(84)
    data['lag8'] = data['y'].shift(96)
    data['lag9'] = data['y'].shift(108)
    data['lag10'] = data['y'].shift(120)   
    data['lag11'] = data['y'].shift(132)
    data['lag12'] = data['y'].shift(144)
    data['lag13'] = data['y'].shift(156)
    data['lag14'] = data['y'].shift(168)
    data['lag15'] = data['y'].shift(180)
    data['lag16'] = data['y'].shift(192)
    data['lag17'] = data['y'].shift(204)
    data['lag18'] = data['y'].shift(216)
    data['lag19'] = data['y'].shift(228)
    data['lag20'] = data['y'].shift(240)
    return data


# Main Forecasting Function
def generateForecast(data, date, value, group = None, forecast_horizon=365, frequency='D', train_size=0.8, make_stationary_flag=False, remove_seasonality_flag=False, hyperparameters=None, confidence_level = 0.95):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")
    
    data = data.rename(columns={date: 'date', value: 'y'})
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    if group is not None and group in data.columns:
        data = data[data[group[0]] == group[1]]
        data.drop(group[0], axis=1, inplace=True)

    # Check for stationarity
    p_value = adf_test(data['y'])
    if p_value <= 0.05:
        print("INFO: The time series is stationary. Model results might be reliable.")
    else:
        if make_stationary_flag:
            data['y'] = make_stationary(data['y'])
            print("INFO: Applied differencing to make the series stationary.")
        else:
            print("WARNING: The time series is not stationary. Model results might not be reliable.")

    # Check for seasonality
    if check_seasonality_acf(data['y']):
        print("INFO: Significant seasonality detected in the time series data.")
        if remove_seasonality_flag:
            data['y'] = remove_seasonality(data['y'])
            print("INFO: Seasonality removal applied to the series.")
    else:
        print("INFO: No significant seasonality detected in the time series data.")
    
    # Adding lag features and # Feature Engineering
    data = create_features(data)
    data = add_lags(data)  

    # Feature and target selection
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3']
    TARGET = 'y'

    X = data[FEATURES]
    y = data[TARGET]

    # Splitting data into training and testing
    split_idx = int(len(data) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Set default hyperparameters if none provided
    if hyperparameters is None:
        hyperparameters = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'enable_categorical': 'False'
        }

    # XGBoost model with hyperparameters
    model = xgb.XGBRegressor(**hyperparameters)
    model.fit(X_train, y_train)

    # Predicting on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)

    # Model evaluation metrics
    metrics_dict = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Model Metrics', 'Performance'])

    # Generating future dates
    last_date = data.index.max()

 # Assuming frequency and forecast_horizon are defined
    if frequency == 'H':
        # Start from the next hour
        future_dates = pd.bdate_range(start=last_date, freq=frequency, periods=forecast_horizon) + pd.Timedelta(hours=1)
    elif frequency == 'D':
        # Start from the next day
        future_dates = pd.bdate_range(start=last_date, freq=frequency, periods=forecast_horizon) + pd.Timedelta(days=1)
    elif frequency == 'M':
        # Start from the next month
        future_dates = pd.bdate_range(start=last_date, freq=frequency, periods=forecast_horizon) + pd.DateOffset(months=1)
    elif frequency == 'Y':
        # Start from the next year
        future_dates = pd.bdate_range(start=last_date, freq=frequency, periods=forecast_horizon) + pd.DateOffset(years=1)
    else:
        raise ValueError("Invalid frequency value. Use 'H', 'D', 'M', or 'Y'.")


   # Create a DataFrame for future prediction dates
    future_dates_df = pd.DataFrame(index = future_dates)
    future_dates_df['isFuture'] = True
    data['isFuture'] = False
    data_and_future = pd.concat([data,future_dates_df])

    # 
    data_and_future = create_features(data_and_future)
    data_and_future = add_lags(data_and_future)

    #
    future_w_features = data_and_future.query('isFuture').copy()

    # Forecasting future values
    forecast = model.predict(future_w_features[FEATURES])

    # Calculate prediction intervals
    residuals = y_test - y_pred
    interval = np.quantile(residuals, [0.5 - confidence_level / 2, 0.5 + confidence_level / 2])

    # Calculate prediction intervals for the forecast
    lower_bound = forecast + interval[0]
    upper_bound = forecast + interval[1]

    forecast_df = pd.DataFrame({
        'Date': future_dates, 
        'Forecast': forecast,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })

    return forecast_df, metrics_df
