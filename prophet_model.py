## Standard Prophet Model

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from statsmodels.tsa.stattools import adfuller, acf
from numpy.fft import fft
from itertools import product
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Stop logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Custom functions for time series analysis
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adf_test(series):
    """ Perform Augmented Dickey-Fuller test """
    result = adfuller(series, autolag='AIC')
    return result[1]  # p-value

def make_stationary(series):
    """ Apply differencing to make series stationary """
    return series.diff().dropna()

def check_seasonality_acf(series, nlags=40):
    """ Check seasonality using Autocorrelation Function """
    autocorr = acf(series, nlags=nlags)
    return max(autocorr[1:nlags]) > 0.5

def check_seasonality_fft(series):
    """ Check seasonality using Fourier Transform """
    fft_result = fft(series)
    magnitudes = np.abs(fft_result)
    return np.any(magnitudes > (0.1 * np.max(magnitudes)))

# Main forecasting function
def generateForecast(data, date, value, group=None, include_holidays=False, forecast_horizon=365, frequency='D', train_size=0.8, confidence_level = 0.95, make_stationary_flag=False, remove_seasonality_flag=False, hyperparameters=None):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    data = data.rename(columns={date: 'ds', value: 'y'})

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
    if check_seasonality_acf(data['y']) or check_seasonality_fft(data['y']):
        print("INFO: Significant seasonality detected in the time series data.")
        if remove_seasonality_flag:
            data['y'] = remove_seasonality(data['y'], period=365)
            print("INFO: Seasonality removal applied to the series.")
    else:
        print("INFO: No significant seasonality detected in the time series data.")

    # Hyperparameter tuning setup
    if hyperparameters is not None:
        changepoint_prior_scale = hyperparameters.get('changepoint_prior_scale', [0.001, 0.5])
        seasonality_prior_scale = hyperparameters.get('seasonality_prior_scale', [0.01, 10])
        holidays_prior_scale = hyperparameters.get('holidays_prior_scale', [0.01, 10])
        seasonality_mode = hyperparameters.get('seasonality_mode', ['additive', 'multiplicative'])
    else:
        # Use default Prophet settings if hyperparameters is None
        changepoint_prior_scale = [0.05]
        seasonality_prior_scale = [10.0]
        holidays_prior_scale = [10.0]
        seasonality_mode = ['additive']
    
    # Splitting data into training and testing

    data_train, data_test = np.split(data, [int(train_size *len(data))])

    # Prophet requires the column names to be 'ds' and 'y'
    data_train_prophet = data_train.reset_index().rename(columns={data_train.columns[0]:'ds', data_train.columns[1]:'y'})
    data_test_prophet = data_test.reset_index().rename(columns={data_train.columns[0]:'ds', data_train.columns[1]:'y'})

    # Initialize model performance tracking
    best_model_performance = float('inf')
    best_model = None

    # Hyperparameter tuning loop
    for cps, sps, hps, sm in product(changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, seasonality_mode):
        # Prophet model setup
        model = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps, holidays_prior_scale=hps, seasonality_mode=sm, interval_width=confidence_level)
        # Initialize and fit the Prophet model
        if include_holidays:
            cal = calendar()
            holidays = cal.holidays(start=data.index.min(), end=data.index.max(), return_name=True)
            holiday_df = pd.DataFrame(data=holidays, columns=['holiday']).reset_index().rename(columns={'index':'ds'})
            model = Prophet(holidays=holiday_df, interval_width=confidence_level)
        else:
            model = Prophet()

    model.fit(data_train_prophet)


    # Evaluate the model if test data is available
    if not data_test_prophet.empty:
        data_test_fcst = model.predict(data_test_prophet)
        rmse = np.sqrt(mean_squared_error(y_true=data_test_prophet['y'], y_pred=data_test_fcst['yhat']))
        mae = mean_absolute_error(y_true=data_test_prophet['y'], y_pred=data_test_fcst['yhat'])
        mape = mean_absolute_percentage_error(y_true=data_test_prophet['y'], y_pred=data_test_fcst['yhat'])


    # Track the best model
    if rmse < best_model_performance:
        best_model_performance = rmse
        best_model = model
    

    # Calculate the number of periods from split date to max date + forecast horizon
    #split_date = data_test_prophet['ds'].min()
    #end_date = data_test_prophet['ds'].max() + pd.Timedelta(days=forecast_horizon)
    #multiplier = (end_date - split_date).days

    # Assuming frequency and forecast_horizon are defined
    if frequency == 'H':
        # Start from the next hour
        future_dates = pd.bdate_range(start=data_test_prophet['ds'].max(), freq=frequency, periods=forecast_horizon) + pd.Timedelta(hours=1)
    elif frequency == 'D':
        # Start from the next day
        future_dates = pd.bdate_range(start=data_test_prophet['ds'].max(), freq=frequency, periods=forecast_horizon) + pd.Timedelta(days=1)
    elif frequency == 'M':
        # Start from the next month
        future_dates = pd.bdate_range(start=data_test_prophet['ds'].max(), freq=frequency, periods=forecast_horizon) + pd.DateOffset(months=1)
    elif frequency == 'Y':
        # Start from the next year
        future_dates = pd.bdate_range(start=data_test_prophet['ds'].max(), freq=frequency, periods=forecast_horizon) + pd.DateOffset(years=1)
    else:
        raise ValueError("Invalid frequency value. Use 'H', 'D', 'M', or 'Y'.")


    # Create a DataFrame for future prediction dates
    future_dates_df = pd.DataFrame(future_dates, columns=['ds'])

    # Use calculated multiplier for future predictions
    #future = best_model.make_future_dataframe(periods=forecast_horizon, freq=frequency, include_history=True)
    forecast = best_model.predict(future_dates_df)

    # Renaming columns in the forecast DataFrame
    forecast_renamed = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
    columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'})

    # Model evaluation metrics
    metrics_dict = {
        'RMSE': best_model_performance,
        'MAE': mean_absolute_error(y_true=data_test_prophet['y'], y_pred=data_test_fcst['yhat']),
        'MAPE': mean_absolute_percentage_error(y_true=data_test_prophet['y'], y_pred=data_test_fcst['yhat'])
    }

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Model Metrics', 'Performance'])


    return forecast_renamed, metrics_df
