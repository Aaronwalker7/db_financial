# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Price Prediction Project

# COMMAND ----------

# MAGIC %md
# MAGIC The aim of this notebook is to build a model that can predict the future price of AAPL's stock for the next day.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Starting Spark Session

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql
import pyspark.pandas
from pyspark.sql.functions import avg
from pyspark.sql import functions as F

# COMMAND ----------

spark = SparkSession \
    .builder \
    .appName("Aaron's Financial Data Session") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS aaronxwalker")

# COMMAND ----------

# MAGIC %md
# MAGIC ## imports

# COMMAND ----------

# MAGIC %pip install yfinance
# MAGIC %pip install fredapi

# COMMAND ----------

import numpy as np
import pandas as pd
import numba
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark import pandas as ps
import datetime
from fredapi import Fred

# COMMAND ----------

fred = Fred(api_key_file='fred_api_key.txt')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extending pandas API

# COMMAND ----------

AAPL.index.day

# COMMAND ----------

@pd.api.extensions.register_dataframe_accessor("date_features")
class DateFeatures:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def get_basic_day_elements(self):
        self._obj['year'] = self._obj.index.year
        self._obj['month'] = self._obj.index.month
        self._obj['dayOfMonth'] = self._obj.index.day
        self._obj['dayOfWeek'] = self._obj.index.to_series().apply(lambda x: x.weekday())


# COMMAND ----------

@pd.api.extensions.register_dataframe_accessor("clean_code")
class CleanCode:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def lower_column_names(self):
        for col in self._obj.columns:
            self._obj.rename(columns = {col:col.lower()}, inplace = True)
        return self._obj

# COMMAND ----------

@pd.api.extensions.register_dataframe_accessor("prices_data")
class PricesData:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def gather_stock_prices(self, stock_list):
        for stock in stock_list:
            stock_data = yf.Ticker(stock).history(start = '2017-01-01')
            stock_data.index = stock_data.index.date
            stock_data = stock_data.stock_data.create_returns(target_column = 'Close', updated_column_name = 'returns', target = False)
            stock_data = stock_data.clean_code.lower_column_names()
            for col in stock_data.columns:
                stock_data.rename(columns = {col: col+'_'+stock}, inplace = True)
            self._obj = self._obj.merge(right = stock_data, left_index = True, right_index = True, how = 'left')
        return self._obj

# COMMAND ----------

@pd.api.extensions.register_dataframe_accessor("stock_data")
class StockData:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def create_returns(self, target_column:str = 'close', updated_column_name:str = 'returns', target: bool = False, log: bool = False):
        """
        Convert price of asset to returns given by price of day t divided by price of day t-1, minus 1. Also lag by a timestep if its the target column. Have the option to create log returns given by log(Rt/Rt-1). Returns the dataframe with the updated target.
        """
        if target:
            self._obj[updated_column_name] = self._obj[target_column]/self._obj[target_column].shift(1)-1
            self._obj[updated_column_name] = self._obj[updated_column_name].shift(-1)
        else:
            self._obj[updated_column_name] = self._obj[target_column]/self._obj[target_column].shift(1)-1
        return self._obj
    
    def convert_index_to_date(self):
        self._obj.index = self._obj.index.date
        return self._obj

# COMMAND ----------

@pd.api.extensions.register_dataframe_accessor("fred_data")
class FredData:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def single_fred_data(self, fred_ticker: str):
        """
        Retrieve data for a single ticker from the St Louis Fed (FREDAPI). Returns the dataframe for the fred ticker.
        """
        fred_ticker_df = fred.get_series_all_releases(fred_ticker, realtime_start=self._obj.index.min().strftime('%Y-%m-%d'), realtime_end=datetime.datetime.now().strftime('%Y-%m-%d'))
        return fred_ticker_df
    
    @staticmethod
    def clean_fred_data(fred_ticker:str, fred_df: pd.DataFrame, tz : str = 'America/New_York'):
        """
        Cleans the data including converting the Datetime column into the local time of the area you want and only keeping the last value when duplicates arise (related to FREDAPI data). Also renames the columns to include ticker as a suffix to differentiate. Returns the fred data except with renamed columns.
        """
        fred_df['realtime_start'] = fred_df['realtime_start'].dt.tz_localize(tz).dt.date
        fred_df.drop_duplicates(subset='realtime_start', keep='last', inplace=True)
        fred_df = fred_df.clean_code.lower_column_names()
        for col in renamed_cols:
            fred_df.rename(columns={col: col + '_' + fred_ticker}, inplace=True)
        return fred_df
    
    def merge_fred_to_stock_data(self, fred_df: pd.DataFrame):
        """
        Returns a merged dataframe of your original data and the fred data.
        """
        self._obj = self._obj.merge(right = fred_df, left_index = True, right_on = 'realtime_start', how = 'left')
        return self._obj
    
    def set_merged_index_to_date(self):
        """
        Returns your original dataframe except with the index being dropped and replaced by the realtime start from the merged fred data.
        """
        self._obj = self._obj.set_index(keys = 'realtime_start', drop = True)
        return self._obj
    
    def clean_all_data(self):
        """
        Forward fills all of the data in the dataframe over missing values. Can be susceptable to missing values from the start with no previous values to forward fill.
        """
        self._obj = self._obj.ffill(axis = 0)
        return self._obj

    def attach_fred_data(self, fred_tickers: list, tz: str = 'America/New_York'):
        """
        Attach macroeconomic data to the StockData object, retrieved from St Louis Fed (FREDAPI).

        Parameters:
        - tickers (list): List of FRED data series tickers.
        - tz (str): Any timezone you want, a list can be found online.
        """
        for fred_ticker in fred_tickers:
            fred_df = self.single_fred_data(fred_ticker)
            fred_df = self.clean_fred_data(fred_ticker, fred_df, tz = tz)
            self._obj = self.merge_fred_to_stock_data(fred_df = fred_df)
            self._obj = self.set_merged_index_to_date()
        self._obj = self.clean_all_data()
        self._obj.index = pd.to_datetime(self._obj.index)
        return self._obj


# COMMAND ----------

# MAGIC %md
# MAGIC ## Curating Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### macroeconomic data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### the following are the tickers for the macro datasets and their name
# MAGIC
# MAGIC T10YIE - 10 year average expected inflation in US
# MAGIC
# MAGIC MORTGAGE30US - 30 year mortgage rate in US
# MAGIC
# MAGIC MORTGAGE15US - 15 year mortgage rate in US
# MAGIC
# MAGIC CPIAUCSL - inflation rate in US
# MAGIC
# MAGIC UNRATE - unemployment rate in US
# MAGIC
# MAGIC GDP - GDP in US
# MAGIC
# MAGIC GDPC1 - real GDP in US
# MAGIC
# MAGIC FEDFUNDS - effective federal funds rate (overnight interbank borrowing) in US
# MAGIC
# MAGIC REAINTRATREARAT10Y - 10 year real interest rate in US (nominal - inflation)
# MAGIC
# MAGIC A792RC0Q052SBEA - personal income per capita in US
# MAGIC
# MAGIC POPTHM - population in US
# MAGIC
# MAGIC GBPUSD=X - GBP to USD conversion rate
# MAGIC
# MAGIC EURUSD=X - EUR to USD conversion rate
# MAGIC
# MAGIC CNY=X - CNY to USD conversion rate
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC get realtime_start to see when it was actually recorded so you know how long to lag it by, for example in the below it has a date of 10th October 2023 for the data (beginning of the quarter) but actually it wasn't available data until 28th February 2024

# COMMAND ----------

ticker = 'AAPL'
start_date = '2017-01-01'

AAPL = yf.Ticker(ticker = ticker).history(start = start_date)
AAPL = AAPL.clean_code.lower_column_names()

# COMMAND ----------

AAPL = AAPL.stock_data.create_returns(target = True)
AAPL = AAPL.stock_data.convert_index_to_date()

# COMMAND ----------

fred_tickers = ['T10YIE', 'CPIAUCSL', 'UNRATE', 'GDP', 'GDPC1', 'FEDFUNDS', 'REAINTRATREARAT10Y', 'MORTGAGE30US', 'MORTGAGE15US', 'A792RC0Q052SBEA']
exchange_rate_tickers = ['GBPUSD=X', 'EURUSD=X', 'CNY=X']
tech_stock_tickers = ['MSFT', 'AMZN', 'TSLA', 'NVDA', 'GOOG', 'META']
index_tickers = ['^FTSE', '^GSPC', '^DJI', '^IXIC', '^GDAXI', '^N225']
renamed_cols = fred.get_series_all_releases('T10YIE', realtime_start='2017-01-01', realtime_end=datetime.datetime.now().strftime('%Y-%m-%d')).columns.drop('realtime_start')

# COMMAND ----------

AAPL = AAPL.fred_data.attach_fred_data(fred_tickers = fred_tickers)
AAPL = AAPL.prices_data.gather_stock_prices(stock_list = exchange_rate_tickers)
AAPL = AAPL.prices_data.gather_stock_prices(stock_list = index_tickers)
AAPL = AAPL.prices_data.gather_stock_prices(stock_list = tech_stock_tickers)

# COMMAND ----------

AAPL

# COMMAND ----------

AAPL.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gathering price data (stock prices, index prices, international index prices)

# COMMAND ----------

stock_list = ['AAPL', 'MSFT', 'TSLA']

# COMMAND ----------

for stock in stock_list:
    stock_data = yf.Ticker(stock).history(start = '2017-01-01')
    stock_data.index = stock_data.index.date
    for col in stock_data.columns:
        stock_data.rename(columns = {col: col+stock}, inplace = True)

# COMMAND ----------

TSLA

# COMMAND ----------

MSFT = yf.Ticker('MSFT').history(start = '2017-01-01')
MSFT.index = MSFT.index.date
for col in MSFT.columns:
    MSFT.rename(columns = {col: col+'_MSFT'}, inplace = True)

# COMMAND ----------

MSFT

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating two targets to see which works better

# COMMAND ----------

AAPL['returns'] = AAPL['Close']/AAPL['Close'].shift(1)-1
# AAPL['log_returns'] = np.log(AAPL['Close']/AAPL['Close'].shift(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lagging all variables so that we dont use same day data that we wouldn't have access to

# COMMAND ----------

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
for feature in features:
    AAPL[feature] = AAPL[feature].shift(1)

# COMMAND ----------

AAPL.rename(columns = {'Stock Splits': 'stock_splits'}, inplace = True)
for i in AAPL.columns:
    AAPL.rename(columns = {i: i.lower()}, inplace = True)

# COMMAND ----------

AAPL.reset_index(inplace = True)

# COMMAND ----------

display(AAPL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning Data

# COMMAND ----------

AAPL.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Val-Test Split

# COMMAND ----------

# MAGIC %md
# MAGIC Here I am splitting the data based on chronological order. The first 70% is to train, the next 15% is to validate and the final 15% is to test on.

# COMMAND ----------

train_samples = int(AAPL.shape[0]*0.7)
val_samples = int(AAPL.shape[0]*0.15)
test_samples = int(AAPL.shape[0]*0.15)

# COMMAND ----------

AAPL_train = AAPL[:train_samples]
AAPL_val = AAPL[train_samples:train_samples + val_samples]
AAPL_test = AAPL[train_samples+val_samples:]

# COMMAND ----------

AAPL_train = spark.createDataFrame(AAPL_train)
AAPL_val = spark.createDataFrame(AAPL_val)
AAPL_test = spark.createDataFrame(AAPL_test)

# COMMAND ----------

display(AAPL_train.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving Data to Table

# COMMAND ----------

AAPL_train.write.option('overwriteSchema','true').mode("overwrite").saveAsTable("aaronxwalker.AAPL_returns_train")
AAPL_val.write.option('overwriteSchema','true').mode("overwrite").saveAsTable("aaronxwalker.AAPL_returns_val")
AAPL_test.write.option('overwriteSchema','true').mode("overwrite").saveAsTable("aaronxwalker.AAPL_returns_test")

# COMMAND ----------

#you can create a new pandas dataframe witht the following command:
AAPL_test = spark.sql('select * from aaronxwalker.aapl_returns_test')

# COMMAND ----------

AAPL_train.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## baseline model (using AutoML)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing results of validation predictions to actual

# COMMAND ----------


 forecast_pred = spark.table('aaronxwalker.forecast_prediction_bca7ef16')

# COMMAND ----------

display(forecast_pred)

# COMMAND ----------

display(AAPL_val)

# COMMAND ----------

display(AAPL_val.agg(avg('returns')).collect()[0][0])

# COMMAND ----------

actual = AAPL_val.withColumnRenamed('returns', 'actual')
actual = actual.withColumnRenamed('Date', 'date2')

# COMMAND ----------

predictions = forecast_pred.join(actual, on = forecast_pred['Date'] == actual['date2'], how = 'inner')

# COMMAND ----------

predictions = predictions.select('Date', 'returns', 'returns_lower', 'returns_upper', 'actual')

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### importing best model

# COMMAND ----------

import mlflow
logged_model = 'runs:/17f389892d1c4cf0abec8975a63deb20/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

AAPL_test.reset_index(inplace= True)

# COMMAND ----------

loaded_model.predict(AAPL_test.iloc[:95])

# COMMAND ----------

results_df = pd.DataFrame({'date':AAPL_test['Date'], 'actual':AAPL_test['returns'], 'predictions': loaded_model.predict(AAPL_test)})

# COMMAND ----------

results_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## selecting a model

# COMMAND ----------

# MAGIC %md
# MAGIC ## hyperparameter tuning (hyperopt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## deploying model

# COMMAND ----------


