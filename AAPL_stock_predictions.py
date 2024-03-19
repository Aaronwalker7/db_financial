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

# dbutils.library.restartPython()

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
# MAGIC ## importing AAPL stock data

# COMMAND ----------

# MAGIC %md
# MAGIC My target is:
# MAGIC - the returns on day t+1
# MAGIC
# MAGIC I want to start with the obvious features to use:
# MAGIC - lagged variables:
# MAGIC   - returns
# MAGIC   - volume
# MAGIC   - open - close
# MAGIC   - trailing volume
# MAGIC   - trailing returns
# MAGIC   - price/earnings
# MAGIC - interest rate
# MAGIC - days till earnings report
# MAGIC - month of year
# MAGIC - week of year
# MAGIC - day of week
# MAGIC - inflation
# MAGIC
# MAGIC To be introduced at a later date:
# MAGIC - pass
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selecting Stock

# COMMAND ----------

class StockData(pd.DataFrame):
    def __init__(self, stock_ticker: str, stock_start_date: str):
        """
        Initialize StockData object. This will 

        Parameters:
        - ticker (str): Ticker symbol of the stock, retrieved from yfinance. 
        - start (str): Start date for stock data retrieval (format: 'YYYY-MM-DD').
        """
        stock_data = yf.Ticker(stock_ticker).history(start=stock_start_date)

        super().__init__(stock_data)

    # @property
    # def _constructor(self):
    #     return StockData

    def single_fred_data(self, fred_ticker: str):
        """
        Retrieve data for a single ticker from the St Louis Fed (FREDAPI).

        Parameters:
        - ticker: FRED data series ticker.
        """
        fred_ticker_df = fred.get_series_all_releases(fred_ticker, realtime_start=self.index.min().strftime('%Y-%m-%d'), realtime_end=datetime.datetime.now().strftime('%Y-%m-%d'))
        return fred_ticker_df
    
    def clean_fred_data(self, fred_ticker:str, fred_df: pd.DataFrame, tz : str = 'America/New_York'):
        """
        Cleans the data including converting the Datetime column into the local time of the area you want and only keeping the last value when duplicates arise (related to FREDAPI data). Also renames the columns to include ticker as a suffix to differentiate.

        Parameters:
        - tz: Any timezone you want, a list can be found online.
        """
        fred_df['realtime_start'] = fred_df['realtime_start'].dt.tz_localize(tz)
        fred_df.drop_duplicates(subset='realtime_start', keep='last', inplace=True)
        for col in renamed_cols:
            fred_df.rename(columns={col: col + '_' + fred_ticker}, inplace=True)
        return fred_df
    
    def merge_stock_to_fred(self, merged_df, fred_df: pd.DataFrame):
        merged_stock = merged_df.merge(right = fred_df, left_index = True, right_on = 'realtime_start', how = 'left')
        merged_stock.__class__ = StockData
        return merged_stock
    
    def copy_df(self, deep=True):
        copied_df = super().copy(deep = True)
        if deep:
            copied_df.__class__ = StockData
        return copied_df
    
    def set_merged_index_to_date(self, merged_df):

        new_data = merged_df.set_index(keys = 'realtime_start', drop = True)
        new_data.__class__ = StockData
        return new_data

    def attach_fred_data(self, fred_tickers: list, tz: str = 'America/New_York'):
        """
        Attach macroeconomic data to the StockData object, retrieved from St Louis Fed (FREDAPI).

        Parameters:
        - tickers (list): List of FRED data series tickers.
        - tz (str): Any timezone you want, a list can be found online.
        """
        merged_df = self.copy_df(deep = True)
        for fred_ticker in fred_tickers:
            fred_df = self.single_fred_data(fred_ticker)
            fred_df = self.clean_fred_data(fred_ticker, fred_df, tz = tz)
            merged_df = self.merge_stock_to_fred(merged_df = merged_df, fred_df = fred_df)
            merged_df = self.set_merged_index_to_date(merged_df = merged_df)
        return merged_df


# COMMAND ----------

fred_tickers = ['T10YIE', 'CPIAUCSL', 'UNRATE', 'GDP', 'GDPC1', 'FEDFUNDS', 'REAINTRATREARAT10Y', 'MORTGAGE30US', 'MORTGAGE15US', 'A792RC0Q052SBEA']
yahoo_data = ['GBPUSD=X', 'EURUSD=X', 'CNY=X']
renamed_cols = fred.get_series_all_releases('T10YIE', realtime_start='2017-01-01', realtime_end=datetime.datetime.now().strftime('%Y-%m-%d')).columns.drop('realtime_start')

# COMMAND ----------

ticker = 'AAPL'
start_date = '2017-01-01'

# Create an instance of StockData
AAPL = StockData(ticker, start_date)

# COMMAND ----------

type(AAPL)

# COMMAND ----------

display(AAPL)

# COMMAND ----------

hasattr(AAPL, 'single_fred_data') and callable(AAPL.single_fred_data)

# COMMAND ----------

fred_tickers

# COMMAND ----------

# Attach FRED data
stock_data_instance = AAPL.attach_fred_data(fred_tickers = fred_tickers)

# COMMAND ----------

type(stock_data_instance)

# COMMAND ----------

stock_data_instance

# COMMAND ----------

# Now, stock_data_instance behaves like a DataFrame with additional functionalities
print(stock_data_instance.head())

# COMMAND ----------

stock = 'AAPL'
aapl = yf.Ticker(stock)
AAPL = aapl.history(start = '2017-01-01')

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

for data in fred_data:
    df = fred.get_series_all_releases(data, realtime_start='2017-01-01', realtime_end=datetime.datetime.now().strftime('%Y-%m-%d'))
    df['realtime_start'] = df['realtime_start'].dt.tz_localize('America/New_York')
    df.drop_duplicates(subset = 'realtime_start', keep = 'last', inplace = True)
    for col in renamed_cols:
        df.rename(columns={col:col+'_'+data}, inplace = True)
    AAPL = AAPL.merge(right = df, left_index= True, right_on= 'realtime_start', how = 'left', suffixes = None)
    AAPL.set_index('realtime_start', inplace = True, drop = True)
    print(f'done {data}')

# COMMAND ----------

AAPL

# COMMAND ----------

df = fred.get_series_all_releases('T10YIE', realtime_start='2017-01-01', realtime_end=datetime.datetime.now().strftime('%Y-%m-%d'))
df['realtime_start'] = df['realtime_start'].dt.tz_localize('America/New_York')
# df.drop_duplicates(subset = 'realtime_start', keep = 'last', inplace = True)
df

# COMMAND ----------

AAPL.merge(df, left_index= True, right_on='realtime_start', how = 'left', suffixes = ('x','y'))

# COMMAND ----------

# MAGIC %md
# MAGIC get realtime_start to see when it was actually recorded so you know how long to lag it by, for example in the below it has a date of 10th October 2023 for the data (beginning of the quarter) but actually it wasn't available data until 28th February 2024

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


