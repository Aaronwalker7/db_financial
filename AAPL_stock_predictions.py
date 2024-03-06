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

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import numba
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark import pandas as ps
import datetime

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

stock = 'AAPL'
aapl = yf.Ticker(stock)
AAPL = aapl.history(start = '2015-01-01')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Curating Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### macroeconomic data

# COMMAND ----------

from fredapi import Fred
fred = Fred(api_key_file='fred_api_key.txt')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### the following are the tickers for the macro datasets and their name
# MAGIC
# MAGIC T10YIE - 10 year average expected inflation in US
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

# COMMAND ----------

macroeconomic = ['T10YIE', 'CPIAUCSL', 'UNRATE', 'GDP', 'GDPC1', 'FEDFUNDS', 'REAINTRATREARAT10Y']

# COMMAND ----------

data = fred.get_series('A939RX0Q048SBEA')
data

# COMMAND ----------

# MAGIC %md
# MAGIC get realtime_start to see when it was actually recorded so you know how long to lag it by, for example in the below it has a date of 10th October 2023 for the data (beginning of the quarter) but actually it wasn't available data until 28th February 2024

# COMMAND ----------

df = fred.get_series_all_releases('A939RX0Q048SBEA')
df.tail()

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


