# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Price Prediction Project

# COMMAND ----------

# MAGIC %md
# MAGIC The aim of this notebook is to build a model that can predict the future price of an asset for the next day.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Starting Spark Session

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql
import pyspark.pandas

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## data collection

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

# COMMAND ----------

AAPL = aapl.history(start = '2015-01-01')

# COMMAND ----------

globals()[stock] = aapl.history(start = '2000-01-01')

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

display(pyspark.pandas.from_pandas(AAPL))

# COMMAND ----------

test = spark.createDataFrame(AAPL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning Data

# COMMAND ----------

AAPL.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split

# COMMAND ----------

AAPL_train = AAPL[:2200]
AAPL_test = AAPL[2200:]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving Model as Table

# COMMAND ----------

spark.sql('DROP TABLE aaronxwalker.aapl_returns_baseline_automl_forecasting')

# COMMAND ----------

spark_df = spark.createDataFrame(AAPL_train)

spark_df.write.option('overwriteSchema','true').mode("overwrite").saveAsTable("aaronxwalker.AAPL_returns_baseline_AutoML_forecasting_train")

# COMMAND ----------

#you can create a new pandas dataframe witht the following command:
AAPL = spark.sql('select * from aaronxwalker.AAPL_returns_baseline_AutoML_forecasting').toPandas()

# COMMAND ----------

AAPL

# COMMAND ----------

# MAGIC %md
# MAGIC ## baseline model (using AutoML)

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


