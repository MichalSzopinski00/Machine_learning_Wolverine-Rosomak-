# Databricks notebook source
from sklearn import svm, datasets
from sklearn.dummy import DummyClassifier
import pandas as pd
from numpy import genfromtxt

# COMMAND ----------

features = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_data-1.csv"
target = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_labels-2.csv"
test = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/test_data-1.csv"

# COMMAND ----------

features_train = genfromtxt(features, delimiter=',')
target_train = genfromtxt(target, delimiter=',')
features_test = genfromtxt(test, delimiter=',')

# COMMAND ----------

print(features_train)

# COMMAND ----------

dummy = DummyClassifier(strategy="stratified")

# COMMAND ----------

dummy.fit(features_train,target_train)

# COMMAND ----------

target_test = dummy.predict(features_test)

# COMMAND ----------

dummy.score(features_test,target_test)

# COMMAND ----------

print(target_test)

# COMMAND ----------


