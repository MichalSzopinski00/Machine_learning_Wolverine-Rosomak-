# Databricks notebook source
# MAGIC %pip install git+https://github.com/hyperopt/hyperopt-sklearn
# MAGIC %pip install requests
# MAGIC %pip install tabulate
# MAGIC %pip install future
# MAGIC %pip uninstall h2o
# MAGIC %pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

# COMMAND ----------

from sklearn import svm, datasets
from sklearn.dummy import DummyClassifier
import pandas as pd
from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold,cross_val_score, GridSearchCV, RandomizedSearchCV,train_test_split
from sklearn.metrics import roc_curve,plot_confusion_matrix,accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from hpsklearn import HyperoptEstimator, svc,any_classifier
from hyperopt import tpe
import pickle
import os

# COMMAND ----------

# MAGIC %md
# MAGIC #Loading data

# COMMAND ----------

def loading_files(file_name):
    df_path = os.path.join("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/", file_name)
    return df_path

# COMMAND ----------

features_df = pd.read_pickle(loading_files("features.pkl"))
test_df = pd.read_csv(loading_files("test_data-1.csv"),header=None)
labels_df = pd.read_csv(loading_files("train_labels-3.csv"),header=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Spliting date to appropriate groups

# COMMAND ----------

features, target = features_df, labels_df

features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0, stratify=target)

# COMMAND ----------

#load data bez standaryzacji
#standaryzacja na trainie?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparing range of hyper parameters

# COMMAND ----------

C_value= np.linspace(1.0,10000.0,num=1000,dtype=float)

# COMMAND ----------

logistic_reg_2 = LogisticRegression(random_state=0,max_iter=5000,solver="liblinear")

pipe = Pipeline([("classifier",RandomForestClassifier())])

search_space = [
                {
                "classifier":[logistic_reg_2],
                "classifier__penalty":['l1','l2'],
                "classifier__C":C_value
                },
                {
                "classifier":[RandomForestClassifier()],
                "classifier__n_estimators":[10,100,1000],
                "classifier__max_features":[1,2,3,4,5,6,7,8,9]
                },
                {
                "classifier":[KNeighborsClassifier()],
                "classifier__n_neighbors":[3,5,7,9,11,13,15,17,19,21]
                }
                ]

# COMMAND ----------

# MAGIC %md
# MAGIC # Grid search initialization

# COMMAND ----------

multi_grid_search_gs = GridSearchCV(pipe, search_space,verbose=1,n_jobs=-1,scoring='f1_micro')
best_models_gs = multi_grid_search_gs.fit(features_train,target_train.values.flatten())

# COMMAND ----------

# MAGIC %md
# MAGIC ## The best results for Grid search

# COMMAND ----------

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",best_models_gs.best_estimator_)
print("\n The best score across ALL searched params:\n",best_models_gs.best_score_)
print("\n The best Hyperparameters across ALL searched params:\n",best_models_gs.best_params_)

# COMMAND ----------

y_predict = best_models_gs.predict(features_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## F1

# COMMAND ----------

f1_score(target_test, y_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix

# COMMAND ----------

plot_confusion_matrix(best_models_gs, features_test, target_test)  
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## We also tried to run Randomized search however the results were not that great. <br>
# MAGIC Randomized Grid Search initialization <br>
# MAGIC multi_grid_search_rs = RandomizedSearchCV(pipe, search_space,verbose=1,n_jobs=-1,scoring='f1_micro',cv=10)<br>
# MAGIC best_models_rs = multi_grid_search_rs.fit(features_train,target_train.values.flatten())<br>
# MAGIC Results from Randomized Grid Search <br>
# MAGIC 
# MAGIC The best estimator across ALL searched params:<br>
# MAGIC Pipeline(steps=[('classifier', RandomForestClassifier(max_features=5))])<br>
# MAGIC 
# MAGIC The best score across ALL searched params:<br>
# MAGIC 0.9000714272169301<br>
# MAGIC 
# MAGIC The best Hyperparameters across ALL searched params:<br>
# MAGIC {'classifier__n_estimators': 100, 'classifier__max_features': 5, 'classifier': RandomForestClassifier(max_features=5)}<br>
# MAGIC F1:0.9472502805836139 <br>
# MAGIC confusion matrix was <br>
# MAGIC 0     94<br>
# MAGIC 0     844<br>

# COMMAND ----------

# MAGIC %md
# MAGIC # H2O (autoML) initialization

# COMMAND ----------

import h2o
from h2o.automl import H2OAutoML
h2o.init()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preparation data for H2O

# COMMAND ----------

train_p = pickle.load(open("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/features.pkl", 'rb'))
features_h2o = h2o.H2OFrame(train_p)

labels = pd.read_csv("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_labels-3.csv", names=["Labels"],header=None)
target_h2o = h2o.H2OFrame(labels)

train_pd = pd.concat([train_p, labels],axis=1)
train_f = h2o.H2OFrame(train_pd)

# COMMAND ----------

train, valid = train_f.split_frame(ratios=[.75])

# COMMAND ----------

# MAGIC %md
# MAGIC ### First run with 20 models

# COMMAND ----------

y="Labels"
# For binary classification, response should be a factor
train[y] = train[y].asfactor()

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# COMMAND ----------

print(aml.leader)

# COMMAND ----------

perf = aml.leader.model_performance(valid)
perf

# COMMAND ----------

perf.F1()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Second run with 10 models

# COMMAND ----------


y="Labels"
# For binary classification, response should be a factor
train[y] = train[y].asfactor()

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=10, seed=1)
aml.train(y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# COMMAND ----------

print(aml.leader)

# COMMAND ----------

perf = aml.leader.model_performance(valid)
perf

# COMMAND ----------

perf.F1()

# COMMAND ----------


