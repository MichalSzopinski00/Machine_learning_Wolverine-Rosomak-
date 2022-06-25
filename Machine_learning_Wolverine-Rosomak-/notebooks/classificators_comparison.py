# Databricks notebook source
# MAGIC %pip install git+https://github.com/hyperopt/hyperopt-sklearn
# MAGIC %pip install requests
# MAGIC %pip install tabulate
# MAGIC %pip install future
# MAGIC %pip uninstall h2o
# MAGIC %pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

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
    
def loading_files(file_name):
    df_path = os.path.join('dbfs','FileStore','shared_uploads','michal.szopinski@interia.eu','ML_data', file_name)
    return df_path

def c_calculation():
    C = np.linspace(1.0,10000.0,num=1000,dtype=float)
    return C

def best_estim_report(model);
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",model.best_estimator_)
    print("\n The best score across ALL searched params:\n",model.best_score_)
    print("\n The best Hyperparameters across ALL searched params:\n",model.best_params_)
    
main():
    
    features_df = pd.read_pickle(loading_files("features.pkl"))
    test_df = pd.read_csv(loading_files("test_data-1.csv"),header=None)
    labels_df = pd.read_csv(loading_files("train_labels-3.csv"),header=None)

    features_train, features_test, target_train, target_test = train_test_split(features_df, labels_df, random_state=0, stratify=labels_df)

    logistic_reg_2 = LogisticRegression(random_state=0,max_iter=5000,solver="liblinear")

    pipe = Pipeline([("classifier",RandomForestClassifier())])

    search_space = [
                    {
                    "classifier":[logistic_reg_2],
                    "classifier__penalty":['l1','l2'],
                    "classifier__C":c_calculation()
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

    multi_grid_search_gs = GridSearchCV(pipe, search_space,verbose=1,n_jobs=-1,scoring='f1_micro')
    best_models_gs = multi_grid_search_gs.fit(features_train,target_train.values.flatten())

  
    best_estim_report(best_models_gs)

    y_predict = best_models_gs.predict(features_test)

    f1_score(target_test, y_predict)

    plot_confusion_matrix(best_models_gs, features_test, target_test)
    
if __name__ == __main__:
    main()
