import pandas as pd
from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,train_test_split
from sklearn.metrics import roc_curve,plot_confusion_matrix,accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import hyper_parameters
import os
    
def loading_files(file_name):
    df_path = os.path.join("D:\ML", file_name)
    return df_path

def c_calculation():
    C = np.linspace(1.0,10000.0,num=1000,dtype=float)
    return C

def best_estim_report(model):
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",model.best_estimator_)
    print("\n The best score across ALL searched params:\n",model.best_score_)
    print("\n The best Hyperparameters across ALL searched params:\n",model.best_params_)
    
def save_to_csv(arr):
    df = pd.DataFrame(arr, columns=["Predicted_val"] )
    path = os.path.join("D:\ML","predicted_val.csv")
    df.to_csv(path)
    
def main():
    
    features_df = pd.read_pickle(loading_files("features.pkl"))
    test_df = pd.read_csv(loading_files("test_data.csv"),header=None)
    labels_df = pd.read_csv(loading_files("train_labels.csv"),header=None)

    features_train, features_test, target_train, target_test = train_test_split(features_df, labels_df, random_state=0, stratify=labels_df)

    pipe = Pipeline([("classifier",RandomForestClassifier())])
    print("initializing grid search")
    multi_grid_search_gs = GridSearchCV(pipe,hyper_parameters.search_space,verbose=1,n_jobs=-1,scoring='f1_micro')
    best_models_gs = multi_grid_search_gs.fit(features_train,target_train.values.flatten())


    best_estim_report(best_models_gs)

    y_predict = best_models_gs.predict(features_test)

    f1_score(target_test, y_predict)

    plot_confusion_matrix(best_models_gs, features_test, target_test)

    save_to_csv(y_predict)
    
if __name__ == "__main__":
    main()
