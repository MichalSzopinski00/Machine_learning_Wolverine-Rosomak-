# Databricks notebook source
# MAGIC %pip install scikit-plot

# COMMAND ----------

from sklearn import svm, datasets
from sklearn.dummy import DummyClassifier
import pandas as pd
from numpy import genfromtxt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC #Loading data

# COMMAND ----------

features_df = pd.read_pickle("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/features.pkl")
test_df = pd.read_csv("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/test_data-1.csv",header=None)
labels_df = pd.read_csv("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_labels-3.csv",header=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test train split

# COMMAND ----------

features, target = features_df, labels_df

features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)

# COMMAND ----------

print(features_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # initializing Dummy Classifier

# COMMAND ----------

dummy = DummyClassifier(strategy="most_frequent")

# COMMAND ----------

dummy.fit(features_train,target_train)

# COMMAND ----------

y_predict = dummy.predict(features_test)

# COMMAND ----------

df = pd.DataFrame(y_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dummy score

# COMMAND ----------

dummy.score(features_test,target_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## F1

# COMMAND ----------

f1_score(target_test, y_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix

# COMMAND ----------

plot_confusion_matrix(dummy, features_test, target_test)  
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accuracy of the prediction

# COMMAND ----------

accuracy_score(target_test, y_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ROC of the prediction

# COMMAND ----------


fpr, tpr, thresholds = roc_curve(target_test, y_predict)

plt.figure(figsize=(15, 7))
plt.plot(fpr, tpr, alpha=0.5, color="blue", label="Elements")
plt.title("ROC Curve", fontsize=20)
plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.legend()

# COMMAND ----------


