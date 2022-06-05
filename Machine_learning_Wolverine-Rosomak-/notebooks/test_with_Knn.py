# Databricks notebook source
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

features = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_data-1.csv"
target = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_labels-2.csv"
test = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/test_data-1.csv"

# COMMAND ----------

features_train = genfromtxt(features, delimiter=',')
target_train = genfromtxt(target, delimiter=',')
features_test = genfromtxt(test, delimiter=',')

# COMMAND ----------

# MAGIC %md
# MAGIC How many neighbors is the best for this prediction?

# COMMAND ----------

train_list = []
test_list = []
number_of_neighbors = []
for i in range(1,16,1):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(features_train,target_train)
    
    target_test = knn.predict(features_test)
    
    number_of_neighbors.append(i)
    train_list.append(knn.score(features_train,target_train))
    test_list.append(knn.score(features_test,target_test))

# COMMAND ----------

test_list

# COMMAND ----------

df = pd.DataFrame(list(zip(number_of_neighbors, train_list,test_list)),
               columns =['number of neighbors', 'train value', 'test value'])
df.set_index('number of neighbors')

# COMMAND ----------

# MAGIC %md this is a bit worried let's try to mitigate this data issue.Let's try to improve it using Standard Scaller

# COMMAND ----------

std = StandardScaler()

features_train_std = std.fit_transform(features_train)
features_test_std = std.fit_transform(features_test)

# COMMAND ----------

train_list_std = []
test_list_std = []
number_of_neighbors_std = []

for i in range(1,16,1):
    
    knn_std = KNeighborsClassifier(n_neighbors=i)

    knn_std.fit(features_train_std,target_train)
    
    target_test = knn.predict(features_test)
    
    number_of_neighbors_std.append(i)
    train_list_std.append(knn_std.score(features_train_std,target_train))
    test_list_std.append(knn_std.score(features_test_std,target_test))

# COMMAND ----------

df2 = pd.DataFrame(list(zip(number_of_neighbors_std, train_list_std,test_list_std)),
               columns =['number of neighbors', 'train value', 'test value'])
df2.set_index('number of neighbors')

# COMMAND ----------


