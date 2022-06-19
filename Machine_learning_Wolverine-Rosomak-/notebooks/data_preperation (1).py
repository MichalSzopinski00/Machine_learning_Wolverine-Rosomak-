# Databricks notebook source
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import matplotlib as plt
import numpy as np
from scipy.stats import zscore as z_score_calculating, median_abs_deviation
from sklearn.decomposition import PCA, KernelPCA

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading data

# COMMAND ----------

features_path = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_data-1.csv"
#test = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/test_data-1.csv"
target_labels_path = "/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_labels-2.csv"

# COMMAND ----------


rows_num = pd.read_csv('/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_data-1.csv', nrows = 1)
observations = []
col_num = len(rows_num.columns)
observations = ['observations_' + str(n) for n in range (0, col_num)]


features_df = pd.read_csv(features_path, names = observations)
#test_1 = pd.read_csv(test,header=None,names="observations_")
target_labels_df = pd.read_csv(target_labels_path, names=['labels'])

features_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now let's check what is the information about the dataset

# COMMAND ----------

features_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculating mean, var, std, median

# COMMAND ----------

description_df = features_df.agg(['mean', 'std','var','median'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Checking normal Distribution

# COMMAND ----------

skew_df = features_df.agg(['skew','kurtosis']).T

# COMMAND ----------

skew_df.min()

# COMMAND ----------

skew_df[skew_df["kurtosis"]<-1]

# COMMAND ----------

# MAGIC %md
# MAGIC as we can see observarions 1017 and 5432 were skew, now we will drop the observations which are skew.

# COMMAND ----------

# MAGIC %md
# MAGIC We also tried:<br>
# MAGIC skew_df[skew_df["kurtosis"]>1]<br>
# MAGIC skew_df[skew_df["skew"]>1]<br>
# MAGIC skew_df[skew_df["skew"]<-1]<br>
# MAGIC however there were no records found
# MAGIC in addition to that we tried to drop these 2 columns however the KPCA/TSNE graph wasn't looking good while we were performing this operation.<br>
# MAGIC features_df = features_df.drop(['observations_1017', 'observations_5432'], axis = 1)

# COMMAND ----------

features_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # Outliers

# COMMAND ----------

def modified_z_score_outlier(dataset: pd.DataFrame) -> pd.DataFrame:   
    mad_column = median_abs_deviation(dataset)
    median = np.median(dataset, axis = 0)
    mad_score = np.abs(0.6745 * (dataset - median) / mad_column)
    return mad_score > 3.5
test = modified_z_score_outlier(features_df)

# COMMAND ----------

test.sum().sum()

# COMMAND ----------

lista = []
for i in test:
    for x in test[i]:
        if x == True:
            lista.append(x)

# COMMAND ----------

lista.count(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count of catched outliners are similar which means that our mechanism is catching outliners properly

# COMMAND ----------

column = []
row = []
for i in test:
    j=0
    for x in test[i]:
        if x == True:
            column.append(i)
            row.append(j)
        j+=1

# COMMAND ----------

location_of_outliners = pd.DataFrame()
location_of_outliners['column_loc'] = column
location_of_outliners['row_loc'] = row
location_of_outliners

# COMMAND ----------

for index, row in location_of_outliners.iterrows():
    features_df.loc[row['row_loc'],row['column_loc']] = np.nan

# COMMAND ----------

features_df.isna().sum().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### this is expected scenario now we need to fill this NaN values using median

# COMMAND ----------

# MAGIC %md 
# MAGIC # NaN values

# COMMAND ----------

features_df_final = features_df.fillna(features_df.median())

# COMMAND ----------

features_df_final.isna().sum().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### As we can see now we filled all NaN values using median.

# COMMAND ----------

# MAGIC %md
# MAGIC # Standarization

# COMMAND ----------

# MAGIC %md
# MAGIC ## We tried to use standarization however this gave us worser results

# COMMAND ----------

# MAGIC %md
# MAGIC scaler = StandardScaler()<br>
# MAGIC X = scaler.fit_transform(features_df_final)<br>
# MAGIC ---------------------------------------------
# MAGIC x_std = scaler.fit(x_train) <br>
# MAGIC x_train_std = x_std.transofrm(x_train) <br>
# MAGIC x_test_std = x_std.transofrm(x_test)<br>

# COMMAND ----------

# MAGIC %md
# MAGIC # Reducing the dimensionality of data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using only TSNE

# COMMAND ----------

X_deep = TSNE().fit_transform(features_df_final)

# COMMAND ----------

sns.set(rc={'figure.figsize':(9.7,7.27)})

sns.scatterplot(X_deep[:,0]
               , X_deep[:,1]
               , hue=target_labels_df["labels"]
               , palette = ['r', 'g'])

# COMMAND ----------

pca_tsne = Pipeline(steps = [
    ("pca", PCA(n_components=50)),
     ("tsne", TSNE(n_components=2))])

# COMMAND ----------

pca_tsne_deep = pca_tsne.fit_transform(features_df_final)

# COMMAND ----------

pca_tsne_deep.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using TSNE + PCA

# COMMAND ----------

sns.set(rc={'figure.figsize':(9.7,7.27)})

sns.scatterplot(pca_tsne_deep[:,0]
               , pca_tsne_deep[:,1]
               , hue=target_labels_df["labels"]
               , palette = ['r', 'g'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using KPCA

# COMMAND ----------

kpca = KernelPCA()
kpca_embeded = kpca.fit_transform(features_df_final)

# COMMAND ----------

sns.set(rc={'figure.figsize':(9.7,7.27)})

sns.scatterplot(kpca_embeded[:,0]
               , kpca_embeded[:,1]
               , hue=target_labels_df["labels"]
               , palette = ['r', 'g'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Saving changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## saving df to pickle

# COMMAND ----------

features_df_final.to_pickle("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/features.pkl")

# COMMAND ----------


