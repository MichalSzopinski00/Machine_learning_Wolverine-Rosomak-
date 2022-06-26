import h2o
import os
import pandas as pd
from h2o.automl import H2OAutoML
h2o.init()


def loading_files(file_name):
    df_path = os.path.join('/dbfs/','FileStore','shared_uploads','michal.szopinski@interia.eu','ML_data', file_name)
    return df_path


def main():
    train_p = pd.read_pickle(loading_files("features.pkl"))
    features_h2o = h2o.H2OFrame(train_p)

    labels = pd.read_csv(loading_files("train_labels.csv"), names=["Labels"],header=None)
    target_h2o = h2o.H2OFrame(labels)

    train_pd = pd.concat([train_p, labels],axis=1)
    train_f = h2o.H2OFrame(train_pd)

    train, valid = train_f.split_frame(ratios=[.80])

    y="Labels"
    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()

    # Run AutoML for 20 base models
    aml = H2OAutoML(max_models=1, seed=1)
    aml.train(y=y, training_frame=train)

    # View the AutoML Leaderboard
    lb = aml.leaderboard
    lb.head(rows=lb.nrows)  

    print(aml.leader)

    perf = aml.leader.model_performance(valid)

    perf.F1()
    
if __name__ == "__main__":
    main()
