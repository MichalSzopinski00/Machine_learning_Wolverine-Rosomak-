import h2o
from h2o.automl import H2OAutoML
h2o.init()

main():

    train_p = pickle.load(open("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/features.pkl", 'rb'))
    features_h2o = h2o.H2OFrame(train_p)

    labels = pd.read_csv("/dbfs/FileStore/shared_uploads/michal.szopinski@interia.eu/ML_data/train_labels-3.csv", names=["Labels"],header=None)
    target_h2o = h2o.H2OFrame(labels)

    train_pd = pd.concat([train_p, labels],axis=1)
    train_f = h2o.H2OFrame(train_pd)


    train, valid = train_f.split_frame(ratios=[.75])

    y="Labels"
    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()

    # Run AutoML for 20 base models
    aml = H2OAutoML(max_models=20, seed=1)
    aml.train(y=y, training_frame=train)

    # View the AutoML Leaderboard
    lb = aml.leaderboard
    lb.head(rows=lb.nrows)  

    print(aml.leader)

    perf = aml.leader.model_performance(valid)

    perf.F1()

if __name__ == __main__: 
    main()
        
