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
