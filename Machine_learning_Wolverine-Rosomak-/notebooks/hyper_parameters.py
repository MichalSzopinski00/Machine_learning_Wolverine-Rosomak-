from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from classificators_comparison import c_calculation
logistic_reg_2 = LogisticRegression(random_state=0,max_iter=5000,solver="liblinear")

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
