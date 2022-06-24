Goal of the project was to find the best prediction model for given data.

We got the data in 3 different files. The first one contain the features, second the labels which contain 2 classes (-1,1) and the last one contains test data. The entire train dataset has 10k columns and 3750 rows. We classified that the prediction issue that we need to solve is classification problem. We checked our train data and examine if it has normal distribution. Only two columns were not normally distributed(it’s clarified within EDA file). Our next step was to identify the outlayers. We was able to catch 18928 outlayers. We turn them into the NaN values and then later on filled them using the median of current column. We also tried to visualize the data using KPCA, Tsne and PCA+ Tsne(with and without standardization). After the testing it turned out that the data with standardization was less clearer to read:

With standardization
 

Without standardization:
 

That’s why we decided to not standardize the data since it could make our labels harder to detect by model.

In the next step we defined our baseline(DummyClassifier) using “most_frequent” strategy. From our Dummy we was able to get this metrics:

F1 Score: 0.9413092550790069
Confusion matrix:
 

To pick the best model we used GridSearch/ Randomized GridSearch and H2O(AutoML) . The best model for Grid search were KNeighborsClassifier.
With this hyper parameters and metrics:
 
F1 Score: 0.9747504403992954
 Confusion matrix:
 
As we can see the KNN model was better than our baseline, which we can see in comparison between F1 and confusion metrics. In the same time also now most of the labels before classified as false positive were now correctly marked as positive negative. In fact the H2O model has a bit better accuracy in classifying the labels however the balance on H2O model is more on false negative rather than on false Positive side, that’s why we believe that KNN model is the best choice compare to H2O model.
H2O model metrics:L
F1 score dla h2o: 0.9786003470213996
confusion matrix dla h2o

 
To sum up we proofed that for our classification problem the best model to predict the labels would be the KNeighborsClassifier with following parameters:
 
Which is providing us relatively low level of prediction with keeping the smallest balance on False negative scores 

, co zapewni nam relatywnie wysoki poziom predykcji przy zachowaniu mniejszego wyniku fałszywie negatywnych odpowiedzi w 95%
