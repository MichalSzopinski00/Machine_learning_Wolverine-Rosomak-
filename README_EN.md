<h1>ML Project</h1><br>
  <i>Michał Szopiński</i> <BR>
  <i>Jan Michalski</i> <BR><BR><BR>
Goal of the project was to find the best prediction model for given data.<br><br>

 feel free to download datasets use below links:<br>
 train(before preparation):<br>
  https://drive.google.com/file/d/1mHJtAClxaURbVJMrckoC8jBLDwA1LqEi/view?usp=sharing<br>
  
 train(after preparation):<br>
 https://drive.google.com/file/d/1tcqziY3sWYN4DBWpvT1Q0l07vqoyeyGJ/view?usp=sharing<br>
 
 labels for train:<br>
 https://drive.google.com/file/d/1IjpiP7pdXgyst9Uz0yL-QjaHaa-3t3r4/view?usp=sharing<br>
 test dataset:<br>
 https://drive.google.com/file/d/1H8IMq3B-9c9slkpwu-s4b3xId9fnTrSl/view?usp=sharing<br>
  
  <b>Note:</b> Before re-running the code please make sure that all above files are included within your local directory + change the path from dbfs to your local path within load_data function.<br>
 
We got the data in 3 different files. The first one contain the features, second the labels which contain 2 classes (-1,1) and the last one contains test data. The entire train dataset has 10k columns and 3750 rows. We classified that this is a classification problem. We checked our train data and examine if it has normal distribution. Only two columns were not normally distributed(it’s clarified within EDA file). Our next step was to identify the outlayers. We were able to catch 17928 outlayers. We turn them into the NaN values and then later on filled them using the median of current column. We also tried to visualize training data using KPCA, Tsne and PCA+ Tsne(with and without standardization). During replications it turned out that the data with standardization was less clearer to read:

With standardization:<br><br>
 ![image](https://user-images.githubusercontent.com/49531926/175608515-a7c340a2-7af4-481e-8695-bb89e15e87e2.png)


Without standardization:<br><br>
 ![image](https://user-images.githubusercontent.com/49531926/175608530-0f0b0773-729c-4b60-8892-32e418ef8537.png)


That’s why we decided to not standardize the data since it could make our labels harder to detect by model.

In the next step we defined our baseline(DummyClassifier) using “most_frequent” strategy. From our Dummy we was able to get this metrics:

F1 Score: 0.9413092550790069<br>
Confusion matrix:<br>
 ![image](https://user-images.githubusercontent.com/49531926/175608651-a1213761-6cb8-4452-b2ca-b5341ccf87e5.png)


To pick the best model we used GridSearch/ Randomized GridSearch and H2O(AutoML) . The best model for Grid search were KNeighborsClassifier.
With this hyper parameters and metrics:<br>
 ![image](https://user-images.githubusercontent.com/49531926/175608636-ba7d24ee-5bef-4c34-85a4-fc1fc1f3ff49.png)

F1 Score: 0.9747504403992954<br>
 Confusion matrix:<br>
![image](https://user-images.githubusercontent.com/49531926/175608662-a379d7f7-0ffd-496f-87e1-ba7899a9e6df.png)

As we can see the KNN model was better than our baseline, which we can see in comparison between F1 and confusion metrics. In the same time also now most of the labels before classified as false positive were now correctly marked as positive negative. In fact the H2O model has a bit better accuracy in classifying the labels however the balance on H2O model is more on false negative rather than on false Positive side, that’s why we believe that KNN model is the best choice compare to H2O model.
H2O model metrics:
F1 score dla h2o: 0.9786003470213996<br>
confusion matrix dla h2o<br>
![image](https://user-images.githubusercontent.com/49531926/175608679-15786d2f-64fa-4a25-93a0-5da7cbe871fe.png)

 
To sum up we proofed that for our classification problem the best model to predict the labels would be the KNeighborsClassifier with following parameters:
 ![image](https://user-images.githubusercontent.com/49531926/175608688-9a4e577f-07cb-4175-b496-6f6bbd8dd769.png)

Which is providing us relatively low level of prediction with keeping the smallest balance on False negative scores.
 
