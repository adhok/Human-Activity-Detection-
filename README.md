# Human-Activity-Detection-
Script for Human Activity detection that is based on Kaggle's Human Activity Recognition Dataset
The link to the dataset can be found [here](https://www.kaggle.com/mboaglio/simplifiedhuarus/kernels)

## Method 1 : Multinomial Logistic Regression Model

* Choose variables that significantly affect the activity. These variables are chosen using the ANOVA testing method on each independent variable against the dependent variable(activity)
* After the variables are chosen , we further drop the variables that display high correlations with other variables. After that the multinom() function is used to build the model(Classification Accuracy:~68%).


## Method 1 : Multinomial Logistic Regression Model with Principal Component Analysis

* Instead of dropping variables that could be essential to the explanation of the data, project the data along directions of maximaum variance.

* The principal components are calculated by eigen decomposition of the covariance matrix that is achieved through the operation Z<sup>T</sup>Z.; here *Z* is the normalized data matrix.

* After this, the principal components are calculated by multiplying the data matrix(Z) by the resulting eigen vectors. Before that, the eigen vectors need to be arranged in the descending order of eigen values

* The test data is scaled and multiplied with the eigen vectors. The resulting matrix is then used as the test data.

* A good test accuracy is achieved (>90%)

