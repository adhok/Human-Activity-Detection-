# Human-Activity-Detection-
Script for Human Activity detection that is based on Kaggle's Human Activity Recognition Dataset
The link to the dataset can be found [here](https://www.kaggle.com/mboaglio/simplifiedhuarus/kernels)

## Method 1 : Multinomial Logistic Regression Model

* Choose variables that significantly affect the activity. These variables are chosen using the ANOVA testing method on each independent variable against the dependent variable(activity)
* After the variables are chosen , we further drop the variables that display high correlations with other variables. After that the multinom() function is used to build the model(Classification Accuracy:~68%).

