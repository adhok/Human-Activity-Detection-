---
title: "Exploratory Data Analysis of Human Activity Recognition Data Set"
output:
  html_document:
    toc: true
    toc_float:
      collapsed: false
      smooth_scroll: false
---



## Motivation

The human activity data consists of 561 features. Using all of these features in a predictive modeling procedure can be computationally tedious. On the other hand, removing features can result in losing vital information that can describe the process that generated the dependent variables.

Instead of feature removal, trying to derive a new and reduced set of features from the existing set, can infact help us reduce computational efforts while maintaining informational integrity. This process is called feature extraction, with PCA being one of them.



## PCA Algorithm function

PCA is short for Principal Component Analysis. The original set of features are combined and linearly transformed into principal components. 
The set of principal components that explain the data effectively ,is chosen as the new feature set. The PCA process can be explained briefly in the following steps.

Let *X* be a matrix that represents the set of features that we are running PCA on. Each column in this matrix represents a feature.

*  __Step 1__ : Subtract each column by its mean and divide it by the standard deviation.
*  __Step 2__ : Let the new normalized matrix be *Z* . Using this normalized matrix , calculate the covariance matrix ie Z<sup>T</sup>Z . Let this covariance matrix be *C*
*  __Step 3__ : Decompose the covariance matrix into eigen values and eigen vectors. Here the `eigen()` function is used. The eigen vector corresponding to the highest eigen value is the direction of maximum variance.

*  __Step 4__ : To acquire the principal components, arrange the eigen values in descending order and reorder the eigen vectors accordingly. This reordering is taken care of by the `eigen()` function.

*  __Step 5__ : Now that we have the directions along which the variance is maximum, we need to project them on the data that we have ,to create the features. This is achieved by multiplying the scaled feature matrix *Z* with the reordered eigen matrix. The explained variance of each component is the eigen value corresponding to that component divided by the sum of eigen values.


These steps are encapsulated via the `pca_function()` that is defined below. This function returns the following

1. The projected vectors

2. The eigen values

3. The directions along which the data is projected.




```r
pca_function = function(x,k){
  
  # x is the input data frame
  ## Step 1 Normalize each column
  
  x_scaled = scale(x)
  ## Convert the data frame to a matrix
  
  x_mat = as.matrix(x_scaled)
  
  ## Calculate the covariance matrix
  
  cov_x = t(x_mat) %*% x_mat
  
  
  ## Eigen Value decomposition
  
  decomp_eigen = eigen(cov_x)
  
  eigen_values = decomp_eigen$values
  
  eigen_vectors = decomp_eigen$vectors

  
  # choose top k eigen values and explain the variance
  
  eigen_values = eigen_values[1:k]
  
  ## calculate the projected vectors
  
  
  projected_vectors = x_mat %*% eigen_vectors
  projected_vectors = projected_vectors[1:nrow(projected_vectors),1:k]
  return(list(projected_vectors=projected_vectors,eigen_values=eigen_values,basis_vectors=eigen_vectors))
}
```



## Choosing the Best K

Its good to visualize the optimal number of principal components by looking how well they explain the data. Consecutive principal components explain lesser and lesser of the variance. 



```r
## removing the redundant columns




set.seed(42)

libraries_needed = c('tidyr','dplyr','ggplot2','caret','purrr','rlang','caret')
lapply(libraries_needed ,require,character.only=TRUE)
```

```
## [[1]]
## [1] TRUE
## 
## [[2]]
## [1] TRUE
## 
## [[3]]
## [1] TRUE
## 
## [[4]]
## [1] TRUE
## 
## [[5]]
## [1] TRUE
## 
## [[6]]
## [1] TRUE
## 
## [[7]]
## [1] TRUE
```

```r
data_raw= read.csv('train.csv',stringsAsFactors = FALSE)


data_raw_pca = data_raw %>% select(-rn,-activity)




train_index = sample(1:nrow(data_raw_pca),0.75*nrow(data_raw_pca))

data_raw_pca_train = data_raw_pca[train_index,]

data_raw_pca_test = data_raw_pca[-train_index,]
  
pca_decomposed_data = pca_function(data_raw_pca_train,561)
```




```r
## Convert to a data frame
options(scipen = 999)

eigen_values_and_vectors = data.frame(principal_component = 1:ncol(data_raw_pca_train),eigen_values = pca_decomposed_data$eigen_values)


eigen_values_and_vectors %>%
  mutate(var = eigen_values/sum(eigen_values)) %>%
  mutate(cumulative_var = cumsum(var)) %>%
  ggplot(aes(x=principal_component,y=cumulative_var))+
  geom_line()+geom_point()
```

![](https://raw.githubusercontent.com/adhok/Human-Activity-Detection-/master/figure/unnamed-chunk-3-1.png)


The first 100 principal components explain about 95% of the variance in the data. Let's apply this transformation and choose the first 100 principal components. We will then use these principal components in our multinomial logistic regression model. 




```r
pca_final_train = pca_function(data_raw_pca_train,100)


data_pca_train = as.data.frame(pca_final_train$projected_vectors)

data_pca_train$activity = as.character(data_raw[train_index,]$activity)
```



Multinomial logistic regression models help in deriving the log odds of an event happening against a base event. This is example, the base event is chosen as `SITTING`.



```r
library(nnet)

data_pca_train$activity = as.factor(data_pca_train$activity)
data_pca_train$activity = relevel(data_pca_train$activity,ref='SITTING')

model_logistic_pca = multinom(activity~.,data=data_pca_train)
```

```
## # weights:  612 (505 variable)
## initial  value 4848.501124 
## iter  10 value 702.867290
## iter  20 value 576.468943
## iter  30 value 509.505561
## iter  40 value 427.630665
## iter  50 value 343.541258
## iter  60 value 282.536232
## iter  70 value 227.906896
## iter  80 value 199.037550
## iter  90 value 175.641086
## iter 100 value 150.658264
## final  value 150.658264 
## stopped after 100 iterations
```

```r
#broom::tidy(model_logistic_pca)

test = predict(model_logistic_pca,newdata=data_pca_train %>% select(-activity))


real_pred = data.frame(real=data_pca_train$activity,test=as.character(test))
caret::confusionMatrix(real_pred$real,real_pred$test)
```

```
## Warning in confusionMatrix.default(real_pred$real, real_pred$test): Levels
## are not in the same order for reference and data. Refactoring data to
## match.
```

```
## Confusion Matrix and Statistics
## 
##                     Reference
## Prediction           LAYING SITTING STANDING WALKING WALKING_DOWNSTAIRS
##   LAYING                530       0        0       0                  0
##   SITTING                 0     429       22       0                  0
##   STANDING                0      13      483       0                  0
##   WALKING                 0       0        0     457                  0
##   WALKING_DOWNSTAIRS      0       0        0       0                361
##   WALKING_UPSTAIRS        0       0        0       0                  1
##                     Reference
## Prediction           WALKING_UPSTAIRS
##   LAYING                            0
##   SITTING                           0
##   STANDING                          0
##   WALKING                           0
##   WALKING_DOWNSTAIRS                0
##   WALKING_UPSTAIRS                410
## 
## Overall Statistics
##                                                
##                Accuracy : 0.9867               
##                  95% CI : (0.9816, 0.9907)     
##     No Information Rate : 0.1959               
##     P-Value [Acc > NIR] : < 0.00000000000000022
##                                                
##                   Kappa : 0.984                
##                                                
##  Mcnemar's Test P-Value : NA                   
## 
## Statistics by Class:
## 
##                      Class: LAYING Class: SITTING Class: STANDING
## Sensitivity                 1.0000         0.9706          0.9564
## Specificity                 1.0000         0.9903          0.9941
## Pos Pred Value              1.0000         0.9512          0.9738
## Neg Pred Value              1.0000         0.9942          0.9900
## Prevalence                  0.1959         0.1633          0.1866
## Detection Rate              0.1959         0.1585          0.1785
## Detection Prevalence        0.1959         0.1667          0.1833
## Balanced Accuracy           1.0000         0.9804          0.9753
##                      Class: WALKING Class: WALKING_DOWNSTAIRS
## Sensitivity                  1.0000                    0.9972
## Specificity                  1.0000                    1.0000
## Pos Pred Value               1.0000                    1.0000
## Neg Pred Value               1.0000                    0.9996
## Prevalence                   0.1689                    0.1338
## Detection Rate               0.1689                    0.1334
## Detection Prevalence         0.1689                    0.1334
## Balanced Accuracy            1.0000                    0.9986
##                      Class: WALKING_UPSTAIRS
## Sensitivity                           1.0000
## Specificity                           0.9996
## Pos Pred Value                        0.9976
## Neg Pred Value                        1.0000
## Prevalence                            0.1515
## Detection Rate                        0.1515
## Detection Prevalence                  0.1519
## Balanced Accuracy                     0.9998
```

* As per the output, the 98% of the data points were classified properly.
* For the purpose of prediction on the test data, we project the scaled test data onto the principal directions.
* The resulting vectors are then used in the model to generate the predicted values.
* These predicted values are then compared against the test values to evaluate the test accuracy,


## Testing 


```r
## testing

## Project scaled test data onto the directons of maximum variance (basis vectors)
data_pca_test = as.matrix(scale(data_raw_pca_test)) %*% pca_final_train$basis_vectors

data_pca_test = as.data.frame(data_pca_test)

data_pca_test = data_pca_test[,1:100]

data_pca_test$activity = data_raw[-train_index,]$activity
 
test_prediction = predict(model_logistic_pca,newdata = data_pca_test %>% select(-activity))
real_pred_test = data.frame(real=data_pca_test$activity,test=as.character(test_prediction))


caret::confusionMatrix(real_pred_test$real,real_pred_test$test)
```

```
## Confusion Matrix and Statistics
## 
##                     Reference
## Prediction           LAYING SITTING STANDING WALKING WALKING_DOWNSTAIRS
##   LAYING                150       1        0       0                  0
##   SITTING                 1     159       11       0                  0
##   STANDING                0      19      153       0                  0
##   WALKING                 0       0        1     143                  0
##   WALKING_DOWNSTAIRS      0       1        2       1                125
##   WALKING_UPSTAIRS        1       2        0       1                  3
##                     Reference
## Prediction           WALKING_UPSTAIRS
##   LAYING                            0
##   SITTING                           1
##   STANDING                          0
##   WALKING                           2
##   WALKING_DOWNSTAIRS                3
##   WALKING_UPSTAIRS                123
## 
## Overall Statistics
##                                                
##                Accuracy : 0.9446               
##                  95% CI : (0.9276, 0.9586)     
##     No Information Rate : 0.2016               
##     P-Value [Acc > NIR] : < 0.00000000000000022
##                                                
##                   Kappa : 0.9334               
##                                                
##  Mcnemar's Test P-Value : NA                   
## 
## Statistics by Class:
## 
##                      Class: LAYING Class: SITTING Class: STANDING
## Sensitivity                 0.9868         0.8736          0.9162
## Specificity                 0.9987         0.9820          0.9742
## Pos Pred Value              0.9934         0.9244          0.8895
## Neg Pred Value              0.9973         0.9685          0.9808
## Prevalence                  0.1683         0.2016          0.1849
## Detection Rate              0.1661         0.1761          0.1694
## Detection Prevalence        0.1672         0.1905          0.1905
## Balanced Accuracy           0.9928         0.9278          0.9452
##                      Class: WALKING Class: WALKING_DOWNSTAIRS
## Sensitivity                  0.9862                    0.9766
## Specificity                  0.9960                    0.9910
## Pos Pred Value               0.9795                    0.9470
## Neg Pred Value               0.9974                    0.9961
## Prevalence                   0.1606                    0.1417
## Detection Rate               0.1584                    0.1384
## Detection Prevalence         0.1617                    0.1462
## Balanced Accuracy            0.9911                    0.9838
##                      Class: WALKING_UPSTAIRS
## Sensitivity                           0.9535
## Specificity                           0.9910
## Pos Pred Value                        0.9462
## Neg Pred Value                        0.9922
## Prevalence                            0.1429
## Detection Rate                        0.1362
## Detection Prevalence                  0.1440
## Balanced Accuracy                     0.9722
```

* 94% of test data is correctly predicted.


## 10 folds Cross validation of the approach


Good training and test accuracies are achieved in the previous sections with one set of test and train data. A good way to make sure that the predictive modeling approach is robust is to run it with different combinations of test and train data.


```r
random_data = dplyr::slice(data_raw,sample(1:n()))
test_idx = round(seq(1,nrow(data_raw),by=nrow(data_raw)/11))
accuracy_df = data.frame(iteration= as.numeric(),training_score= as.numeric(),testing_score= as.numeric())
for(i in 1:10){
  
  
  ## Training
  
  test_data = slice(random_data,test_idx[i]:test_idx[i+1])
  train_data = slice(random_data,-test_idx[i]:-test_idx[i+1])
  pca_final_train = pca_function(train_data %>% select(-rn,-activity),100)
  
  data_pca_train = as.data.frame(pca_final_train$projected_vectors)
  data_pca_train$activity = as.character(train_data$activity)
  #data_pca_train$activity = relevel(data_pca_train$activity,ref='SITTING')

  model_cv = multinom(activity~.,data=data_pca_train)
  
  train_predict = as.character(predict(model_cv,newdata=data_pca_train %>% select(-activity)))
  
  training_score = sum(as.character(train_predict)==as.character(train_data$activity))/nrow(train_data)
  
  
  
  
  
  ## Testing 
  data_pca_test = as.matrix(scale(test_data %>% select(-rn,-activity))) %*% pca_final_train$basis_vectors

  data_pca_test = as.data.frame(data_pca_test)

  data_pca_test = data_pca_test[,1:100]

  data_pca_test$activity = test_data$activity
  
  
  
  predict_cv = as.character(predict(model_cv,newdata=data_pca_test %>% select(-activity)))
  testing_score = sum(as.character(predict_cv)==as.character(test_data$activity))/nrow(test_data)
  accuracy_df = rbind(accuracy_df,data.frame(iteration=i,testing_score=testing_score,training_score =training_score ))
  
  

  
  
  
  
  
}
```

```
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 1050.001866
## iter  20 value 813.032889
## iter  30 value 734.207693
## iter  40 value 681.914029
## iter  50 value 585.109672
## iter  60 value 522.264751
## iter  70 value 461.713076
## iter  80 value 326.419585
## iter  90 value 284.633728
## iter 100 value 262.248291
## final  value 262.248291 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 984.777490
## iter  20 value 755.697797
## iter  30 value 709.440073
## iter  40 value 664.231434
## iter  50 value 568.319443
## iter  60 value 504.365404
## iter  70 value 444.803457
## iter  80 value 306.282217
## iter  90 value 265.246137
## iter 100 value 243.659261
## final  value 243.659261 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 1015.516374
## iter  20 value 779.146394
## iter  30 value 720.814104
## iter  40 value 671.778591
## iter  50 value 574.020011
## iter  60 value 523.444769
## iter  70 value 470.882583
## iter  80 value 332.261768
## iter  90 value 271.110588
## iter 100 value 247.833111
## final  value 247.833111 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 962.980901
## iter  20 value 764.793931
## iter  30 value 710.758908
## iter  40 value 663.497487
## iter  50 value 556.037458
## iter  60 value 494.486654
## iter  70 value 437.619508
## iter  80 value 296.778773
## iter  90 value 258.850692
## iter 100 value 233.159554
## final  value 233.159554 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 977.271922
## iter  20 value 770.788527
## iter  30 value 713.968195
## iter  40 value 669.732514
## iter  50 value 570.095391
## iter  60 value 501.455867
## iter  70 value 437.379239
## iter  80 value 314.445908
## iter  90 value 274.827634
## iter 100 value 249.823170
## final  value 249.823170 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5875.179300 
## iter  10 value 1022.003607
## iter  20 value 780.069176
## iter  30 value 713.048752
## iter  40 value 665.900903
## iter  50 value 590.169319
## iter  60 value 510.002601
## iter  70 value 426.708675
## iter  80 value 307.457189
## iter  90 value 268.393700
## iter 100 value 243.319061
## final  value 243.319061 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 1115.583024
## iter  20 value 853.653604
## iter  30 value 759.083997
## iter  40 value 702.167937
## iter  50 value 619.787102
## iter  60 value 538.689701
## iter  70 value 500.708958
## iter  80 value 347.697787
## iter  90 value 294.991414
## iter 100 value 267.773347
## final  value 267.773347 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 1010.651841
## iter  20 value 776.024297
## iter  30 value 718.425077
## iter  40 value 671.321199
## iter  50 value 576.193096
## iter  60 value 506.319719
## iter  70 value 454.701566
## iter  80 value 326.339174
## iter  90 value 273.432564
## iter 100 value 248.129307
## final  value 248.129307 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 1028.010462
## iter  20 value 809.071083
## iter  30 value 756.336823
## iter  40 value 701.222864
## iter  50 value 601.862669
## iter  60 value 535.073398
## iter  70 value 498.022598
## iter  80 value 390.210103
## iter  90 value 290.164599
## iter 100 value 258.730903
## final  value 258.730903 
## stopped after 100 iterations
## # weights:  612 (505 variable)
## initial  value 5876.971059 
## iter  10 value 1022.039165
## iter  20 value 816.169150
## iter  30 value 741.826763
## iter  40 value 690.795501
## iter  50 value 618.941938
## iter  60 value 535.685635
## iter  70 value 474.775392
## iter  80 value 347.390115
## iter  90 value 284.448361
## iter 100 value 259.196300
## final  value 259.196300 
## stopped after 100 iterations
```



```r
scaleFUN = function(x) sprintf("%.f", x)

accuracy_df %>%
  tidyr::gather(type_of_score,value,2:3) %>%
  mutate(type_of_score = ifelse(grepl('training',type_of_score),'Training Score','Testing Score')) %>%
  ggplot(aes(x=iteration,y=value,color=type_of_score))+geom_line()+
  scale_x_continuous(labels = scaleFUN)+
  labs(x='Iteration Number',y='Score')
```
![](https://raw.githubusercontent.com/adhok/Human-Activity-Detection-/master/figure/unnamed-chunk-8-1.png)

* Training scores are stable and testing scores range from 0.94 to 0.97 which is acceptable for a good multi class classifier.



## Closing remarks

* PCA allows for modeling in a smaller dimension without the loss of vital information. 
* Running a multi class logistic model with PCA yields better testing and training results.








