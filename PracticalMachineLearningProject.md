---
title: "Practical Machine Learning Project"
author: "Paresh Kumar Mishra"
date: "21 Sep, 2014"
output: 
  html_document:
    toc: true
    theme: spacelab
---
## Introduction
This project is the final project of the practical machine learning course n the Data Science specialization of Coursera.
In the below document I have explained my analysis.
This document describes the analysis I conducted for my final project for the Johns Hopkins' Coursera course "Practical Machine Learning" in the Data Science specialization.  You can learn more about the course [here](https://www.coursera.org/course/predmachlearn).
[RStudio](http://www.rstudio.com) was used for conducting all of the project work.  

##Data Collection
I have downloaded the data for the  assignment from [here](http://groupware.les.inf.puc-rio.br/har). I have split the data into a training group (19,622) observations and testing group (20 observations).    

## Methods Used
First, training set is split into 90/10 smaller samples.  

```r
set.seed(614)
library(lattice); library(ggplot2); library(caret)
```

```
## Warning: package 'lattice' was built under R version 3.1.1
## Warning: package 'ggplot2' was built under R version 3.1.1
## Warning: package 'caret' was built under R version 3.1.1
```

```r
pml.training <- read.csv("C:/Users/pkmiteng/PracticalMachineLearning/pml-training.csv")
inTrain <- createDataPartition(y=pml.training$classe, p=0.9, list=FALSE)
training <- pml.training[inTrain,]
testing <- pml.training[-inTrain,]
```
Note: To run this code, the data inside `read.csv("")` is switched to the location of the data.   The  10 percent sample is used for cross-validation.  I did this simple cross-validation  to cut down on execution time.
Next, I implement a Stochastic Gradient Boosting algorithm via the `gbm` package.

```r
ptm <- proc.time()
modFit <- train(classe ~ user_name + pitch_arm + yaw_arm + roll_arm + roll_belt + pitch_belt + yaw_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + gyros_arm_x + gyros_arm_y + gyros_arm_z + accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + magnet_arm_z + roll_dumbbell + pitch_dumbbell + yaw_dumbbell, method="gbm", data=training, verbose=FALSE)
```

```
## Loading required package: gbm
```

```
## Warning: package 'gbm' was built under R version 3.1.1
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```
## Warning: package 'plyr' was built under R version 3.1.1
```

```r
proc.time() - ptm
```

```
##    user  system elapsed 
## 2034.47    6.81 2076.66
```
 I've used `ptm` and `proc.time()` to record the time taken for execution. It was found to be approximately 24 minutes.

```r
print(modFit)
```

```
## Stochastic Gradient Boosting 
## 
## 17662 samples
##   159 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 17662, 17662, 17662, 17662, 17662, 17662, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
##   1                   50      0.7       0.6    0.005        0.007   
##   1                  100      0.7       0.6    0.007        0.009   
##   1                  150      0.7       0.7    0.008        0.010   
##   2                   50      0.8       0.7    0.008        0.010   
##   2                  100      0.8       0.8    0.005        0.006   
##   2                  150      0.9       0.8    0.005        0.006   
##   3                   50      0.8       0.8    0.006        0.007   
##   3                  100      0.9       0.9    0.004        0.006   
##   3                  150      0.9       0.9    0.004        0.005   
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```

```r
predictTr <- predict(modFit,training)
table(predictTr, training$classe)
```

```
##          
## predictTr    A    B    C    D    E
##         A 4749  145   51   39   37
##         B   75 3131  119   17   34
##         C   62  113 2839  138   18
##         D   98   14   63 2678   15
##         E   38   15    8   23 3143
```
The model correctly classifies 93.6 percent of the observations in the training sample.  The "roll_belt"" and "yaw_belt"" features were by far the most important in terms of variable influence.  

```r
summary(modFit,n.trees=150)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4.png) 

```
##                                 var  rel.inf
## roll_belt                 roll_belt 24.15343
## yaw_belt                   yaw_belt 13.33063
## magnet_belt_z         magnet_belt_z  7.01097
## roll_dumbbell         roll_dumbbell  6.55280
## pitch_belt               pitch_belt  6.23819
## magnet_arm_x           magnet_arm_x  4.59912
## accel_arm_x             accel_arm_x  3.74037
## roll_arm                   roll_arm  3.64898
## magnet_arm_z           magnet_arm_z  3.25981
## gyros_belt_z           gyros_belt_z  3.12695
## user_nameeurico     user_nameeurico  3.00163
## yaw_dumbbell           yaw_dumbbell  2.63164
## accel_arm_z             accel_arm_z  2.63096
## pitch_dumbbell       pitch_dumbbell  2.59930
## yaw_arm                     yaw_arm  2.33455
## magnet_belt_x         magnet_belt_x  1.82863
## pitch_arm                 pitch_arm  1.67995
## accel_belt_z           accel_belt_z  1.42771
## magnet_arm_y           magnet_arm_y  1.38889
## gyros_arm_y             gyros_arm_y  1.29886
## magnet_belt_y         magnet_belt_y  1.26592
## user_namecharles   user_namecharles  0.74598
## gyros_belt_y           gyros_belt_y  0.63706
## gyros_arm_x             gyros_arm_x  0.36557
## gyros_belt_x           gyros_belt_x  0.25119
## accel_arm_y             accel_arm_y  0.17786
## accel_belt_x           accel_belt_x  0.07305
## user_namecarlitos user_namecarlitos  0.00000
## user_namejeremy     user_namejeremy  0.00000
## user_namepedro       user_namepedro  0.00000
## accel_belt_y           accel_belt_y  0.00000
## gyros_arm_z             gyros_arm_z  0.00000
```

A plot of these top two features colored by outcome demonstrates their relative importance.  

```r
qplot(roll_belt, yaw_belt,colour=classe,data=training)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 
Even though these are the top features, they're still not great predictors in their own right.  Nonetheless, you can see some bunching in this simple plot.  This confirms the choice of a boosting algorithm as a good choice given the large set of relatively weak predictors.  This next plot further demonstrates the improved performance gained by using boosting iterations.


```r
ggplot(modFit)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6.png) 

Next, I check the performance on the 10 percent subsample to get an estimate of the algorithm's out-of-sample performance.

```r
predictTe <- predict(modFit,testing)
table(predictTe, testing$classe)
```

```
##          
## predictTe   A   B   C   D   E
##         A 530  19   4   2   6
##         B   3 346  16   0   5
##         C   4  13 312  19   3
##         D  15   1  10 297   0
##         E   6   0   0   3 346
```
The accuracy of the result of the algorithm is slightly below in the testing subset than it did on the full training set, correctly classifying 93.4 percent of the observations.

## Predicting on the Test Set
Finally, I use the algorithm to predict using the testing set.  The results are run through the `pml_write_files()` function from the course Coursera site, and stored for submission.  

```r
pml.testing <- read.csv("C:/Users/pkmiteng/PracticalMachineLearning/pml-testing.csv")
answers <- as.character(predict(modFit, pml.testing))
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```
After submitting these answers, it turns out that the algorithm correctly predicted the outcome for 20/20 observations further confirming its strong out-of-sample classification accuracy.  
