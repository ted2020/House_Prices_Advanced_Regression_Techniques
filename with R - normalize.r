
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(data.table)
library(mlr)
library(Hmisc)
library(caTools)
library(mltools)
library(ggplot2)
library(dplyr)

train <- read.csv("train_no_missing3.csv")

test <- read.csv("test_no_missing3.csv")

# normalize data
train_norm <- preProcess(train, method=c("center", "scale"))
print(train_norm)

trainnorm <- predict(train_norm, train)
dim(trainnorm)

head(trainnorm)

write.csv(trainnorm,"train_normalized_factor.csv")



# normalize data
test_norm <- preProcess(test, method=c("center", "scale"))
print(test_norm)

testnorm <- predict(test_norm, test)
dim(testnorm)

head(testnorm)

write.csv(testnorm,"test_normalized_factor.csv")


