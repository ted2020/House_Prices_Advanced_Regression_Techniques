library(randomForest)
library(caret)
library(dplyr)
library(ggplot2)

train_fixed <- read.csv("train_no_missing.csv")
anyNA(train_fixed)

set.seed(100)
train <- sample(nrow(train_fixed), 0.7*nrow(train_fixed), replace = FALSE)
TrainSet <- train_fixed[train,]
ValidSet <- train_fixed[-train,]

fit <- randomForest(SalePrice ~., TrainSet)

fit

plot(fit)

fit2 <-randomForest(SalePrice ~., TrainSet,ntree = 300, mtry = 15)

fit2

y_pred = predict(fit,TrainSet$SalePrice)
# table(y_pred,TrainSet$SalePrice)

plot(y_pred)

fit3 <- predict(fit,ValidSet)
mean(fit3==ValidSet$SalePrice)


randomForest::varImpPlot(fit)




