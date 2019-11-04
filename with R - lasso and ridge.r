library(dplyr)
library(olsrr)
library(fastDummies)
library(knitr)
library(ggplot2)
library(MASS)
library(corrplot)
library(reshape2)
# library(HotDeckImputation)
# library(bnstruct)
library(VIM)
library(mice)
library(caret)
library(RANN)
library(Hmisc)
library(deldir)
library(kknn)
library(RcmdrMisc)
library(glmnet)
library(ridge)
library(car)

train <- read.csv("train_no_missing.csv")
anyNA(train)

# ridge regression

train_x <- model.matrix(SalePrice ~ ., train)[, -1]
train_y <- (train$SalePrice)

# test_x <- model.matrix(SalePrice ~ ., test)[, -1]
# test_y <- (test$SalePrice)

ridge <- glmnet(
  x = train_x,
  y = train_y,
  alpha = 0
)

plot(ridge,xvar = "lambda")

# Apply CV Ridge regression
ridge <- cv.glmnet(
  x = train_x,
  y = train_y,
  alpha = 0
)

# plot results
plot(ridge)

rmse_ridge <- sqrt(min(ridge$cvm))
rmse_ridge

bestlam = ridge$lambda.min # Select lamda that minimizes training MSE
bestlam = ridge$lambda.min
ridge_coef = predict(ridge, type = "coefficients", s = bestlam)[1:100,]
ridge_coef

## Apply lasso regression
lasso <- glmnet(
  x = train_x,
  y = train_y,
  alpha = 1
)

plot(lasso, xvar = "lambda")

# Apply CV Ridge regression
lasso <- cv.glmnet(
  x = train_x,
  y = train_y,
  alpha = 1
)
# plot results
plot(lasso)

rmse_lasso <- sqrt(min(lasso$cvm))
rmse_lasso

bestlam = lasso$lambda.min # Select lamda that minimizes training MSE
bestlam = lasso$lambda.min
lasso_coef = predict(lasso, type = "coefficients", s = bestlam)[1:100,]
lasso_coef


