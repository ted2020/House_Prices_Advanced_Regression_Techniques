
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

df <- read.csv("train_xgboost.csv")
anyNA(df)

df <- df[,-which(names(df) %in% c("Id"
                                 
                                 
                                 ))]

# head(log(df$SalePrice))

df$SalePrice <- log(df$SalePrice)

df$LotArea <- log(df$LotArea)

df$GrLivArea <- log(df$GrLivArea)

head(df[280])

set.seed(123)
split = sample.split(df$SalePrice, SplitRatio = 0.7)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

which(names(training_set)=="SalePrice")

xgb_grid_1 = expand.grid(
nrounds = 1000,
eta = c(0.1, 0.05, 0.01),
max_depth = c(2, 3, 4, 5, 6),
gamma = 1,
colsample_bytree=c(0.5, 0.6),
min_child_weight=seq(1),
subsample= c(0.5, 0.6)
)

xgb_trcontrol_1 <-trainControl(
method = "cv",
number = 5,
verboseIter = TRUE,
returnData = FALSE,
returnResamp = "all",                                                        # save losses across all models
classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
summaryFunction = twoClassSummary,
allowParallel = TRUE
)

xgb_train_1 = xgboost(data=as.matrix(training_set[-280]),label = training_set$SalePrice,
trControl = xgb_trcontrol_1,
tuneGrid = xgb_grid_1,
method = "xgbTree"
)

gc()

searchGridSubCol <- expand.grid(subsample = c(0.5, 0.6,0.7,0.8,0.9,1), 
                                colsample_bytree = c(0.5, 0.6,0.7,0.8,0.9,1),
                                max_depth = c(2,3,4,5,6),
                                min_child = seq(1), 
                                eta = c(0.1)
)

ntrees <- 500

system.time(
rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentDepth <- parameterList[["max_depth"]]
  currentEta <- parameterList[["eta"]]
  currentMinChild <- parameterList[["min_child"]]
  xgboostModelCV <- xgb.cv(data = as.matrix(training_set[-280]),label = training_set$SalePrice,
                           nrounds = ntrees, nfold = 5, showsd = TRUE, 
                       metrics = "rmse", verbose = TRUE, "eval_metric" = "rmse",
                     "objective" = "reg:linear", "max.depth" = currentDepth, "eta" = currentEta,                               
                     "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate
                      , print_every_n = 10, "min_child_weight" = currentMinChild, booster = "gbtree",
                     early_stopping_rounds = 10)
  
  xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
  rmse <- tail(xvalidationScores$test_rmse_mean, 1)
  trmse <- tail(xvalidationScores$train_rmse_mean,1)
  output <- return(c(rmse, trmse, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta, currentMinChild))}))

output <- as.data.frame(t(rmseErrorsHyperparameters))
varnames <- c("TestRMSE", "TrainRMSE", "SubSampRate", "ColSampRate", "Depth", "eta", "currentMinChild")
names(output) <- varnames
head(output)



# ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) +
# geom_point() +
# theme_bw() +
# scale_size_continuous(guide = "none")

classifier = xgboost(data = as.matrix(training_set[-280]), label = training_set$SalePrice,
                     nrounds = 1000,
                    method='xgbTree', trControl= my_control, tuneGrid=xgb_grid)

classifier



mat <- xgb.importance(colnames(training_set),classifier)
head(mat %>% arrange(desc(Gain)),10)

# xgb.plot.importance(importance_matrix = mat)

test_original <- read.csv("test_xgboost.csv")



test_original <- test_original[,-which(names(test_original) %in% c("Id"  , "SalePrice"  
))]

head(test_original)

dim(test_original)
dim(df)

pred <- predict(classifier,as.matrix(test_original))

xgb_pred <- as.data.frame(exp(pred))
head(xgb_pred)

test_original2 <- read.csv("test.csv")

out <- cbind(test_original2["Id"],xgb_pred["exp(pred)"])
names(out) <- c("Id","SalePrice")

dim(out)

write.csv(out,"out_xgb.csv",row.names=FALSE)

# score 0.14183







