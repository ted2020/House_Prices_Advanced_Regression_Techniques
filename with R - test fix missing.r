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

test <- read.csv("test.csv")

test <- test[,-which(names(test) %in% c("Id","Alley","FireplaceQu","PoolQC","Fence","MiscFeature"))]

sapply(test, function(x){sum(is.na(x))})

test_ex <- test[,which(names(test) %in% c("BsmtQual","Utilities","MSZoning","MasVnrType","Exterior2nd","Exterior1st",
                                          "BsmtCond","BsmtExposure","GarageCond","GarageQual",
                                          "GarageFinish","GarageType","BsmtFinType1","BsmtFinType2",
                                          "Functional","KitchenQual","SaleType"))]

sapply(test_ex, function(x){sum(is.na(x))})

n=round(sqrt(nrow(test))-1)
df_imputedkNN = kNN(test_ex, k = n)


head(df_imputedkNN[,c(1:17)],1)


test[,"MSZoning"]=df_imputedkNN[,"MSZoning"]
test[,"Utilities"]=df_imputedkNN[,"Utilities"]
test[,"Exterior1st"]=df_imputedkNN[,"Exterior1st"]
test[,"Exterior2nd"]=df_imputedkNN[,"Exterior2nd"]
test[,"MasVnrType"]=df_imputedkNN[,"MasVnrType"]
test[,"BsmtQual"]=df_imputedkNN[,"BsmtQual"]
test[,"BsmtCond"]=df_imputedkNN[,"BsmtCond"]
test[,"BsmtExposure"]=df_imputedkNN[,"BsmtExposure"]
test[,"BsmtFinType1"]=df_imputedkNN[,"BsmtFinType1"]
test[,"BsmtFinType2"]=df_imputedkNN[,"BsmtFinType2"]
test[,"KitchenQual"]=df_imputedkNN[,"KitchenQual"]
test[,"Functional"]=df_imputedkNN[,"Functional"]
test[,"GarageType"]=df_imputedkNN[,"GarageType"]
test[,"GarageFinish"]=df_imputedkNN[,"GarageFinish"]
test[,"GarageQual"]=df_imputedkNN[,"GarageQual"]
test[,"GarageCond"]=df_imputedkNN[,"GarageCond"]
test[,"SaleType"]=df_imputedkNN[,"SaleType"]



test <- test %>% 
     group_by(Neighborhood) %>% 
     mutate_each(funs(replace(., which(is.na(.)),
                                mean(., na.rm=TRUE))))

anyNA(test)

write.csv(test,"test_no_missing.csv",row.names=FALSE)


