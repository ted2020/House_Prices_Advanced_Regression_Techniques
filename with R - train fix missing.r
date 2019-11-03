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

df <- read.csv("train.csv")


sapply(df, function(x){sum(is.na(x))})

df_ex <- df[,which(names(df) %in% c("MasVnrType","GarageCond","GarageQual","GarageFinish",
                                    "GarageType","Electrical","BsmtFinType2",
                                    "BsmtFinType1","BsmtExposure","BsmtCond","BsmtQual"))]

sapply(df_ex, function(x){sum(is.na(x))})

n=round(sqrt(nrow(df))-1)
df_imputedkNN = kNN(df_ex, k = n)
# table(df$GarageCond, df_imputedkNN$GarageCond)
# table(df$GarageCond)

# table(df$GarageQual, df_imputedkNN$GarageQual)
# table(df$GarageQual)

# table(df$GarageFinish, df_imputedkNN$GarageFinish)
# table(df$GarageFinish)

# table(df$GarageType, df_imputedkNN$GarageType)
# table(df$GarageType)

# table(df$Electrical, df_imputedkNN$Electrical)
# table(df$Electrical)

# table(df$BsmtFinType2, df_imputedkNN$BsmtFinType2)
# table(df$BsmtFinType2)

# table(df$BsmtFinType1, df_imputedkNN$BsmtFinType1)
# table(df$BsmtFinType1)

# table(df$BsmtExposure, df_imputedkNN$BsmtExposure)
# table(df$BsmtExposure)

# table(df$BsmtCond, df_imputedkNN$BsmtCond)
# table(df$BsmtCond)

# table(df$BsmtQual, df_imputedkNN$BsmtQual)
# table(df$BsmtQual)

head(df_imputedkNN[,c(1:11)],1)


df[,"BsmtQual"]=df_imputedkNN[,"BsmtQual"]
df[,"BsmtCond"]=df_imputedkNN[,"BsmtCond"]
df[,"BsmtExposure"]=df_imputedkNN[,"BsmtExposure"]
df[,"BsmtFinType1"]=df_imputedkNN[,"BsmtFinType1"]
df[,"BsmtFinType2"]=df_imputedkNN[,"BsmtFinType2"]
df[,"Electrical"]=df_imputedkNN[,"Electrical"]
df[,"GarageType"]=df_imputedkNN[,"GarageType"]
df[,"GarageFinish"]=df_imputedkNN[,"GarageFinish"]
df[,"GarageQual"]=df_imputedkNN[,"GarageQual"]
df[,"GarageCond"]=df_imputedkNN[,"GarageCond"]
df[,"MasVnrType"]=df_imputedkNN[,"MasVnrType"]


# levels(df$GarageCond)
# levels(df$GarageQual)
# levels(df$GarageFinish)
# levels(df$GarageType)
# levels(df$Electrical)
# levels(df$BsmtFinType2)
# levels(df$BsmtFinType1)
# levels(df$BsmtExposure)
# levels(df$BsmtCond)
# levels(df$BsmtQual)
# levels(df$MasVnrType)

# n_distinct(df$GarageYrBlt)
# n_distinct(df$MasVnrArea)
# n_distinct(df$LotFrontage)

df <- df[,-which(names(df) %in% c("Id","Alley","FireplaceQu","PoolQC","Fence","MiscFeature"))]

df <- df %>%
     group_by(Neighborhood) %>%
     mutate_each(funs(replace(., which(is.na(.)),
                                mean(., na.rm=TRUE))))

anyNA(df)

write.csv(df,"train_no_missing.csv",row.names=FALSE)
