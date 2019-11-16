
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
library(outreg)
library(pscl)
# library(boot)
library(psych)

train <- read.csv("train_no_missing3.csv")

test <- read.csv("test_no_missing3.csv")



dim(train)
dim(test)

anyNA(train)
anyNA(test)

range((train$WoodDeckSF))

fit_stepAIC <- lm(
    log(SalePrice) ~ 
                  Neighborhood + 
                #  HouseStyle +
                  RoofMatl +
     OverallQual + 
                  BsmtQual + 
                  KitchenQual  + log(GrLivArea) +
                 # log(LotFrontage) +
     log(LotArea) + Street +
                  (BsmtFinSF1) + 
             #     (BsmtUnfSF) + 
                 MSZoning + 
              #    log(MSSubClass) + 
    log(X1stFlrSF) + (X2ndFlrSF)  + 
    OverallCond + GarageCars + 
                   (GarageArea) + 
                  #log(TotalBsmtSF+1) + 
    CentralAir  + 
    BsmtFullBath +
    FullBath + 
    HalfBath + 
    KitchenAbvGr + 
    Fireplaces +  
    (WoodDeckSF) + (ScreenPorch) + 
                 # BsmtExposure + 
                  (EnclosedPorch) + 
     YearRemodAdd_age +YearBuilt_age + 
             #     YrSold_age + 
                  SaleCondition + Functional + Fireplaces + TotalBsmtSF
                
    ,train)
summary(fit_stepAIC)


ols_vif_tol(fit_stepAIC) %>% arrange(desc(VIF))

plot(fit_stepAIC)

p <- predict(fgls,test)

p <- (as.data.frame(exp(p)))
head(p)

test_original2 <- read.csv("test.csv")

out <- cbind(test_original2["Id"],p["exp(p)"])
names(out) <- c("Id","SalePrice")

head(out)

write.csv(out,"out_fgls_int.csv",row.names=FALSE)

#  0.12477
# 0.12474
# 0.12528
# 0.12383
# 0.12104




























class(train$LotFrontage)

head(train[,c(
"SalePrice",  "GrLivArea" ,"LotFrontage" , "LotArea"  ,"BsmtFinSF1" , "BsmtUnfSF"  , "MSSubClass" ,
    "X1stFlrSF" , "X2ndFlrSF" , "OverallCond" , "GarageCars" , "GarageArea",
    "BsmtFullBath" , "FullBath" , "HalfBath", "KitchenAbvGr" , "Fireplaces",
   
    "ScreenPorch", "OpenPorchSF" , "EnclosedPorch"  
#      "YearRemodAdd_age" , "YearBuilt_age" , "YrSold_age" 
                  )])

Data.num = train[,c(
"SalePrice",  "GrLivArea" ,
                  "LotFrontage" ,
     "LotArea"  , 
                   "BsmtFinSF1" , "BsmtUnfSF"  , "MSSubClass" , 
    "X1stFlrSF" , "X2ndFlrSF" , 
    "OverallCond" , "GarageCars" , "GarageArea" ,
    "BsmtFullBath" , "FullBath" , "HalfBath" , 
    "KitchenAbvGr" , 
    "Fireplaces" ,  
     "ScreenPorch", "OpenPorchSF" , "EnclosedPorch" , 
     "YearRemodAdd_age" , "YearBuilt_age" , "YrSold_age" 
                  )]



corr.test(Data.num,
          use    = "pairwise",
          method = "pearson",
          adjust = "none")





plot(fit_stepAIC$residuals^2, cex = 0.7)


vcovHC(fit_stepAIC, type = 'HC')

fit2_2 <- lm(
  log(SalePrice) ~ 
                  Neighborhood + 
#     HouseStyle + 
    RoofMatl +
    OverallQual + OverallCond + 
    BsmtFinSF1 + BsmtFinSF1 + TotalBsmtSF + BsmtQual +
#       + BsmtExposure + BsmtUnfSF + BsmtCond
    KitchenQual + KitchenAbvGr + 
    log(GrLivArea) +
    log(LotFrontage) + log(LotArea) + #LotShape + 
    Street +
#      LandContour + 
#     Utilities + 
    Condition1 + Condition2 + 
    SaleCondition + SaleType + 
    MSZoning + 
#     log(MSSubClass) + 
    log(X1stFlrSF) + X2ndFlrSF + 
#      Exterior1st + Exterior2nd + 
#     MasVnrType + 
    GarageCars + GarageArea + 
#      GarageCond + GarageQual + GarageType + 
    CentralAir  + 
    BsmtFullBath + FullBath + HalfBath +  
    Fireplaces + 
#     Heating + 
    HeatingQC +
    WoodDeckSF + ScreenPorch + EnclosedPorch +
    YearRemodAdd_age +YearBuilt_age + #YrSold_age + GarageYrBlt_age + 
    Functional + Foundation + 
    PoolArea
#     ExterQual 
#     X1stFlrSF + X2ndFlrSF
  
          
                
            
            ,train)

class(train$PoolArea)

fgls <- lm(
   log(SalePrice) ~ 
               Neighborhood + 
    RoofMatl +
     BsmtQual + KitchenQual +  Street +
     Condition1 + Condition2 + 
    SaleCondition + SaleType +  MSZoning +  CentralAir  +  HeatingQC + Functional + Foundation + 
    
    
    
   ( OverallQual + OverallCond +
#     BsmtFinSF1 + 
    TotalBsmtSF + 
    KitchenAbvGr + 
    #log(GrLivArea) +
#     log(LotFrontage) + 
    #log(LotArea) + 
    #log(X1stFlrSF) + 
    X2ndFlrSF + 
    GarageArea +    
    BsmtFullBath + FullBath + HalfBath +      
    Fireplaces +    
    WoodDeckSF + 
#     ScreenPorch + 
    EnclosedPorch +
    YearRemodAdd_age 
#     +YearBuilt_age  
    )^2

           
       
                   
    
    , data  = train, weights = 1/fit2_2$fitted.values^2)
summary(fgls)

plot(fgls)

ols_vif_tol(fgls) %>% arrange(desc(VIF))

# Get the coefficient matrix
coefs <- summary(fgls)$coefficients

# Identify the variables with "3 stars"
vars <- rownames(coefs)[which(coefs[, 4] < 0.4)]
(vars)

paste((vars),collapse = "+")


