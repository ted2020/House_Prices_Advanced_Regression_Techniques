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

df <- read.csv("train_no_missing.csv")

anyNA(df)

df4 <- fastDummies::dummy_cols(df, remove_first_dummy = TRUE)
df4 <- df4[, !sapply(df4, is.factor)]

c <- cor(df4)
## c is the correlations matrix

## keep only the lower triangle by
## filling upper with NA
c[upper.tri(c, diag=TRUE)] <- NA
m <- melt(c)
## sort by descending absolute correlation
m <- m[order(- abs(m$value)), ]
## omit the NA values
dfOut <- na.omit(m)
head(dfOut,100)

df3 <- df[,-which(names(df) %in% c(
    "Exterior1st","SaleType","GarageCars",
    "GrLivArea","GarageYrBlt","X2ndFlrSF","X1stFlrSF",
    "BsmtFinSF2","BsmtFinSF1",
    "OverallQual","YearBuilt","MasVnrArea",
    "KitchenAbvGr","BedroomAbvGr",
    "RoofStyle","GarageCond","GarageQual",
    "ExterQual",
    "ExterCond","GarageType","MSZoning",
    "BsmtCond",
    "RoofMatl",
    "Heating","MSSubClass",
    "HalfBath","FullBath","YearRemodAdd","BsmtUnfSF",
    "LowQualFinSF","Foundation",
    "LotArea","TotalBsmtSF",
    "BsmtFullBath","BsmtQual",
    "Exterior2nd","MiscVal",
    "TotRmsAbvGrd",
    "MasVnrType",
    "BldgType"

                                   ))]

# CORR ELIMINATION
fit3 <- lm(SalePrice~.,df3)
summary(fit3)
tail(outreg::outreg(fit3,robust = TRUE),4)
ncol(df3)
# ols_vif_tol(fit) %>% arrange(desc(VIF))
# plot(fit)

RSS <- c(crossprod(fit3$residuals))
# paste("RSS is: ",RSS)
MSE <- RSS / length(fit3$residuals)
# paste("MSE is: ",MSE)
RMSE <- sqrt(MSE)
paste("RMSE is: ", RMSE)

# with factors FULL DATASET
fit <- lm(SalePrice~.,df)
summary(fit)
tail(outreg::outreg(fit,robust = TRUE),4)
ncol(df)
# ols_vif_tol(fit) %>% arrange(desc(VIF))
# plot(fit)

RSS <- c(crossprod(fit$residuals))
# paste("RSS is: ",RSS)
MSE <- RSS / length(fit$residuals)
# paste("MSE is: ",MSE)
RMSE <- sqrt(MSE)
paste("RMSE is: ", RMSE)

stepwise(fit,
    direction = c("backward/forward", "forward/backward", "backward", "forward"),
    criterion = c("BIC", "AIC"))

step <- stepAIC(fit, direction="both")
step$anova

# stepAIC SELECTION
fit13 <- lm(SalePrice ~ MSZoning + LotArea + Street + LandContour + LotConfig +
    LandSlope + Neighborhood + Condition1 + Condition2 + BldgType +
    HouseStyle + OverallQual + OverallCond + YearBuilt + YearRemodAdd +
    RoofStyle + RoofMatl + Exterior1st + MasVnrType + MasVnrArea +
    ExterQual + BsmtQual + BsmtCond + BsmtExposure + BsmtFinType1 +
    BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + X1stFlrSF + X2ndFlrSF +
    FullBath + BedroomAbvGr + KitchenAbvGr + KitchenQual + TotRmsAbvGrd +
    Functional + Fireplaces + GarageFinish + GarageCars + GarageArea +
    GarageQual + GarageCond + WoodDeckSF + ScreenPorch + PoolArea +
    MoSold + SaleCondition,df)
summary(fit13)
tail(outreg::outreg(fit13,robust = TRUE),4)
# ols_vif_tol(fit) %>% arrange(desc(VIF))
# plot(fit)

RSS <- c(crossprod(fit13$residuals))
# paste("RSS is: ",RSS)
MSE <- RSS / length(fit13$residuals)
# paste("MSE is: ",MSE)
RMSE <- sqrt(MSE)
paste("RMSE is: ", RMSE)

# stepwise SELECTION
fit9 <- lm(formula = SalePrice ~ MSSubClass + LotArea + Street + Neighborhood +
    Condition2 + OverallQual + OverallCond + YearBuilt + RoofMatl +
    MasVnrArea + ExterQual + BsmtQual + BsmtExposure + BsmtFinSF1 +
    BsmtFinSF2 + BsmtUnfSF + X1stFlrSF + X2ndFlrSF + BedroomAbvGr +
    KitchenQual + GarageArea + ScreenPorch + PoolArea + SaleCondition,
    data = df)
summary(fit9)
tail(outreg::outreg(fit9,robust = TRUE),4)
# ols_vif_tol(fit) %>% arrange(desc(VIF))
# plot(fit)

RSS <- c(crossprod(fit9$residuals))
# paste("RSS is: ",RSS)
MSE <- RSS / length(fit9$residuals)
# paste("MSE is: ",MSE)
RMSE <- sqrt(MSE)
paste("RMSE is: ", RMSE)

# from RANDOMFOREST
fitRF <- lm(SalePrice ~
            OverallQual+Neighborhood+GrLivArea+GarageCars+ExterQual+TotalBsmtSF+X1stFlrSF+
            KitchenQual+GarageArea+X2ndFlrSF+BsmtQual+BsmtFinSF1+LotArea+FullBath+YearBuilt+
            TotRmsAbvGrd+MasVnrArea+YearRemodAdd+Exterior1st+LotFrontage+Exterior2nd+
            GarageYrBlt+Fireplaces+GarageType+BsmtUnfSF+OpenPorchSF
            ,df)
summary(fitRF)
tail(outreg::outreg(fitRF,robust = TRUE),4)
# ncol(df3)
# ols_vif_tol(fit) %>% arrange(desc(VIF))
# plot(fit)

RSS <- c(crossprod(fitRF$residuals))
# paste("RSS is: ",RSS)
MSE <- RSS / length(fitRF$residuals)
# paste("MSE is: ",MSE)
RMSE <- sqrt(MSE)
paste("RMSE is: ", RMSE)

plot(fit13)

# toselect.x <- summary(fit9)$coeff[-1,4]<1
# # select sig. variables
# relevant.x <- names(toselect.x)[toselect.x == TRUE]
# paste(shQuote(relevant.x, type="cmd"), collapse=", ")

# anova(fit9)

# influence(fit9)

# fit_ols <- ols_step_all_possible(fit9)
# fit_ols

# best_aic <- arrange(fit_ols,aic)
# best_aic[1:5,]
# best_sbic <- arrange(fit_ols,sbic)
# best_sbic[1:5,]
