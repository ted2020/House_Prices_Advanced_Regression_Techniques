#!/usr/bin/env python
# coding: utf-8

# In[346]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math
import datetime
from sklearn import preprocessing
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.api as sm
from scipy import stats
from statsmodels.compat import lzip
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics.regressionplots import *
from yellowbrick.regressor import CooksDistance
from yellowbrick.datasets import load_concrete
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from yellowbrick.model_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from yellowbrick.datasets import load_credit
from yellowbrick.target import FeatureCorrelation
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[347]:


train=pd.read_csv("train_no_missing_from_python.csv")


# In[348]:


train.columns[train.isnull().any()]


# In[349]:


# limit to categorical data using df.select_dtypes()
train_objects = train.select_dtypes(include=[object])
train_objects.head()


# In[350]:


# check original shape
train.shape


# In[351]:


train_objects.columns


# In[352]:


# TODO: create a LabelEncoder object and fit it to each feature in X


# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()


# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
X_2 = train_objects.apply(le.fit_transform)
X_2.head()


# In[353]:


# OneHotEncoder

# Encode categorical integer features using a one-hot aka one-of-K scheme.
# The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features.
# The output will be a sparse matrix where each column corresponds to one possible value of one feature.
# It is assumed that input features take on values in the range [0, n_values).
# This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.

# TODO: create a OneHotEncoder object, and fit it to all of X

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_2)

# 3. Transform
onehotlabels = enc.transform(X_2).toarray()
onehotlabels.shape

# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data


# In[354]:


train_objects_columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
       'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType',
       'SaleCondition', 'Neighborhood_y', 'BsmtQual_y', 'GarageType_y',
       'GarageFinish_y', 'GarageQual_y', 'GarageCond_y', 'Electrical_y',
       'BsmtFinType2_y', 'BsmtFinType1_y', 'BsmtExposure_y', 'BsmtCond_y']


# In[355]:


df = pd.concat([train,pd.get_dummies(train_objects,prefix=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',       'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',       'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',       'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType',       'SaleCondition', 'Neighborhood_y', 'BsmtQual_y', 'GarageType_y',       'GarageFinish_y', 'GarageQual_y', 'GarageCond_y', 'Electrical_y',       'BsmtFinType2_y', 'BsmtFinType1_y', 'BsmtExposure_y', 'BsmtCond_y'],drop_first=True)],axis=1)


# In[356]:


df.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',       'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType',       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',       'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',       'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType',       'SaleCondition', 'Neighborhood_y', 'BsmtQual_y', 'GarageType_y',       'GarageFinish_y', 'GarageQual_y', 'GarageCond_y', 'Electrical_y',       'BsmtFinType2_y', 'BsmtFinType1_y', 'BsmtExposure_y', 'BsmtCond_y'],axis=1, inplace=True)


# In[357]:


df.head()


# In[358]:


df=df.apply(pd.to_numeric,downcast='integer')


# In[359]:


# df.to_csv("train_one_hot_encoded_with_python.csv")


# In[59]:


# df["SalePrice_y"].head()


# In[157]:


y=df.loc[:, df.columns == 'SalePrice_y']


# In[333]:


X=df.loc[:, df.columns != 'SalePrice_y']
# XX=df.drop("SalePrice_y",axis=1)

# X = sm.add_constant(X
X.shape


# In[341]:


lm = smf.ols(formula='y~X',data=df).fit()
print(lm.summary(xname = ['bias'] + [l for l in X], yname='SalePrice'))


# In[343]:


lm2 = smf.ols(formula='y~X',data=df).fit(cov_type='HC0')
print(lm2.summary(xname = ['bias'] + [l for l in X], yname='SalePrice'))


# In[69]:


# residual plot
pred_val = lm.fittedvalues.copy()
true_val = (df['SalePrice_y']).values.copy()
residual = true_val - pred_val

fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(residual, pred_val)


# In[70]:


residual.sum()


# In[71]:


# RMSE
def rmse_accuracy_percentage(a,b): 
    print("RMSE is:",np.round(np.sqrt(sum(((np.array(a)-np.array(b))**2))/len(a)),2))
rmse_accuracy_percentage(true_val,pred_val)
print("stdev: ", str(np.std(df["SalePrice_y"])))


# In[90]:


variables = lm.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif


# In[73]:


np.array(vif).mean()


# In[74]:


# correlation 
al_cor=df.corr()
al_cor=al_cor.unstack()
al_cor["SalePrice_y"].sort_values(ascending=False)


# In[75]:


# Assumption of Independent Errors

statsmodels.stats.stattools.durbin_watson(lm.resid)


# In[76]:


# Assumption of Normality of the Residuals

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(lm.resid)
print(lzip(name, test))


# In[77]:


# Assumption of Normality of the Residuals

df['SalePrice_y'].plot(kind='hist', 
                       title= 'Log of Sale Price Distribution')


# In[78]:


# Assumption of Normality of the Residuals

df['SalePrice_y'] = np.log(df['SalePrice_y'])
df['SalePrice_y'].plot(kind='hist', 
                       title= 'Log of Sale Price Distribution')


# In[80]:


# Assumption of Normality of the Residuals

#Running plot & giving it a title
stats.probplot(lm.resid, dist="norm", plot= plt)
plt.title("Model1 Residuals Q-Q Plot")


# In[81]:


# Assumption of Homoscedasticity
name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sms.het_breuschpagan(lm.resid, lm.model.exog)
lzip(name, test)


# In[92]:


# cook's distance
influence = lm.get_influence()
#c is the distance and p is p-value
(c, p) = influence.cooks_distance
plt.stem(np.arange(len(c)), c, markerfmt=",")


# In[95]:


# plotting residuals against leverage
# plot_leverage_resid2(lm)
# influence_plot(lm)


# In[96]:


# sm_fr has the columns cooks_d and dffits
infl = lm.get_influence()
sm_fr = infl.summary_frame()


# In[100]:


# https://www.scikit-yb.org/en/latest/api/regressor/influence.html
# Instantiate and fit the visualizer
visualizer = CooksDistance()
visualizer.fit(X, y)
visualizer.show()


# In[109]:


# # Instantiate RFECV visualizer with a linear SVM classifier
# visualizer = RFECV(SVC(kernel='linear', C=1))

# visualizer.fit(X, y)        # Fit the data to the visualizer
# visualizer.show()           # Finalize and render the figure


# In[108]:


# # Load classification dataset
# X, y = load_credit()

# cv = StratifiedKFold(5)
# visualizer = RFECV(RandomForestClassifier(), cv=cv, scoring='f1_weighted')

# visualizer.fit(X, y)        # Fit the data to the visualizer
# visualizer.show()           # Finalize and render the figure


# In[125]:


#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
(model.pvalues).sort_values(ascending=False)


# In[154]:


# outreg2 output of Stata
dfoutput = summary_col([lm,lm2],stars=True)
print(dfoutput)


# In[ ]:





# In[ ]:




