#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import missingno as msno
import math
from fancyimpute import KNN


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


# In[64]:


train=pd.read_csv("train.csv")


# In[67]:


train = train.replace('None',np.nan)


# In[68]:


print(train.isnull().sum())


# In[69]:


train=train.drop(["MasVnrType","MiscFeature","Fence","PoolQC","FireplaceQu","Alley", "Id"],axis=1)


# In[70]:


# print(train.dtypes)

# numeric missing = lotFrontage, GarageYrBlt, 
# categorical missing = BsmtQual, GarageType, GarageFinish, GarageQual, GarageCond, Electrical, BsmtFinType2,
                        #            BsmtFinType1, BsmtExposure, BsmtCond , MasVnrType


# In[92]:


train2=train[["SalePrice","Neighborhood","OverallQual","BsmtQual", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "Electrical", "BsmtFinType2","BsmtFinType1", "BsmtExposure", "BsmtCond"]]


# In[115]:


train2.head()


# In[94]:


print(train2.isnull().sum()), print(train2.dtypes)


# In[85]:


# train2 = train2.replace('None',np.nan)


# In[102]:


# for each column, get value counts in decreasing order and take the index (value) of most common class
train2_imputed = train2.apply(lambda x: x.fillna(x.value_counts().index[0]))


# In[ ]:





# In[159]:


train4.columns[train4.isnull().any()]


# In[158]:


print(train4.isnull().sum())


# In[121]:


train2_imputed.head()


# In[123]:


# train3=pd.concat([train,train2_imputed],axis=1)


# In[129]:


# train3.head()


# In[154]:


train4 = pd.merge(train, train2_imputed, left_index=True, right_index=True, how='outer')


# In[155]:


# train2_imputed.columns


# In[156]:


train4=train4.drop(['SalePrice_x', 'Neighborhood_x', 'OverallQual_x', 'BsmtQual_x',                    'GarageType_x','GarageFinish_x', 'GarageQual_x', 'GarageCond_x',                    'Electrical_x','BsmtFinType2_x', 'BsmtFinType1_x', 'BsmtExposure_x', 'BsmtCond_x'],axis=1)


# In[157]:


train4 = train4.groupby(['Neighborhood_y']).apply(lambda x: x.fillna(x.mean()))


# In[161]:


train4.to_csv("train_no_missing_from_python.csv",index=False)


# In[ ]:





# In[18]:


# train["BsmtQual"].unique()


# In[19]:


# train2 = train2.replace(np.nan,'None')


# In[20]:


# train2=pd.get_dummies(train2,dummy_na=True,drop_first=True)
# train2=pd.get_dummies(train2,dummy_na=True)


# In[22]:


# train_cols = list(train2)


# In[25]:


# n=round(math.sqrt(len(train2)))
# train2 = pd.DataFrame(KNN(k=n).fit_transform(train2))
# train.columns = train_cols


# In[ ]:





# In[153]:


# train2=train2.astype(str)
# train2["SalePrice"]=train2["SalePrice"].astype(int)


# In[154]:


# label_encoder = preprocessing.LabelEncoder() 


# In[155]:


# train2["nBsmtQual"]= label_encoder.fit_transform(train2["BsmtQual"])
# train2["nGarageType"]= label_encoder.fit_transform(train2["GarageType"])
# train2["nGarageFinish"]= label_encoder.fit_transform(train2["GarageFinish"])
# train2["nGarageQual"]= label_encoder.fit_transform(train2["GarageQual"])
# train2["nGarageCond"]= label_encoder.fit_transform(train2["GarageCond"])
# train2["nElectrical"]= label_encoder.fit_transform(train2["Electrical"])
# train2["nBsmtFinType2"]= label_encoder.fit_transform(train2["BsmtFinType2"])
# train2["nBsmtFinType1"]= label_encoder.fit_transform(train2["BsmtFinType1"])
# train2["nBsmtExposure"]= label_encoder.fit_transform(train2["BsmtExposure"])
# train2["nBsmtCond"]= label_encoder.fit_transform(train2["BsmtCond"])
# train2["nMasVnrType"]= label_encoder.fit_transform(train2["MasVnrType"])


# In[139]:


# train3=train2.loc[:, train2.dtypes == np.int32]


# In[127]:


# y = train3.iloc[:, :1].values
# X = train3.iloc[:,1:].values


# In[128]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[129]:


# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# In[108]:


# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




