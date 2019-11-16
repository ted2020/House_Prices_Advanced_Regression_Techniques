#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
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


# In[2]:


df=pd.read_csv("train_one_hot_encoded_with_python.csv")


# In[3]:


y=df.loc[:, df.columns == 'SalePrice_y']


# In[4]:


X=df.loc[:, df.columns != 'SalePrice_y']


# In[5]:


# lasso

reg_lasso = LassoCV()
reg_lasso.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg_lasso.alpha_)
print("Best score using built-in LassoCV: %f" %reg_lasso.score(X,y))
coef_lasso = pd.Series(reg_lasso.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef_lasso != 0)) + " variables and eliminated the other " +  str(sum(coef_lasso == 0)) + " variables")

imp_coef = coef_lasso.sort_values()

matplotlib.rcParams['figure.figsize'] = (15.0, 35.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[6]:


# ridge

reg_ridge = RidgeCV()
reg_ridge.fit(X, y)
print("Best alpha using built-in RidgeCV: %f" % reg_ridge.alpha_)
print("Best score using built-in RidgeCV: %f" %reg_ridge.score(X,y))

coef_ridge = pd.Series(reg_ridge.coef_.ravel(), index = X.columns)

imp_coef = coef_ridge.sort_values()
imp_coef.sort_values(ascending=False).head(20)
matplotlib.rcParams['figure.figsize'] = (15.0, 35.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Ridge Model")


# In[ ]:




