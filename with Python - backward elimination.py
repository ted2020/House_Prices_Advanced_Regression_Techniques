#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


df=pd.read_csv("train_one_hot_encoded_with_python.csv")


# In[8]:


y=df.loc[:, df.columns == 'SalePrice_y']


# In[9]:


X=df.loc[:, df.columns != 'SalePrice_y']


# In[11]:


#Adding constant column of ones, mandatory for sm.OLS modelX_1 = sm.add_constant(X)

#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues.head()


# In[12]:


#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:





# In[ ]:





# In[ ]:




