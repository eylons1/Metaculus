#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
from pathlib import Path
import subprocess
import os
import gc
from glob import glob

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer



# In[64]:


df = pd.read_csv(r"C:\Users\maorb\CSVs\MetaculusNewest18_6.csv")
df_new = pd.read_csv(r"C:\Users\maorb\CSVs\MetaculusNewest17_6.csv")


# In[3]:


df['Result'].value_counts()


# In[12]:


df.info()


# In[65]:


df1 = df.copy() #df without title
df1.drop(columns = 'title',inplace = True)


# In[4]:


df_new1 = df_new.copy()
df_new1.drop(columns = 'title',inplace = True)


# In[74]:


# DAYOFWEEK, HOUR, WEEKLY Cyclical FEATURES

# PUBLISH TIME
df1['publish_time'] = pd.to_datetime(df1['publish_time'])

# RESPNSE DATE
df1['Response_date'] = df1['Response_date']
df1['Response_date'] = pd.to_datetime(df1['Response_date'], errors='coerce')
df1['Response_date_dayofweek'] = df1['Response_date'].dt.dayofweek
df1['Response_date_hour'] = df1['Response_date'].dt.hour


# In[75]:


df1


# In[5]:


df_new1['publish_time'] = pd.to_datetime(df_new1['publish_time'])


# RESPNSE DATE
df_new1['Response_date'] = pd.to_datetime(df_new1['Response_date'], errors='coerce')
df_new1['Response_date_dayofweek'] = df_new1['Response_date'].dt.dayofweek
df_new1['Response_date_hour'] = df_new1['Response_date'].dt.hour


# In[76]:


import numpy as np
# For now, We'll create features for hour and dayofweek, more to come!
# For further reading: https://medium.com/@axelazara6/why-we-need-encoding-cyclical-features-79ecc3531232

df1['Response_date_Hour_sin'] = np.sin(df1['Response_date_hour'] * np.pi/12)
df1['Response_date_Hour_cos'] = np.cos(df1['Response_date_hour'] * np.pi/12)
df1['Response_date_dayofweek_sin'] = np.sin(df1['Response_date_dayofweek'] * np.pi/3.5)
df1['Response_date_dayofweek_cos'] = np.cos(df1['Response_date_dayofweek'] * np.pi/3.5)


# In[87]:


mean_hour_sin = df1.groupby('id')['Response_date_Hour_sin'].mean()
mean_hour_cos = df1.groupby('id')['Response_date_Hour_cos'].mean()
mean_dayofweek_sin = df1.groupby('id')['Response_date_dayofweek_sin'].mean()
mean_dayofweek_cos = df1.groupby('id')['Response_date_dayofweek_cos'].mean()


# In[78]:


df_new1['Response_date_Hour_sin'] = np.sin(df_new1['Response_date_hour'] * np.pi/12)
df_new1['Response_date_Hour_cos'] = np.cos(df_new1['Response_date_hour'] * np.pi/12)
df_new1['Response_date_dayofweek_sin'] = np.sin(df_new1['Response_date_dayofweek'] * np.pi/3.5)
df_new1['Response_date_dayofweek_cos'] = np.cos(df_new1['Response_date_dayofweek'] * np.pi/3.5)


# In[79]:


mean_hour_sin_new = df_new1.groupby('id')['Response_date_Hour_sin'].mean()
mean_hour_cos_new = df_new1.groupby('id')['Response_date_Hour_cos'].mean()
mean_dayofweek_sin_new = df_new1.groupby('id')['Response_date_dayofweek_sin'].mean()
mean_dayofweek_cos_new = df_new1.groupby('id')['Response_date_dayofweek_cos'].mean()


# In[93]:


last_entries_1 = df1.groupby('id').last()
last_q2_1 = last_entries_1['q2']
id_column_1 = df1['id'].unique()
last_q2_1


# In[81]:


last_entries = df_new1.groupby('id').last()
last_q2 = last_entries['q2']
id_column = df_new1['id'].unique()


# In[89]:


mean_hour_sin


# In[94]:


df1_n = pd.DataFrame({'id_column': id_column_1,
                                     'mean_hour_sin_new': mean_hour_sin,
                                     'mean_hour_cos_new': mean_hour_cos,
                                    'mean_dayofweek_sin_new': mean_dayofweek_sin,
                                     'mean_dayofweek_cos_new': mean_dayofweek_cos,
                                     'last_q2': last_q2_1}).reset_index(drop=True)


# In[95]:


df1_n


# In[11]:


df_newwest = pd.DataFrame({'id_column': id_column,
                                     'mean_hour_sin_new': mean_hour_sin_new,
                                     'mean_hour_cos_new': mean_hour_cos_new,
                                    'mean_dayofweek_sin_new': mean_dayofweek_sin_new,
                                     'mean_dayofweek_cos_new': mean_dayofweek_cos_new,
                                     'last_q2': last_q2}).reset_index(drop=True)


# In[12]:


df_newwest


# In[53]:


#We removed datetime type features and the created features without cyclical sin|cos manipulation

df1.drop(columns = ['Response_date','publish_time',
                    'Response_date_hour','Response_date_dayofweek','resolved_time'],inplace = True) 


# In[54]:


df1= df1.drop(columns= ['predictions_num','Unique_predictors','number_of_forecasters','prediction_count','id'])


# In[ ]:


from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold,StratifiedKFold
y = df1["Result"]
df_train= df1.drop(columns='Result')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[53]:


from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold,StratifiedKFold
y_new = df_newwest["last_q2"]
df_train_new= df_newwest.drop(columns= ['last_q2','id_column'])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[14]:


cv


# In[35]:


params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "max_depth": 10,  
    "learning_rate": 0.05,
    "n_estimators": 2000,  
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees":True,
    'num_leaves':64,
    "verbose": -1,
}


# In[15]:


df_train_new


# In[25]:


y.value_counts()


# In[36]:


"""
%%time

random_state =  42
import logging
import warnings
warnings.filterwarnings("ignore")
# Set LightGBM logging level to suppress warnings
logging.getLogger('lightgbm').setLevel(logging.ERROR)
fitted_models_lgb = []
cv_scores_lgb = []

for idx_train, idx_valid in cv.split(df_train_new, y_new):
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]
    
    # Define LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    
    # Train LightGBM model
    model = lgb.train(params, train_data, valid_sets=[valid_data],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)])
    
    # Evaluate model on validation data
    y_pred_valid = model.predict(X_valid)
    auc_score = roc_auc_score(y_valid, y_pred_valid)
    
    # Store model and score
    fitted_models_lgb.append(model)
    cv_scores_lgb.append(auc_score)

# Display CV AUC scores
print("CV AUC scores: ", cv_scores_lgb)
print("Maximum CV AUC score: ", max(cv_scores_lgb))

""""


# In[56]:


#Copied 100 % from chatgpt with directions

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",  # Use a regression metric
    "max_depth": 10,  
    "learning_rate": 0.05,
    "n_estimators": 2000,  
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees":True,
    'num_leaves':64,
    "verbose": -1,
}

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# Assuming df_train_new is the training dataframe and y_new is the target variable
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fitted_models_lgb = []
cv_scores_lgb = []
iters = []
learning_curves= []
for idx_train, idx_valid in kf.split(df_train_new):
    X_train, y_train = df_train_new.iloc[idx_train], y_new.iloc[idx_train]
    X_valid, y_valid = df_train_new.iloc[idx_valid], y_new.iloc[idx_valid]
    
    # Define LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    
    # Train LightGBM model
    model = lgb.train(params, train_data, valid_sets=[valid_data],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)])
    # Evaluate model on validation data
    y_pred_valid = model.predict(X_valid)
    rmse_score = mean_squared_error(y_valid, y_pred_valid, squared=False)  # RMSE
    
    # Store model and score
    fitted_models_lgb.append(model)
    cv_scores_lgb.append(rmse_score)
    learning_curves.append(rmse_score)
    iters.append(idx_train)

# Display CV RMSE scores
print("CV RMSE scores: ", cv_scores_lgb)
print("Maximum CV RMSE score: ", max(cv_scores_lgb))

# Display CV RMSE scores
print("CV RMSE scores: ", cv_scores_lgb)
print("Maximum CV RMSE score: ", max(cv_scores_lgb))


# In[96]:


from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold,StratifiedKFold
y_new_1 = df1_n["last_q2"]
df_train_n= df1_n.drop(columns= ['last_q2','id_column'])


# In[114]:


predictions = np.zeros(df_train_n.shape[0])
best_model_index = np.argmin(cv_scores_lgb)
best_model = fitted_models_lgb[best_model_index]
predictions = best_model.predict(df_train_n)


# In[115]:


df1_n['preds'] = predictions


# In[116]:


df1_n.head(50)


# In[97]:





# # QUESTIONS AND IDEAS

# ### Q: HOW CAN WE ELIMINATE ROWS DEPENDENCY?
# #### For every question, we'll only keep the variance and features that arew time related, for every question MAYBE only 1 row 
# 
# ### option1: let's draw infinite number of questions and look only at the last predictions -> not good enough, we want to refer to the timelines in our research
# #### We'll start with the mean/median an then we'll see
# 
# 
# ### Q: HOW CAN WE CREATE BETTER VALIDATION?
# #### When we fetch all the questions, we leave create test set of 20/30% of the questions which is untouchable and the rest is training, 20% percent of the training we'll not work with as well
# 
# ### A:
# 
# ### HOW MANY QUESTION SHOULD WE FETCH FROM THE API?
# #### Let's start with a lot, than narrow it down
# 
# ### SHOULD WE SET A LOWER LIMIT FOR NUM OF PREDICTORS FOR EACH QUESTION?
# #### We need to decide on a limit, not that relevant right now
# 
# ### WHICH FEATURES CAN WE ADD?
# #### Time related, we need to find articles
# 
# ### DO WE NEED MORE METACULUS HIDDEN FEATURES?
# #### We have an api token, we need to check in the documentation. NEED TO ASK MAAYAN 
# 
# ### DO WE WANT TO CREATE FEATURES BASED ON THE WORDING OF THE TITLES?
# #### FOR THE SEMINAR NO BUT WE CAN DO IT AFTERWARDS

# 
