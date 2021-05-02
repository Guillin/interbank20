#!/usr/bin/env python
# coding: utf-8

# # EXPANDING WINDOW SPLIT

# ### LOAD LIBRARIES

# In[ ]:


import os
import gc
import warnings
import pandas as pd
import numpy as np
import pickle
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)


# ### GLOBAL VARIABLES

# In[ ]:


INPUT_PATH = '../../data/features'
OUTPUT_PATH = '../../data/train_test'
INPUT_FILE_NAME = 'features_v001'
N_SPLITS = 3 # numbers of folds
DAY_COL = 'd'
DATE_COL = "date"
D_THRESH = 1941 - int(365 * 2) # he only left 2 years of training data, from 2014-05-23 to 2016-05-24
DAYS_PRED = 28


# ### FUNCTIONS

# In[ ]:


def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# ### LOAD DATASET

# In[ ]:


print("Reading files...")
data = pd.read_pickle(f'{INPUT_PATH}/{INPUT_FILE_NAME}.pkl').pipe(reduce_mem_usage)


# In[ ]:


train = data[data.part == 'train']


# In[ ]:


test = data[data.part == 'validation']


# ### SPLIT DATASET

# In[ ]:


features = [c for c in data.columns if c not in [DAY_COL, 'id', 'part', 'demand']]


# In[ ]:


features_types = data[features].dtypes.reset_index() # save feature name in order to be used to show which are more important
features_types.columns = ['feature', 'type']


# In[ ]:


del data


# In[ ]:


D_THRESH = train.d.max() - DAYS_PRED

X_train = train[train.d <= D_THRESH][features].reset_index(drop=True)
y_train = train[train.d <= D_THRESH]["demand"].reset_index(drop=True)

X_val = train[train.d > D_THRESH][features].reset_index(drop=True)
y_val = train[train.d > D_THRESH]["demand"].reset_index(drop=True)


X_test  = test[features].reset_index(drop=True)


# In[ ]:


del train,test
gc.collect()


# In[ ]:


print("X_train shape:", X_train.shape)
print("X_val shape:",   X_val.shape)
print("X_test shape:",  X_test.shape)


# ### SAVE DATASET

# In[ ]:


pickle.dump(X_train,      open(f'{OUTPUT_PATH}/X_train.pkl', 'wb'))
pickle.dump(X_val,        open(f'{OUTPUT_PATH}/X_val.pkl',   'wb'))
pickle.dump(y_train,      open(f'{OUTPUT_PATH}/Y_train.pkl', 'wb'))
pickle.dump(y_val,        open(f'{OUTPUT_PATH}/Y_val.pkl',   'wb'))
pickle.dump(X_test,       open(f'{OUTPUT_PATH}/X_test.pkl',  'wb'))


# In[ ]:


features_types.to_csv(f'{OUTPUT_PATH}/{INPUT_FILE_NAME}_info.csv', index=False)

