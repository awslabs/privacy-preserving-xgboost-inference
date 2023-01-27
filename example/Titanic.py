
# coding: utf-8

# In[1]:


# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# ( Run <code>jupyter notebook</code> under the project directory )

# In[2]:



import sys

from ppxgboost import PPModel as internalmodel
from ppxgboost import PPPrediction as ppprediction
from ppxgboost import PPQuery as ppquery
from ppxgboost import PaillierAPI as paillier
from ppxgboost.OPEMetadata import *
from ppxgboost.PPKey import *
import sys
import random
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from secrets import token_bytes
from ope.pyope.ope import OPE


# # XGBoost for Titanic Dataset
# 
# (We use this example to demenstrate how to use ppxgboost for encypting an xgboost model and query it.)
# 
# Please go to https://www.kaggle.com/c/titanic/data and download the dataset.
# In the following example, the datasets are downloaded in the example directory
# 

# ### Data Preparation and Train an XGBoost ML model

# In[3]:


# The pp-xgboost for titanic 
# In the following example, the datasets are downloaded in the example directory
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# Training dataset. We skip the data exploration part ...
# Only get the features that are useful for building the ML model
X_train = train[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']]
y_train = train[['Survived']]

# Testing dataset
X_test = test[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']]

X_train.head()


# In[4]:


y_train.head()


# In[5]:


X_test.head()


# In[6]:


# Train a xgboost model 
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'eta': 0.1}
model = xgb.train(params=params, dtrain=dtrain)

# predict using the plaintext prediction
plaintext_predict = model.predict(xgb.DMatrix(X_test))


# Dump the model

# In[7]:


model.dump_model('tree.txt')


# ### Encryption Preparation for XGBoost Model
# 1. Set up some metadata information for the dataset.
# 2. Set up the encryption materials
# 3. Encrypt the model
# 4. Encrypt the query
# 5. Perform the prediction
# 6. Decrypt the prediction

# In[8]:


# 1. parsing to internal tree data structure, and output feature set
model = internalmodel.from_xgboost_model(model)
model.discretize()
feature_set = model.get_features()
test_queries = ppquery.pandas_to_queries(X_test)
test_min, test_max = get_test_data_extreme_values(test_queries)
metadata = OPEMetadata(model, test_min, test_max)

# 2. Set up encryption materials.
start = time.time()
pp_boost_key, client_key = generatePPXGBoostKeys()
print('Key generation time: {:0.2f} seconds'.format(time.time() - start))

# 3. process the tree into enc_model
start = time.time()
enc_model = model.encrypt(pp_boost_key, metadata)
print('Model encryption time: {:0.2f} seconds'.format(time.time() - start))

# 4. Encrypt the input vector for prediction based on the feature set.
start = time.time()
q_encryptor = ppquery.QueryEncryptor(client_key, feature_set, metadata)
encrypted_queries = list(map(q_encryptor.encrypt_query, test_queries))
print('Time to encrypt ' + str(len(test_queries)) + ' queries: {:0.2f} seconds'.format(time.time() - start))

# In[9]:


# 5. privacy-preserving evaluation.
start = time.time()
encrypted_query_results = list(map(lambda q: ppprediction.predict_single_input_binary(enc_model, q), encrypted_queries))
print('Time to predict ' + str(len(test_queries)) + ' queries: {:0.2f} seconds'.format(time.time() - start))


# In[10]:


# 6. decryption
start = time.time()
decryptions = map(lambda c: paillier.decrypt(client_key.get_private_key(), c), encrypted_query_results)
decryptions = list(map(lambda x: round(x, 7), decryptions))
print('Time to decrypt ' + str(len(test_queries)) + ' results: {:0.2f} seconds'.format(time.time() - start))

assert len(plaintext_predict) == len(decryptions)

# if the predicted values are same (the ppxgboost might not produce same values 
#                                    as the plaintext value due to precision)
for i in range(len(plaintext_predict)):
    assert abs(plaintext_predict[i] - decryptions[i]) < 0.000001
