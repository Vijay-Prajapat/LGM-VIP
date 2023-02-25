#!/usr/bin/env python
# coding: utf-8

# #                                         Lets Grow More 
#  
# ##                           Virtual Internship Program - *Data Science* (Feb 2023)
# 
# #                                 Name - Vijay Prajapat
# 
# # 
# 
# ## Task 3 - Music Recommendations
# 
# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm


# # Connecting Google Drive with Google Colab

# In[3]:


# Connecting Google Drive with Google Colab
from google.colab import drive
drive.mount('/content/drive')


# # Importing Data Set from google drive

# In[4]:


# Importing Data Set from google drive
import os
os.chdir('/content/drive/My Drive')


# # Loading Data

# In[5]:


ntr = 7000
nts = 3000

train = pd.read_csv('train.csv',nrows=ntr)

names=['msno','song_id','source_system_tab','source_screen_name',\
      'source_type','target']

test1 = pd.read_csv('train.csv',names=names,skiprows=ntr,nrows=nts)

songs = pd.read_csv('songs.csv')

members = pd.read_csv('members.csv')


# # Analysing Data

# In[6]:


train.head()


# In[7]:


test1.head()


# In[8]:


songs.head()


# In[9]:


members.head()


# # Data Processing

# In[10]:


test = test1.drop(['target'],axis=1)
ytr = np.array(test1['target'])


# In[11]:


test.head()


# # Rearranging column so data fits into code

# In[12]:


test_name = ['id','msno','song_id','source_system_tab',\
             'source_screen_name','source_type']
test['id']=np.arange(nts)
test = test[test_name]


# In[13]:


test.head()


# In[14]:


songs.head()


# In[15]:


print('Data preprocessing...')

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')


# # Splitting into year, month and date

# In[16]:


members.head()


# In[17]:


members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))


# In[18]:


members.head()


# # Dropping Registration time column

# In[19]:


members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)


# In[20]:


members.head()


# # Left join with training and testing data

# In[21]:


members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')


# In[22]:


train = train.fillna(-1)
test = test.fillna(-1)


# In[23]:


train.head()


# In[24]:


test.head()


# # Collection work for members and songs

# In[25]:


import gc
del members, songs; gc.collect();


# In[26]:


cols = list(train.columns)
cols.remove('target')


# In[27]:


train.head(6)


# In[28]:


test.head()


# # LabelEncoder

# In[29]:


for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])


# # Song Popularity

# In[30]:


unique_songs = range(max(train['song_id'].max(), test['song_id'].max()))
song_popularity = pd.DataFrame({'song_id': unique_songs, 'popularity':0})

train_sorted = train.sort_values('song_id')
train_sorted.reset_index(drop=True, inplace=True)
test_sorted = test.sort_values('song_id')
test_sorted.reset_index(drop=True, inplace=True)


# # Model Training and Prediction

# In[31]:


# User library size
X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values

del train, test; gc.collect();

X_train, X_valid, y_train, y_valid = train_test_split(X, y, \
    test_size=0.1, random_state = 12)
    
del X, y; gc.collect();

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]


# In[32]:


print('Training LGBM model...')

params = {}

params['learning_rate'] = 0.4
params['application'] = 'binary'
params['max_depth'] = 15
params['num_leaves'] = 2**8
params['verbosity'] = 0
params['metric'] = 'auc'

model = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, \
early_stopping_rounds=10, verbose_eval=10)


# In[33]:


print('Making predictions and saving them...')
p_test = model.predict(X_test)
p_test


# # Creating csv file to store id and target columns

# In[34]:


subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')


# In[35]:


yhat = (p_test>0.5).astype(int)
comp = (yhat==ytr).astype(int)
acc = comp.sum()/comp.size*100
print('The accuracy of lgbm model on test data is: {0:f}%'.format(acc))


# In[36]:


rd_seed = np.random.uniform(0,1,nts)
yhat_rand = (rd_seed>0.5).astype(int)
comp_rand = (yhat_rand==ytr).astype(int)
acc_rand = comp_rand.sum()/comp_rand.size*100
print('The accuracy of random model on test data is: {0:f}%'.format(acc_rand))


# # Conclusion:
# ## The Accuracy of lgbm model is 78.53%
# 
# ## The Accuracy of Random model is 49.30%
# 
# ## Hence we conclude that lgbm model is better than random model
