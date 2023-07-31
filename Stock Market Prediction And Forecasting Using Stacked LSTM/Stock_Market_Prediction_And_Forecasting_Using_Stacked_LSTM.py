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
# ## Task 2 - Stock Market Prediction And Forecasting Using Stacked LSTM 
# 
# #Importing Libraries

# In[48]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Importing Data Set

# In[2]:


url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
data = pd.read_csv(url)
data


# ## Describing the Dataset

# In[3]:


data.describe()


# In[4]:


data.tail()


# In[5]:


data.dtypes


# In[6]:


data['Date'].value_counts()


# In[7]:


data['High'].hist()


# In[8]:


plt.figure(figsize=(20,8))
data.plot()


# In[9]:


data_set = data.filter(['Close'])
dataset = data.values
training_data_len=math.ceil(len(data) * 8)
training_data_len


# In[10]:


dataset


# In[11]:


data = data.iloc[:, 0:5]
data


# In[13]:


training_set = data.iloc[:, 1:2].values
training_set


# ## Scalling of Data Set

# In[14]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
data_training_scaled = scaler.fit_transform(training_set)


# In[15]:


features_set = []
labels = []
for i in range(60, 586):
  features_set.append(data_training_scaled[i - 60:i, 0])
  labels.append(data_training_scaled[i, 0])


# In[16]:


features_set, labels = np.array(features_set), np.array(labels)


# In[17]:


features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
features_set.shape


# ## Building The LSTM

# In[18]:


import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM


# In[19]:


model = Sequential()


# In[20]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[21]:


model.fit(features_set, labels, epochs=50, batch_size=20)


# In[22]:


data_testing_complete = pd.read_csv(url)
data_testing_processed = data_testing_complete.iloc[:, 1:2]
data_testing_processed


# ## Prediction of the Data

# In[23]:


data_total = pd.concat((data['Open'], data['Open']), axis=0)


# In[24]:


test_inputs = data_total[len(data_total) - len(data) - 60:].values
test_inputs.shape


# In[25]:


test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)


# In[33]:


test_feature = []
for i in range(60, 89):
  test_feature.append(test_inputs[i-60:i, 0])


# In[34]:


test_feature = np.array(test_feature)
test_feature = np.reshape(test_feature, (test_feature.shape[0] - test_feature.shape[1], 1))
test_feature.shape


# In[30]:


predictions = model.predict(test_features)


# In[31]:


predictions


# In[36]:


x_train = data[0:1256]
y_train = data[1:1257]
print(x_train.shape)
print(y_train.shape)


# In[37]:


x_train


# In[38]:


np.random.seed(1)
np.random.randn(3, 3)


# ## Drawing a Single number from the Normal Distribution

# In[39]:


np.random.normal(1)


# ## Drawing 5 numbers from Normal Distribution

# In[40]:


np.random.normal(5)


# In[41]:


np.random.seed(42)


# In[42]:


np.random.normal(size=1000, scale=100).std()


# ## Ploting Results

# In[43]:


plt.figure(figsize=(18,6))
plt.title("Stock Market Price Prediction")
plt.plot(data_testing_complete['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Total Trade Quantity', fontsize=18)
plt.show()


# ## Analyze the Closing price from the dataframe

# In[44]:


data["Date"] = pd.to_datetime(data.Date)
data.index = data['Date']

plt.figure(figsize=(20, 10))
plt.plot(data["Open"], label='ClosePriceHist')


# In[45]:


plt.figure(figsize=(12,6))
plt.plot(data['Date'])
plt.xlabel('Turnover (Lacs)', fontsize=18)
plt.ylabel('Total Trade Quantity', fontsize=18)
plt.show()


# ## Analyze the Closing price from the dataframe

# In[46]:


data["Turnover (Lacs)"] = pd.to_datetime(data.Date)
data.index = data['Turnover (Lacs)']

plt.figure(figsize=(20, 10))
plt.plot(data["Turnover (Lacs)"], label='ClosePriceHist')


# In[49]:


sns.set(rc = {'figure.figsize': (20, 5)})
data['Open'].plot(linewidth = 1,color='blue')


# In[50]:


data.columns


# In[51]:


df = pd.read_csv(url)
df


# In[52]:


cols_plot = ['Open','High','Low','Last','Close']
axes = df[cols_plot].plot(alpha = 1, figsize=(20, 30), subplots = True)

for ax in axes:
    ax.set_ylabel('Variation')

