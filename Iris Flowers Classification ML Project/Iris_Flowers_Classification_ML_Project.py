#!/usr/bin/env python
# coding: utf-8

# #                                         Lets Grow More 
#  
# ##                           Virtual Internship Program - *Data Science* (Feb 2023)
# 
# #                               Name - Vijay Prajapat
# 
# # 
# 
# ## Task 1 - Iris Flowers Classification ML Project 

# ### Importing Libraries

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# ### Connecting Google Drive with Google Colab

# In[2]:


# Connecting Google Drive with Google Colab
from google.colab import drive
drive.mount('/content/drive')


# ### Importing Data Set from google drive

# In[3]:


# Importing Data Set from google drive
import os
os.chdir('/content/drive/My Drive')


# ### Reading Data set

# In[4]:


# Reading Data set
data=pd.read_csv('iris_data.csv')
data.head()


# ### Giving Proper Heading to Columns

# In[5]:


# Giving Proper Heading to Columns
data_header = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
data.to_csv('Iris.csv', header = data_header, index = False)
new_data = pd.read_csv('Iris.csv')
new_data.head()


# ### Checking no. of rows and columns

# In[6]:


# Checking no. of rows and columns
new_data.shape


# ### Checking datatypes in dataset

# In[7]:


# Checking datatypes in dataset
new_data.info()


# ### Describing the Dataset

# In[8]:


# Describing Dataset
new_data.describe()


# ### Checking null values in Dataset

# In[9]:


# Checking Null Values in DataSet
new_data.isnull().sum()


# # Data Visualization 
# 
# ## Graphs of features vs Species

# In[10]:


# Sepal Length vs Type
plt.bar(new_data['Species'],new_data['SepalLength'], width = 0.5) 
plt.title("Sepal Length vs Type")
plt.show()


# In[11]:


# Sepal Width vs Type
plt.bar(new_data['Species'],new_data['SepalWidth'], width = 0.5) 
plt.title("Sepal Width vs Type")
plt.show()


# In[12]:


# Petal Length vs Type
plt.bar(new_data['Species'],new_data['PetalLength'], width = 0.5) 
plt.title("Petal Length vs Type")
plt.show()


# In[13]:


# Petal Width vs Type
plt.bar(new_data['Species'],new_data['PetalWidth'], width = 0.5) 
plt.title("Petal Width vs Type")
plt.show()


# ## Pair plot for Dataset

# In[14]:


sns.pairplot(new_data,hue='Species')


# # Splitting the Dataset

# In[15]:


x = new_data.drop(columns="Species")
y = new_data["Species"]


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)


# In[17]:


x_train.head()


# In[18]:


x_test.head()


# In[19]:


y_train.head()


# In[20]:


y_test.head()


# In[21]:


print("x_train: ", len(x_train))
print("x_test: ", len(x_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))


# # Building Model using Logistic Regression

# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[27]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[28]:


predict = model.predict(x_test)
print("Pridicted values on Test Data", predict)


# In[29]:


y_test_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)


# In[30]:


print("Training Accuracy : ", accuracy_score(y_train, y_train_pred))
print("Test Accuracy : ", accuracy_score(y_test, y_test_pred))


# # Conclusion
# # Hence we conclude that we did Iris Flower Classification using Logistic Regression and we got Training Accuracy: 97% and Test Accuracy: 95%.
