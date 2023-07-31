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
# ## Task 6 - Prediction using Decision Tree Algorithm

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Connecting Google Drive with Google Colab

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# ## Importing Data Set from google drive

# In[3]:


import os
os.chdir('/content/drive/My Drive')


# ## Reading The Data Set

# In[4]:


data=pd.read_csv('Iris.csv')
data.head()


# In[5]:


data.tail()


# ## Getting the Size of Data

# In[6]:


data.shape


# In[7]:


data.columns


# ## Checking for Null Values

# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# ## Getting some Statistical Inference from the Data

# In[10]:


data.describe(include='all')


# ## Data Visualization

# In[12]:


count = data['Species'].value_counts()
count.to_frame()


# In[13]:


label = count.index.tolist()
val = count.values.tolist()


# In[14]:


exp = (0.05,0.05,0.05)
fig,ax = plt.subplots()
ax.pie(val, explode=exp, labels=label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Different Species of flower present in the Data")
ax.axis('equal')
plt.show()


# In[15]:


sns.pairplot(data=data, hue='Species')
plt.show()


# In[16]:


corr = data.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')


# ## Data Preparation

# In[17]:


data = data.drop('Id', axis=1)
data.head()


# In[19]:


x = data.iloc[:, 0:4]
x.head()


# In[20]:


y = (data.iloc[:, 4])
y.head().to_frame()


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


std = StandardScaler()
x = std.fit_transform(x)


# In[23]:


from sklearn.model_selection import train_test_split


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)


# ## Model Creation

# In[26]:


from sklearn.tree import DecisionTreeClassifier


# In[27]:


model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# ## Prediction using the Created Model

# In[28]:


y_pred = model.predict(x_test)
y_pred


# ## Model Evaluation

# In[29]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[30]:


acc = accuracy_score(y_test, y_pred)
print("The Accuracy of the Decision Tree Algorithms is : ", str(acc*100) + "%")


# In[31]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[32]:


lst = data['Species'].unique().tolist()
df_cm = pd.DataFrame(data = cm, index = lst, columns = lst)
df_cm


# ## Data Visualization for the Model

# In[33]:


data.columns


# In[35]:


col = data.columns.tolist()
print(col)


# In[37]:


from sklearn.tree import plot_tree


# In[39]:


fig = plt.figure(figsize=(25, 20))
tree_img = plot_tree(model, feature_names = col, class_names = lst, filled = True)

