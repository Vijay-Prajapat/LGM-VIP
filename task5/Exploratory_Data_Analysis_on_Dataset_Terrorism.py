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
# ## Task 1 - Exploratory Data Analysis on Dataset - Terrorism

# In[1]:


# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[4]:


# Connecting Google Drive with Google Colab
from google.colab import drive
drive.mount('/content/drive')


# In[5]:


# Importing Data Set from google drive
import os
os.chdir('/content/drive/My Drive')


# In[16]:


# Reading Data set
data=pd.read_csv('globalterrorismdb_0718dist.csv', encoding ='latin1')
data.head()


# In[17]:


# Describing the data
data.describe()


# In[18]:


# Data Info
data.info()


# In[19]:


# Data Types
data.dtypes


# In[20]:


# Data Set Columns
data.columns


# In[21]:


data.columns.values


# In[22]:


# Taking out required Columns for Analysis
data=data[['eventid', 'iyear', 'imonth', 'country', 'region','provstate','city','crit1', 'crit2', 'crit3','success', 'suicide', 'attacktype1','targtype1','natlty1','gname','guncertain1','claimed','weaptype1','nkill','nwound']]
data.head()


# In[23]:


# Returning Number of Missing values
data.isnull().sum()


# In[24]:


# Combining Two Columns i.e., 'nkill' and 'nwound' into a new column 'casualities'
data['nkill']=data['nkill'].fillna(0)
data['nwound']=data['nwound'].fillna(0)
data['casualities']=data['nkill']+data['nwound']
data.isnull().sum()


# In[25]:


data.describe()


# In[31]:


print(f"""
There are {data.country.nunique()} Countries from {data.region.nunique()} Regions covered in the dataset terrorist attack data in {data.claimed.nunique()} years from {data.iyear.min()} to {data.iyear.max()}. Overall {data.index.nunique()} terrorist attacks are recorded here which caused about {data.casualities.sum()} Casualities Consisted of {data.nkill.sum()} Kills and {data.nwound.sum()} Wounded.
""")


# # Data Visualization

# In[52]:


plt.subplots(figsize=(15,6))
sns.countplot('iyear', data=data, palette='RdYlGn_r', edgecolor=sns.color_palette('dark', 10))
plt.xticks(rotation = 90)
plt.title("Number of Terrorist Activities Each Year")
plt.show()


# In[27]:


yearc=data[['iyear','casualities']].groupby('iyear').sum()
yearc.plot(kind='bar',color='red',figsize=(15,6))
plt.title("Casualities")
plt.xlabel('Years',fontsize=15)
plt.ylabel('No. of Casualities',fontsize=15)
plt.show()


# In[28]:


data['attacktype1'].value_counts().plot(kind='bar',figsize=(20,10),color='blue')
plt.xticks(rotation=50)
plt.xlabel('Attack Type',fontsize=20)
plt.ylabel('Number of attacks')
plt.title('Number of attacks')
plt.show()


# In[29]:


plt.subplots(figsize=(20,10))
sns.countplot(data['targtype1'],order=data['targtype1'].value_counts().index,palette='gist_heat',edgecolor=sns.color_palette("crest"));
plt.xticks(rotation=90)
plt.xlabel('Attack type',fontsize=20)
plt.ylabel('count')
plt.title('Type of attack')
plt.show()


# In[47]:


plt.subplots(figsize=(15,6))
country_attacks = data.gname.value_counts()[:15].reset_index()
country_attacks.columns = ['gname', 'Total Attacks']
sns.barplot(x = country_attacks.gname, y = country_attacks['Total Attacks'], palette='OrRd_r', edgecolor=sns.color_palette('dark', 10))
plt.xticks(rotation = 90)
plt.title("Number of Total Attacks in Each Country")
plt.show()


# In[48]:


sattk=data.success.value_counts()[:10]
sattk


# In[49]:


data.gname.value_counts()[1:11]


# # Conclusions
# ### There are 205 Countries from 12 Regions covered in the dataset terrorist attack data in 3 years from 1970 to 2017. Overall 181691 terrorist attacks are recorded here which caused about 935737.0 Casualities Consisted of 411868.0 Kills and 523869.0 Wounded.
# 
# 1.Taliban has done most of attacks
# 2.Most of the attacks were made in the year 2014
# 3.bombing type attack were used most of time
