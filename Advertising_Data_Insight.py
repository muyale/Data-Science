#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import flask


# In[38]:


df= pd.read_csv("C:\\Datasets\\advertising.csv")


# In[39]:


df.head()


# In[40]:


df.isnull()


# In[41]:


def top_ten(column,df=df):
    """Args:
    column(str):This is the olumn we intend to count 
    df(DataFrame):Our data frame whose attributes we intend to study
    This function returns the most common ten appearing items in a column
    Returns:
    """
    return df[column].value_counts().head(10)


# In[19]:


top_ten('Age')


# In[21]:


top_ten('City')


# In[22]:


top_ten('Ad Topic Line')


# In[23]:


top_ten('Country')


# In[69]:


def data_plot(column,df=df):
    data = top_ten(column)
    sns.histplot(data=data,x=df[column])


# In[72]:


def comparison(colA,colB,df=df):
    x=df[colA]
    y=df[colB]
    plt.xlabel(colA)
    plt.ylabel(colB)
    plt.title(f"this is a representation of:{colA}vs{colB}")
    sns.scatterplot(x=x,y=y,data=df)
    plt.show()
              


# In[73]:


comparison('Area Income','Daily Internet Usage')


# In[76]:


sns.heatmap(df.corr())


# In[79]:


sns.countplot(x= df['Male'])
plt.title('Total Number of Male and Female participating')


# In[102]:


def check(column,df=df):

    sns.distplot(df[column],kde = False)
    plt.title(f"How {column}was distributed")
    


# In[103]:


check('Age')


# In[104]:


check('Area Income')


# In[99]:


check('Daily Time Spent on Site')


# In[ ]:




