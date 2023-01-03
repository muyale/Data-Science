#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("C:\\Datasets\\Employment on leaving university - ALL INSTITUTIONS.csv")
df.head()
df.isnull().sum()


# In[4]:


df.dropna()


# In[5]:


df.Institution.nunique()
df.Institution.value_counts()
df.columns


# In[11]:


df.columns


# In[10]:


df.head()


# In[39]:


total_working  = df['Number \nemployed \nor studying'].sum()
total_working
total_unemployed = df['Unemployed']
total_unemployed
total_population = df['Eligible \npopulation'] +  df['Base \npopulation']
percentage_unemployment = (total_unemployed/total_population)*100
percentage_unemployment
percentage_working =(total_working/total_population)*100
df['Percentage_Employed'] = 100- df['Percentage_Unemployed']
df.sort_values(by='Percentage_Unemployed',ascending=True)


# In[23]:


plt.barh(width=df['Eligible \npopulation'],y=df.Percentage_Unemployed)
plt.xlabel('Eligible Population')
plt.ylabel('Percentage_Unemployment')
plt.grid()
plt.title('Does a  higher population have more unemployed people?')


# In[15]:


sns.pairplot(df)


# In[13]:


plt.scatter(df['Eligible \npopulation'],df.Percentage_Unemployed)
plt.xlabel('Eligible Population')
plt.ylabel('Unemployed Percentage')
plt.xticks()
plt.title('A scatter plot showing Percentage of Unemployed in an Eligible Population')


# In[13]:


sns.violinplot(x=df['Eligible \npopulation'],y=df.Percentage_Unemployed)
plt.xlabel('Eligible Population')
plt.ylabel('Unemployed Percentage')
plt.xticks()
plt.title('A violin plot showing Percentage of Unemployed in an Eligible Population')


# In[16]:


# SAME FOR THE EMPLOYED POPULATION
plt.barh(width=df['Eligible \npopulation'],y=df.Percentage_Employed)
plt.xlabel('Eligible Population')
plt.ylabel('Percentage_Unemployment')
plt.grid()
plt.title('Does a  higher population have more employed people?')


# In[17]:


plt.scatter(x=df['Eligible \npopulation'],y=df.Percentage_Employed)
plt.xlabel('Eligible Population')
plt.ylabel('Unemployed Percentage')
plt.xticks()
plt.title('A scatter plot showing Percentage of Employed in an Eligible Population')


# In[24]:


sns.histplot(x=df['Eligible \npopulation'],y=df.Percentage_Employed,data=df)
plt.xlabel('Eligible Population')
plt.ylabel('Unemployed Percentage')
plt.xticks()
plt.title('A scatter plot showing Percentage of Employed in an Eligible Population')


# In[25]:


df.columns


# In[26]:


df.drop(['Bench-\nmark \n(%)', 'Standard \ndeviation \n(%)', '+/-'],axis= 1,inplace=True)


# In[27]:


df.columns


# In[28]:


# Now for the correlation and feature importance
correlation  = df.corr()
correlation


# In[29]:


sns.heatmap(correlation)


# In[17]:


df['Unemployed'] = df['Eligible \npopulation']-df['Number \nemployed \nor studying']
df['Percentage_Unemployed']= (df['Unemployed']/df['Eligible \npopulation'])*100
df.dropna()
Population =df['Eligible \npopulation']
Unemployed = df['Unemployed'] 
sns.jointplot(x=Population,y=Unemployed)
plt.title('How population compares with unemployment')


# In[ ]:




