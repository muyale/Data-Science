#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[9]:


df = pd.read_csv("C:\\Datasets\\wine-clustering.csv")
df.head()


# In[28]:


df.describe()


# In[11]:


df.isnull().sum()


# In[35]:


sns.pairplot(df)
plt.title('A pairplot for our dataset')


# In[36]:


sns.heatmap(df.corr())
plt.show()


# In[37]:


df.corr()


# In[38]:


sns.lmplot(x='Alcohol',y='Proline',data=df)
plt.title('Linear Plot of Alcohol and Proline')
plt.show()


# In[39]:


df.columns


# In[54]:


features = ['Ash_Alcanity','Nonflavanoid_Phenols','Hue','Alcohol']
feature_variables= df.drop(features,axis=1)
feature_variables
X = feature_variables
y = df['Alcohol']


# In[55]:


feature_variables.corr()
#The negative correlation is a good sign for our feature variables ,they have a positive correlation with our target variable


# In[56]:


# How alcohol is distributed interms of percentage
sns.histplot(x=df['Alcohol'])
plt.title('The Distribution of Alcohol interms of percentage content')
plt.ylabel('Alcohol Content in %')
plt.show()


# In[57]:


sns.violinplot(x=df['Alcohol'])
plt.title('The Distribution of Alcohol interms of percentage content')
plt.ylabel('Alcohol Content in %')
plt.show()


# In[58]:


sns.displot(x=df['Alcohol'],kde=True)
plt.title('The Distribution of Alcohol interms of percentage content')
plt.ylabel('Alcohol Content in %')
plt.show()


# In[59]:


df['Alcohol'].median()


# In[60]:


# How does proline affect the alcoholic percentage and viceversa?
sns.jointplot(x='Proline',y='Alcohol',data=df)
plt.title('A PLOT SHOWING HOW ALCOHOL % IS AFFECTED BY PROLINE')
plt.xlabel('Proline in Alcohol')
plt.ylabel('Alcohol %')
plt.show()


# In[86]:


"""I will first fit a Linear Regression Model and get some Insights ,then I will do a Multivariate Regression 
and Compare the Results"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 101)
from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(X_train,y_train)
reg_pred = reg_model.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
print(f"Our Mean Squared error is {mean_squared_error(y_test,reg_pred)}")
print(f"Our Mean Absolute error is {mean_absolute_error(y_test,reg_pred)}")
plt.scatter(y_test,reg_pred)
plt.show()


# In[64]:


from sklearn.ensemble import RandomForestRegressor
rfc= RandomForestRegressor()
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)
print(f"Our Mean Squared error is {mean_squared_error(y_test,rfc_preds)}")
print(f"Our Mean Absolute error is {mean_absolute_error(y_test,rfc_preds)}")


# In[80]:


"""PRINCIPAL COMPONENT ANALYSIS """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
pca = PCA(n_components = 2)
pca.fit(scaled_data)


# In[82]:


x_pca = pca.transform(scaled_data)
x_pca.shape


# In[85]:



fig=plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['Alcohol'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.title('First and Second Principal Components')
plt.show()


# In[ ]:




