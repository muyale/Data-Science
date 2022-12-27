#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[4]:


usa = pd.read_csv("C:\\Datasets\\USA REAL ESTATE\\realtor-data.csv")
usa.head()


# In[5]:


usa.drop( ['full_address','street','sold_date'], axis=1,inplace=True)


# In[6]:


usa.head()
usa.dropna()


# In[7]:


usa['status'].value_counts()
#This is to check on the unique value in our columns
def unique_value(column,df=usa):
    return  df[column].value_counts()


# In[8]:


unique_value('state')


# In[9]:


def total_unique():
    for column in usa.columns:
        print(unique_value(column))


# In[10]:


total_unique()


# In[11]:


def top_ten(column,df=usa):
    return df[column].value_counts().head(10)


# In[12]:


top_ten(usa.columns)


# In[13]:


def important_statistics(df=usa):
    for columns in df.columns:
        print (df[column].describe)
        


# In[14]:


#important_statistics()


# In[15]:


#usa.columns


# In[16]:


usa.head()


# In[17]:


usa['bath'].value_counts().head()


# In[18]:


# Create a column called total rooms which we will use the bedrooms +the bathrooms
usa['total_rooms']= usa['bed']+usa['bath']


# In[19]:


usa.head()


# In[20]:


usa['total_rooms'].value_counts().head()


# In[21]:


top_ten(usa.columns)


# In[22]:


plt.pie(x=top_ten(usa.columns))


# In[23]:


# A correlation heatmap ,for those correlated columns in our data sets
correlations = usa.corr()
correlations
# Clearly all are correlated with price


# In[24]:


# To check how price compares for each of the correlated columns,I will use scatter plot
def price_comparison(column,df = usa):
    sns.displot(x=df[column])
    plt.xlabel(f"{column}")
    plt.title(f"How{column}compares with price")
    plt.show()
    sns.pairplot(correlations)


# In[25]:


price_comparison('bed')


# In[ ]:


price_comparison('acre_lot')


# In[ ]:


sns.pairplot(correlations)
#The all have a linear relationship


# In[ ]:


plt.histplot(x =usa['bed'],y=usa['price'])


# In[ ]:





# In[ ]:


feature_columns = ["bed","bath","acre_lot","zip_code","house_size"]
X=usa[feature_columns]
y =usa['price']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)


# In[ ]:


from sklearn.linear_model import LinearRegression()
model = LinearRegression()
model.fit(X_train,y_train)
predictor = model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print(f"The Mean Squared error is {mean_squared_error(y_test,predictor)}")
print(f"The Mean absoluteerror is {mean_absolute_error(y_test,predictor)}")


# In[ ]:


sns.scatterplot(y_test,predictor)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




