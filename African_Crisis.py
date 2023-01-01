#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[114]:


africa_df = pd.read_csv("C:\\Users\\EDGAR MUYALE DAVIES\\Downloads\\african_crises.csv")
africa_df = africa_df.head(500)


# In[115]:


africa_df.head()
africa_df.isnull().sum()


# In[116]:


africa_df.columns
#africa_df['cc3'].head()


# In[117]:


africa_df['banking_crisis'].value_counts()


# In[118]:


africa_df['case'].nunique()


# In[119]:


# Frequency of countries in our data set
africa_df['country'].value_counts()


# In[120]:


africa_df.columns


# In[121]:


def insight(df=africa_df):
    """Args: 
    df: DataFrame under study
    column:The column that we intend to derive important information from
    Using a loop to get every individual aspect from our data set,using the value count this function counts all the unique values 
    and returns the top 10 appearing values in all categorical categories
    Returns:
    
    """
    for column in df.columns:
        print(df[column].describe())
        print(df[column].value_counts().head())


# In[122]:


insight()


# In[123]:


"""EXPLOLATORY DATA ANALYSIS AND VISUAL REPRESENTATION"""


# In[124]:


africa_df.head()


# In[64]:


africa_df['banking_crisis'] = africa_df['banking_crisis'].replace(to_replace='no_crisis',value='0')
africa_df['banking_crisis'] = africa_df['banking_crisis'].replace(to_replace='crisis',value='1')


# In[65]:


africa_df['banking_crisis'].head()


# In[66]:


africa_df['inflation_crises'].head()


# In[67]:


correlation = africa_df.corr()


# In[68]:


sns.heatmap(correlation)


# In[69]:


correlation


# In[70]:


sns.pairplot(correlation)


# In[71]:


# How do we get the feature importance in order?
# first we Normalize the data


# In[72]:


africa_df.columns


# In[73]:


# Normalizing the data
africa_df['year_norm'] = africa_df.year-africa_df.year.min()/africa_df.year.max()-africa_df.year.min()
africa_df['exchange_usd_norm'] = africa_df.exch_usd-africa_df.exch_usd.min()/africa_df.exch_usd.max()-africa_df.exch_usd.min()
africa_df['gdp_weighted_default_norm'] = africa_df.gdp_weighted_default-africa_df.gdp_weighted_default.min()/africa_df.gdp_weighted_default.max()-africa_df.gdp_weighted_default.min()
africa_df['inflation_annualcpi_norm'] = africa_df.inflation_annual_cpi-africa_df.inflation_annual_cpi.min()/africa_df.inflation_annual_cpi.max()-africa_df.inflation_annual_cpi.min()


# In[74]:


africa_df.columns


# In[75]:


africa_df.columns
 


# In[103]:


features_list = ['case', 'exch_usd','year',
       'domestic_debt_in_default', 'sovereign_external_debt_default',
       'gdp_weighted_default', 'inflation_annual_cpi', 'independence',
       'currency_crises', 'inflation_crises', 'banking_crisis', 'year_norm',
       'exchange_usd_norm', 'gdp_weighted_default_norm',
       'inflation_annualcpi_norm']
X = africa_df[features_list]
y = africa_df.systemic_crisis


# In[104]:


# We will use logistic regression ,more specifically the coefficient to know the importance of each feature


# In[105]:


from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
lrmodel.fit(X,y)


# In[106]:


importance = lrmodel.coef_[0]


# In[107]:


feature_importance = {'Features':features_list,'Coef_Score':importance}


# In[108]:


feature_importance = pd.DataFrame(feature_importance)
feature_importance =feature_importance.sort_values(by='Coef_Score',ascending =True)
feature_importance


# In[109]:


# From our result ,inflation and domestic debt in default are negatively correlated with the systemic crises
# I will now visualize the data
plt.barh(y=feature_importance.Features,width = feature_importance.Coef_Score)
plt.title('Bar Chart of Logistic Regression Coefficients as Feature Importance Scores')


# In[110]:


# Now we try using the Random Forest Classifier
from sklearn.ensemble import RandomForestRegressor
RFmodel = RandomForestRegressor()
RFmodel.fit(X,y)
RF_importance = RFmodel.feature_importances_


# In[100]:


# As was the case with our logistic model we will create a data frame
RF_feature_importance = {'Features':features_list,'Imp_Score':RF_importance}
RF_feature_importance = pd.DataFrame(RF_feature_importance) 
RF_feature_importance = RF_feature_importance.sort_values(by ='Imp_Score',ascending = True)
RF_feature_importance


# In[111]:


# For the visualization
plt.barh(y=RF_feature_importance.Features,width =RF_feature_importance.Imp_Score)
plt.title('Bar Chart of Random Forest Regression coefficients as Feature Importances'


# The most corelated ar banking crisis,exchange usd,year and inflation cpi


"""THE INFLATION  RATE AND INFLATION CRISIS OF KENYA"""
plt.figure(figsize=(10,8))
plt.scatter(x=africa_df.year[(africa_df.country=='Kenya')&(africa_df.inflation_crises == 1 )],y=africa_df.inflation_annual_cpi[(africa_df.country=='Kenya')&(africa_df.inflation_crises == 1 )],c='tomato',label ='Inflation_Crisis')
plt.scatter(x= africa_df.year[(africa_df.country == 'Kenya')&(africa_df.inflation_crises ==0)],y=africa_df.inflation_annual_cpi[(africa_df.country =='Kenya')&(africa_df.inflation_crises ==0)],c='mediumseagreen',label ='No_Inflation_Crisis')
plt.grid()
plt.xticks()
plt.xlabel('Year')
plt.ylabel('Inflation Rate')
plt.title('Inflation Rate in Kenya')
plt.legend()
plt.show()



