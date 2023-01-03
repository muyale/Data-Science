#!/usr/bin/env python
# coding: utf-8

# In[2]:


# this is a project to highlight the main socio economic indicators and their trends overtime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# Reading in the data
df = pd.read_csv("C:\\Users\\EDGAR MUYALE DAVIES\\Downloads\\indicators_ken.csv")
df.head()
df = df.rename(columns={'Indicator Name':'indicator_name'})
df.head(21)


# In[4]:


df= df.sort_values(by='Year',ascending=True)
df.head()
df.isnull().sum()
df.tail(30)
df.iloc[14976]['indicator_name']


# In[5]:


df['indicator_name'].value_counts().tail(10)


# In[6]:


value=df[df['indicator_name']=='Agriculture, forestry, and fishing, value added (% of GDP)']['Value']
year_of_value =df[df['indicator_name']=='Agriculture, forestry, and fishing, value added (% of GDP)']['Year']
year_of_value
sns.lineplot(x=year_of_value,y=value)
plt.title('Agriculture as an indicator of growth over the years')
#agriculture has been slowly decreasing and this can be attributed to the rise of white collar jobs,in2020 it had started steadily rising


# In[7]:


value=df[df['indicator_name']=='Life expectancy at birth, female (years)']['Value']
year_of_value =df[df['indicator_name']=='Life expectancy at birth, female (years)']['Year']
fig = plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(year_of_value,value,color='black')
plt.title('Female Life Expectancy at Birth over the years')
plt.subplot(2,2,2)
plt.plot(df[df['indicator_name']=='Net official development assistance received (current US$)']['Year'],df[df['indicator_name']=='Net official development assistance received (current US$)']['Value'],color='grey')
plt.title('Net Official Development Assistance received')
plt.subplot(2,2,3)
plt.plot(df[df['indicator_name']=='Net migration']['Year'],df[df['indicator_name']=='Net migration']['Value'],color='grey')
plt.title('Net Migration as Economic Indicators in the Kenyan economy')
plt.subplot(2,2,4)
plt.plot(df[df['indicator_name']=='Merchandise imports from low- and middle-income economies in Middle East & North Africa (% of total merchandise imports)']['Year'],df[df['indicator_name']=='Merchandise imports from low- and middle-income economies in Middle East & North Africa (% of total merchandise imports)']['Value'],color='grey')
plt.title('Merchandise Import from Middle East and North Africa')
plt.subplot(2,3,1)
plt.plot(df[df['indicator_name']=='Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)']['Year'],df[df['indicator_name']=='Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)']['Value'],color='yellow')
plt.title('Merchandise Import from Sub Saharan Africa')


# In[8]:


#Agricultural machinery, tractors
fig=plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(df[df['indicator_name']=='Agricultural machinery, tractors']['Year'],df[df['indicator_name']=='Agricultural machinery, tractors']['Value'],color='grey')
plt.title('The use of Machinery and Tractors in Agriculture')
plt.grid()
plt.show()
plt.subplot(2,2,4)
value=df[df['indicator_name']=='Agriculture, forestry, and fishing, value added (% of GDP)']['Value']
year_of_value =df[df['indicator_name']=='Agriculture, forestry, and fishing, value added (% of GDP)']['Year']
plt.plot(year_of_value,value)
plt.grid()
plt.title('Agriculture changing overtime')


# In[9]:


"""WITH THE INVENTION OF MACHINERY AGRICULTURE SEEMS TO THRIVE"""


# In[10]:


#Only year and value have a correlation
corr = df.corr()
corr


# In[100]:


value=df[df['indicator_name']=='Agriculture, forestry, and fishing, value added (% of GDP)']['Value']
year_of_value =df[df['indicator_name']=='Agriculture, forestry, and fishing, value added (% of GDP)']['Year']
sns.jointplot(x=year_of_value, y= value)
plt.grid()
plt.title('Agriculture changing overtime')


# In[18]:


"""What role has primary school enrollment and secondary school enrollment played in adding value to the economy?
To answer this I will plot a line plot and a joint plot for the two factors in the years"""
fig =plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(df[df['indicator_name']=='School enrollment, primary and secondary (gross), gender parity index (GPI)']['Year'],df[df['indicator_name']=='School enrollment, primary and secondary (gross), gender parity index (GPI)']['Value'],color='grey')
plt.title('Total Primary and Secondary School Enrollment by Gender')
plt.subplot(2,2,2)
plt.plot(df[df['indicator_name']=='Primary completion rate, total (% of relevant age group)']['Year'],df[df['indicator_name']=='Primary completion rate, total (% of relevant age group)']['Value'],color='blue')
plt.title('Primary completion rate of school')

"""As We advanced ,people saw the importance of Education and  this was instigated with the introduction of Free Primary Education by President Daniel Moi and Kibaki in 2003"""


# In[20]:


# Maternal Healthcare ,Mortality and Mortality rate
fig =plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(df[df['indicator_name']=='Mortality rate, under-5 (per 1,000 live births)']['Year'],df[df['indicator_name']=='Mortality rate, under-5 (per 1,000 live births)']['Value'],color='black')
plt.title('Mortality rate for under 5 years per 1000 live births')
plt.subplot(2,2,2)
plt.plot(df[df['indicator_name']=='Maternal mortality ratio (modeled estimate, per 100,000 live births)']['Year'],df[df['indicator_name']=='Maternal mortality ratio (modeled estimate, per 100,000 live births)']['Value'],color='red')
plt.title('Maternal Mortality ratio ')
plt.subplot(2,2,3)
plt.plot(df[df['indicator_name']=='Pregnant women receiving prenatal care (%)']['Year'],df[df['indicator_name']=='Pregnant women receiving prenatal care (%)']['Value'],color='pink')
plt.title('Pregnant women receiving prenatal care (%)')
"""Clearly as more women received maternal care ,the mortality rate reduced ,both for the mother and the children"""


# In[ ]:





# In[ ]:




