#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score
from sklearn.metrics import mean_poisson_deviance,mean_gamma_deviance,accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[3]:


# Read in our data
bitcoin_df = pd.read_csv("C:\\Users\EDGAR MUYALE DAVIES\\Downloads\\BTC-USD.csv")
bitcoin_df.head()
bitcoin_df = bitcoin_df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
bitcoin_df.head()


# In[4]:


binance_df = pd.read_csv("C:\\Users\EDGAR MUYALE DAVIES\\Downloads\\BNB-USD.csv")
binance_df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'},inplace=True)
binance_df.head()


# In[5]:


cardano_df = pd.read_csv("C:\\Users\EDGAR MUYALE DAVIES\\Downloads\\BNB-USD.csv")
cardano_df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'},inplace=True)
cardano_df


# In[6]:


dataframes = [binance_df,bitcoin_df,cardano_df]
def important_insight():
   
    dataframes = [binance_df,bitcoin_df,cardano_df]
    for dataframe in dataframes:
        print (f"Important_information for{dataframe}|Which is{dataframe.info()}")
        print(dataframe.describe())
        print(dataframe.isnull().sum())


# In[7]:


important_insight()


# In[8]:


# Preprocessing by filling NA values using forward fill
dataframes
def fill_all_null():
    for dataframe in dataframes:
        dataframe = dataframe.fillna(method='ffill')
        print(dataframe.isnull().sum())


# In[9]:


fill_all_null()


# In[10]:


binance_df['date'] = pd.to_datetime(binance_df.date)
binance_df.head().style.set_properties(subset=['date','close'], **{'background-color': 'pink'})
bitcoin_df['date'] = pd.to_datetime(bitcoin_df.date)
bitcoin_df.head().style.set_properties(subset=['date','close'], **{'background-color': 'skyblue'})
cardano_df['date']= pd.to_datetime(cardano_df.date)
cardano_df.head().style.set_properties(subset=['date','close'], **{'background-color': 'yellow'})
bitcoin_df.head()
bitcoin_df['date'] = pd.to_datetime(bitcoin_df.date)
bitcoin_df.head()


# In[11]:



bitcoin_df.head()


# In[12]:


cardano_df.head()


# In[13]:


sns.lineplot(data=bitcoin_df,x=bitcoin_df.date,y=bitcoin_df.adj_close)
plt.title('Bitcoin Changing overtime')


# In[14]:


# Creating Subplots using plt.subplots
fig = plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(bitcoin_df['date'],bitcoin_df['adj_close'],color='red')
plt.title('Bitcoin close prices')
plt.subplot(2,2,2)
plt.plot(binance_df['date'],binance_df['adj_close'],color='green')
plt.title('Binance close prices')
plt.subplot(2,2,3)
plt.plot(cardano_df['date'],cardano_df['adj_close'],color='blue')
plt.title('Cardano close prices')


# In[15]:


last_year_bitcoin_df = bitcoin_df[bitcoin_df['date']>'09-2020'] 
last_year_binance_df = binance_df[cardano_df['date']>'09-2020'] 
last_year_cardano_df = cardano_df[cardano_df['date']>'09-2020'] 



# In[16]:


fig=plt.figure(figsize=(15,12))
fig.suptitle('2021 close prices of Bitcoin Cardano and Binance')
plt.subplot(4,1,1)
plt.plot(last_year_bitcoin_df['date'],last_year_bitcoin_df['adj_close'],color='black')
plt.legend('BT')
plt.subplot(4,1,2)
plt.plot(last_year_binance_df['date'],last_year_binance_df['adj_close'],color='blue')
plt.legend('N')
plt.subplot(4,1,3)
plt.plot(last_year_cardano_df['date'],last_year_cardano_df['adj_close'],color='yellow')
plt.legend('C')
# From the figure all the three financial assets seem to have increased for the one year period


# In[17]:


# Since THE STOCK PRICES ARE HIGHLY VOLATILE WE WILL COMPUTE THEIR MOVING AVERAGES
fig = plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(bitcoin_df['date'],bitcoin_df['adj_close'].rolling(50).mean())
plt.plot(bitcoin_df['date'],bitcoin_df['adj_close'].rolling(200).mean())
plt.title('Bitcoin Prices moving Average')
plt.subplot(2,2,2)
plt.plot(binance_df['date'],binance_df['adj_close'].rolling(50).mean(),color='red')
plt.plot(binance_df['date'],binance_df['adj_close'].rolling(200).mean(),color='blue')
plt.title('Binance Prices Moving Averages')
plt.subplot(2,2,3)
plt.plot(cardano_df['date'],cardano_df['adj_close'].rolling(50).mean(),color='green')
plt.plot(cardano_df['date'],cardano_df['adj_close'].rolling(200).mean(),color='black')
plt.title('Cardano Prices moving averages')


# In[18]:


# Creating a dataframe for close_price and date
close_df = bitcoin_df[['date','close']]
close_df = close_df[close_df['date'] > '2020-09-13']
close_df.shape
close_stock = close_df.copy()
close_df


# In[19]:



del close_df['date']
scaler=MinMaxScaler(feature_range=(0,1))
close_df=scaler.fit_transform(np.array(close_df).reshape(-1,1))
print(close_df.shape)


# In[20]:


X=close_stock['date']
y = close_stock['close']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.7,random_state =101)


# In[21]:


# Visualizing our test and train data
fig = plt.figure(figsize=(15,10))
plt.subplot(3,1,1)
sns.lineplot(x=X_train,y=y_train,color ='black')
plt.title('Train Data')
plt.legend('TrainData')
plt.subplot(3,1,2)
sns.lineplot(x=X_test,y=y_test)
plt.title('Test Data')


# In[22]:


# Using Random forest to predict future bitcoin prices
bitcoin_df.head()


# In[29]:


sns.lineplot(y=bitcoin_df.volume,x=bitcoin_df.date)
plt.title('Volume Changes overtime')


# In[33]:


sns.lineplot(y=bitcoin_df.volume,x=bitcoin_df.close)
plt.title('Volume Changes overtime with respect to close price')


# In[62]:



important_features =['open', 'high', 'low','adj_close']
x = bitcoin_df.drop(['close'],axis=1).astype('float64')
y = bitcoin_df['close'].astype('float64')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state =101)
train_data = (x_train,y_train)
test_data = (x_test,y_test)


# In[63]:


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)


# In[65]:


""" from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor()
rfc.fit(x,y)
predictions = rfc.predict(x_test)"""


# In[ ]:





# In[ ]:




