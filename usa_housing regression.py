import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# linear regression
housing_df = pd.read_csv("C:\\Datasets\\USA_Housing.csv")
# print(housing_df.shape)
# print(housing_df.head())
# print(housing_df.info())
# print(housing_df.columns)
# print(housing_df.isnull())
bedrooms = housing_df['Avg. Area Number of Bedrooms']
prices = housing_df['Price']
# sns.displot(x=prices)
plt.title('How Prices range with number of Bedrooms')
my_corr = housing_df.corr()
# sns.heatmap(my_corr, cmap='PuBu')
# plt.title('Heatmap ofHow Prices range ')
# plt.show()
# print(housing_df['Address'].value_counts().head(5)
housing_df.drop(['Address'], axis=1, inplace=True)
print(housing_df.columns)
# print(housing_df['Avg. Area Income'].mean())
# plt.scatter(x=bedrooms,y=prices,ls='--')
# plt.title('Scatter Plot ')
# plt.show()
# print(housing_df.corr())
# sns.heatmap(housing_df.corr())
# plt.show()

X = housing_df.drop(['Price'], axis=1)
y = housing_df['Price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
my_prediction = model.predict(X_test)
# print(my_prediction)
# plt.scatter(y_test,my_prediction)
# plt.xlabel('Y test')
# plt.ylabel('My Predict')
# plt.title('A prediction scatter plot')
# plt.show()
sns.displot(my_prediction, rug_kws=True)
plt.title('A Plot to check if residuals are normally distributed')
plt.show()
from sklearn import metrics

print('MSE:', metrics.mean_squared_error(y_test, my_prediction))
print('MAE:', metrics.mean_absolute_error(y_test, my_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, my_prediction)))
