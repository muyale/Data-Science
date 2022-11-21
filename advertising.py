import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

advert = pd.read_csv("C:\\Datasets\\advertising.csv")
'''print(advert.columns)
print(advert.isnull().value_counts())
print(advert.describe())
print(advert.info())
print(advert.shape)'''
# EDA
columns = ['Daily Time Spent on Site', 'Age', 'Area Income',
           'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country',
           'Timestamp', 'Clicked on Ad']
# sns.jointplot(data=advert, x='Daily Time Spent on Site', y='Age')
# plt.title('Time Spent on Title based off age relationship')
# plt.show()
advert_correlation = advert.corr()
# sns.heatmap(advert_correlation)
# plt.title('Different Correlations in the Adverts')
# plt.show()
# sns.histplot(x='Age', data=advert)
# plt.title('How Customers clicked on Ad based of their age')
# plt.show()
# sns.jointplot(x='Area Income', y='Age', data=advert, bins=50, kind='hist',palette='coolwarm')
# plt.title('A joint plot of Area income vs Age')
# plt.show()
# sns.jointplot(x='Daily Time Spent on Site', y='Age', data=advert, bins=50, kind='kde',palette='coolwarm')
# plt.title('KDE plot of daily time spent on site VS age')
# plt.show()
# sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=advert, kind='scatter')
# plt.title('Daily Internet Usage vs Time Spent on Site')
# plt.show()
# sns.pairplot(advert, hue='Clicked on Ad')
# plt.title('My Pairplot hued by clicked on AD')
# plt.show()
X = advert[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = advert['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
my_model = LogisticRegression()
my_model.fit(X_train, y_train)
my_prediction = my_model.predict(X_test)
print(metrics.classification_report(y_test, my_prediction))
print('MSE:', metrics.mean_squared_error(y_test, my_prediction))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, my_prediction)))
print('MAE:', metrics.mean_absolute_error(y_test, my_prediction))
sns.countplot(my_prediction)
plt.title('My Predicted clicks on Adverts')
plt.show()
