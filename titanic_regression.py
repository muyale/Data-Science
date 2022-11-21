import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv("C:\\Datasets\\titanic_train.csv")
'''print(titanic.head())

print(titanic.info())
print(titanic.isnull().value_counts())
print(titanic.describe())'''
'''sns.countplot(x='Survived', hue='Pclass', data=titanic)
plt.title('Survival Rate by Class')
plt.xlabel('Survival Rates')
plt.show()'''
print(titanic.columns)
# sns.pairplot(titanic)
# plt.title('Titanic Pairplot')
'''''titanic_corr = titanic.corr()
print(titanic_corr)
sns.heatmap(titanic_corr, cmap='Blues')
plt.title('Titanic Data Correlation')
plt.show()'''
# print(titanic.head(5))
titanic.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
# print(titanic.columns)
# print(titanic.head(5))
print(titanic['Age'].isnull().value_counts())


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        if Pclass == 2:
            return 28
        if Pclass == 3:
            return 19
    else:
        return Age


titanic['Age'] = titanic[['Age', 'Pclass']].apply(impute_age, axis=1)
# print(titanic.columns)
embark = pd.get_dummies(titanic['Embarked'], drop_first=True)
sex = pd.get_dummies(titanic['Sex'], drop_first=True)
titanic.drop(['Sex', 'Embarked', 'Name'], axis=1, inplace=True)
titanic = pd.concat([titanic, embark, sex], axis=1)
print(titanic.columns)
X = titanic.drop(['Survived'], axis=1)
y = titanic['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_prediction = log_model.predict(X_test)
# print(log_prediction)
# sns.countplot(log_prediction)
# plt.title('Predicted Survival ')
# plt.show()
from sklearn import metrics

print('MSE ;', metrics.mean_squared_error(y_test, log_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, log_prediction)))
print('MEAN ABSOLUTE:', metrics.mean_absolute_error(y_test, log_prediction))
