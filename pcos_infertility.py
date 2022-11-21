import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics

p_cos = pd.read_csv("C:\\Datasets\\archive\\PCOS_infertility.csv")

p_cos.drop(['Sl. No', 'Patient File No.'], axis=1, inplace=True)
print(p_cos.columns)
print(p_cos.head(3))
# sns.countplot(x='PCOS (Y/N)', data=p_cos)
# plt.title('P COS count plot')
# plt.show()
p_corr = p_cos.corr()
# sns.violinplot(x='PCOS (Y/N)', y='  I   beta-HCG(mIU/mL)', data=p_cos)
# plt.title('A violin plot for PCOS And Beta II')
# plt.show()
# sns.heatmap(p_corr, cmap='BuPu')
# plt.title('A heatmap of Factors leading to P COS diagnosis')
# plt.show()
X = p_cos.drop(['PCOS (Y/N)'], axis=1)
y = p_cos['PCOS (Y/N)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=10)
'''lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_prediction = lin_model.predict(X_test)
sns.countplot(lin_prediction, palette='deep')
sns.set_style('ticks')
plt.title('PREDICTED PCOS DIAGNOSIS')
plt.show()'''
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_prediction = log_model.predict(X_test)
sns.countplot(log_prediction)
plt.title('Logistic PREDICTED PCOS')
plt.show()
# Regression Evaluation Metrics
print('MAE LINEAR:', metrics.mean_absolute_error(y_test, lin_model))
print('MSE LINEAR:', metrics.mean_squared_error(y_test, lin_model))
print(np.sqrt('RMSE LINEAR:', metrics.mean_squared_error(y_test, lin_model)))
