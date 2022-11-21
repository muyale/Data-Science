import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.utils
from sklearn.datasets import load_iris

sklearn.utils.Bunch

iris = load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df.columns = iris.feature_names
iris_df['Class'] = iris.target
print(iris_df.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(iris_df.drop(['Class'], axis=1))
scaled_features = scaler.transform(iris_df.drop(['Class'], axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=iris_df.columns[:-1])
print(scaled_df.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, iris_df['Class'], test_size=0.3, random_state=107)

from sklearn.neighbors import KNeighborsClassifier

model_classifiers = KNeighborsClassifier(n_neighbors=43)
model_classifiers.fit(X_train, y_train)
pred = model_classifiers.predict(X_test)

from sklearn import metrics

print(metrics.confusion_matrix(y_test, pred))
print(metrics.classification_report(y_test, pred))

# choosing the best K value
error_rate = []
for i in range(40, 60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(40, 60), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
