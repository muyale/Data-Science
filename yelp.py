import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

yelp = pd.read_csv("C:\\Datasets\\yelp.csv")
# print(yelp.head(4))
# print(yelp.describe())
# print(yelp['text'].head(5))
# print(yelp.info())
# print(yelp.columns)

yelp['text_length'] = yelp['text'].apply(len)
# print(yelp['text_length'].head(5))
# print(yelp.head(6))
# print(yelp['text_length'].max())

"""EDA"""
# sns.boxplot(x=yelp['stars'], y =yelp['text_length'])
# plt.title('A boxplot of text length and star category')
# plt.show()
# sns.countplot(yelp['stars'])
# plt.title('How People Rated The Business')
# plt.show()
correlation = yelp.corr()
# print(correlation)
# sns.heatmap(correlation,cmap='Blues')
# plt.title('Yelp heat map')
# plt.show()

"""NLP Classification Task"""
# a data frame containing five or one star

yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]
print(yelp_class)
X = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer

my_vectorizer = CountVectorizer()
X = my_vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB

bayes_classifier = MultinomialNB()
bayes_classifier.fit(X_train, y_train)
my_prediction = bayes_classifier.predict(X_test)

from sklearn import metrics

print(metrics.classification_report(y_test, my_prediction))

"""Using Text Processing"""
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

from sklearn.pipeline import Pipeline

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline = Pipeline(
    [('bow', CountVectorizer(analyzer='word')), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])

pipeline.fit(X_train, y_train)
pipe_prediction = pipeline.predict(X_test)
print(metrics.classification_report(y_test, pipe_prediction))

