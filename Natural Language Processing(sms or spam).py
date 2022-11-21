import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string

messages = pd.read_csv("C:\\Datasets\\smsspamcollection\\SMSSpamCollection", sep='\t',
                       names=["label", "message"])
print(messages.head())
"""Exploratory Data Analysis"""
print(messages.describe())
print(messages.groupby('label').describe())
# a column to show how long the messages are
messages['length'] = messages['message'].apply(len)
print(messages.head())
# sns.countplot(messages['label'])
# plt.title('How the  Messages Differ HAM vs SPAM ')
# plt.show()
# print(messages[messages['length'] == 910]['message'].iloc[0])
# print(messages['length'].min())
# sns.histplot(data=messages, x='label', y='length', kde_kws=True)
# plt.show()

"""Text Preprocessing"""
mess = 'This is a message: Do Not Despair/Give Up'
no_punc = [char for char in mess if mess not in string.punctuation]
no_punc = ''.join(no_punc)

# Removing StopWords
from nltk.corpus import stopwords

# print(stopwords.words('english')[1:20])
print(no_punc.split())
clean_message = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
print(clean_message)


def text_edit(mess):
    no_punc = [char for char in mess if mess not in string.punctuation]
    no_punc = ''.join(no_punc)

    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


print(messages['message'].head(4).apply(text_edit))
# Tokenization
from sklearn.feature_extraction.text import CountVectorizer

Vector = CountVectorizer(analyzer=text_edit)
transform = Vector.fit_transform(messages['message'])
# print(len(Vector.vocabulary_))
bow_transform = Vector.transform(messages['message'])
# print(bow_transform)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()
tfidf_transformed = tfidf.fit_transform(bow_transform)
# print(tfidf_transformed)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB()
spam_detect_model.fit(tfidf_transformed, messages['label'])
all_predictions = spam_detect_model.predict(tfidf_transformed)
# print(all_predictions)
# sns.countplot(all_predictions)
# plt.title('predicted Ham vs SPAM')
# plt.show()

"""MODEL EVALUATION"""
from sklearn import metrics

print('This is My Model Performance :', metrics.classification_report(messages['label'], all_predictions))
"""USING TRAIN TEST SPLIT"""
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

"""USING PIPELINE"""
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_edit)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(messages['message'], messages['label'])
pipe_prediction = pipeline.predict(messages['label'])
print('pipeline prediction:', metrics.classification_report(messages['message'], pipe_prediction))
