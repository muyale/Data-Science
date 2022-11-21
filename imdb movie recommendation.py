import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

imdb = pd.read_csv("C:\\Datasets\\IMDB Dataset.csv")
print(imdb.head(4))
imdb['length'] = imdb['review'].apply(len)
print(imdb['length'].head(5))
print(imdb['length'].sort_values(ascending=False))
print(imdb['sentiment'].value_counts())
lanc = LancasterStemmer()
snow = SnowballStemmer('english')
porter = PorterStemmer()


def tokenize(text):
    return text.split()


def all_stems(text):
    stems = [[lanc(text), snow(text), porter(text)]]
    return stems


def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


Count = CountVectorizer(analyzer=text_process)
bow = Count.fit_transform(imdb['review'])
print(Count.vocabulary)
Tf = TfidfVectorizer.fit_transform(imdb['review'])
tf_vect = Tf[:30]
print(tf_vect)
