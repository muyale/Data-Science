import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics

import nltk
import string
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords

books = pd.read_csv("C:\\Datasets\\books\\Books.csv")
ratings = pd.read_csv("C:\\Datasets\\books\\Ratings.csv")
users = pd.read_csv("C:\\Datasets\\books\\Users.csv")
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
# print(books.head(5))
# print(books.columns)
books['Book-Author'] = books['Book-Author'].fillna('Anonymous')
books['Publisher'] = books['Publisher'].fillna('Junior')

print(books.columns)
print(books.isnull().any())
print(users.columns)
users['Age'] = users['Age'].fillna(users['Age'].mean())
print(users.isnull().sum())
print(ratings.columns)
print(ratings.head(5))
print(ratings.isnull().sum())
book_system = pd.concat([books, ratings, users], axis=1)
book_system = book_system.head(10000)
book_system.drop(['ISBN'], axis=1, inplace=True)
print(book_system.columns)
book_author = book_system['Book-Author'].value_counts().head(5)
year_publish = book_system['Year-Of-Publication']
print(book_author.nunique())
book_system.drop(['Age'], axis=1, inplace=True)
print(book_system.isnull().sum())
book_system.dropna()
# Top 5 Authors
print(book_author)
# 10 most common publishing years
print(year_publish)
# Top Locations that read
locations = book_system['Location']
print(locations)
publishers = book_system['Publisher']
print(publishers)
important = book_system[book_system.columns]
# plt.pie(x=book_author, labels=book_author)
# plt.title('Pie chart showing top 5 authors')
# plt.show()
columns = ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',
           'User-ID', 'Book-Rating', 'User-ID', 'Location', 'Age']


def get_important_features(df):
    important_features = []
    for i in range(0, book_system.shape[0]):
        important_features.append(
            book_system['Book-Title'][i] + book_system['Book-Author'][i] + book_system['Publisher'][i] +
            book_system['Location'][i])
    return important_features


book_system['important features'] = get_important_features(book_system)
my_important_features = book_system['important features']
print(book_system.head())


def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


Count = CountVectorizer(analyzer=text_process)
bow = Count.fit_transform(book_system['important features'])

Tfidf = TfidfVectorizer()
relevance = Tfidf.fit_transform(book_system['important features'])
c_similar = cosine_similarity(relevance)

indices = pd.Series(book_system.index, index=book_system['Book-Title'])


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(c_similar[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]
    book_indices = [i[0] for i in sim_scores]

    return book_system['Book-Title'].iloc[book_indices]


def get_similar_books(title):
    input('Type in Your book')
    similar_book = get_recommendations(title)
    return similar_book


def ranking(column, df):
    top_items = []
    for i in range(0, df.head(5)):
        top_five = df[column].value_counts().head(5)
        top_items.append(top_five)

        return top_items


def new_column(df, column):
    for i in range(0, df.shape[0]):
        df['Ranked_list'] = ranking('columns', book_system)
        return df['Ranked_List']


book_system['Rankings'] = book_system.columns.apply(new_column(book_system, book_system.columns))
print(book_system.columns)


def latent():
    """This is to categorize all the books into genre"""
    # lda is an online bayes algorithm
    from sklearn.decomposition import LatentDirichletAllocation

    lda = LatentDirichletAllocation()
    x = book_system['important features']
    x = lda.fit_transform(x)
    return print(x)


print(ranking('Book-Title', book_system))