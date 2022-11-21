import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics

books = pd.read_csv("C:\\Datasets\\books\\Books.csv")
ratings = pd.read_csv("C:\\Datasets\\books\\Ratings.csv")
users = pd.read_csv("C:\\Datasets\\books\\Users.csv")
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
print(books.head(5))
print(books.columns)
books['Book-Author'] = books['Book-Author'].fillna('Anonymous')
print(books.columns)
print(books.isnull().any())
book_author = books['Book-Author'].value_counts().sort_values(ascending=True)
year_publish = books['Year-Of-Publication'].value_counts().sort_values(ascending=True)
books.drop(['ISBN'], axis=1, inplace=True)
""" print(books.columns)
# print(ratings.columns)
print(ratings.head(5))
# print(ratings.isnull.any())
print(users.columns)
print(users.head(5))
print(ratings.isnull().any())"""
