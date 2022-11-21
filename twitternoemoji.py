import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

twitter = pd.read_csv("C:\\Datasets\\twitter_data.csv", sep='/t')
twitter = twitter.head(50)
print(twitter.head(8))
"""
print('Hello how can we help')
responses = input('Respond').lower()
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
sia_score = sia.polarity_scores(responses)
for responses in responses:
    if sia_score['neg'] > sia_score['pos']:
        print('This pretty bad,how can we help')
        answer = input('How Can we help,1 -talk,2-Help,3-Family')
        for answer in answer:

            if answer == 1:
                print('Sure We can,how about start with whats bothering')
                bothering = input('Whats bothering you,career or personal?')
            elif answer == 2:
                print('Try the following websites')
            elif answer == 3:
                print('Who can we inform about this ')
            else:
                print('Thanks for Reaching out')
    elif sia_score['neg'] < sia_score['pos']:
        print('Tell us the highlight of your day')
        answer = input('Highlight')
    else:
        print('Thanks for Reaching out to Mental Stability ')"""
