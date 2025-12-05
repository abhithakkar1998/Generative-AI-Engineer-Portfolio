# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 14:37:59 2025

@author: kthakkara
"""

#Import data
import pandas as pd

df = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])

df.head()


#Create word corpus
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-z]', ' ',df['message'][i])
    review = review.lower()
    review = review.split()
    review = [ ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
    

print(corpus)


# Create BoW
from sklearn.feature_extraction.text import CountVectorizer
countVectorizer = CountVectorizer(max_features=100, binary=True) #take top 2500 words which have top frequency
X = countVectorizer.fit_transform(corpus).toarray()
X

X.shape

