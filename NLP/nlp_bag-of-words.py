# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 07:36:37 2022

@author: SHERIF ATITEBI O
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
#%%

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

#%%
#Clean the texts

corpus = []
for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    
    # Simplifying each word by the root of the words (loved --> love, bought --> buy)
    
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    
    review = " ".join(review)
    corpus.append(review)

#%%
#Bag of Words model

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#%%
# Split into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#%%

#Naive Bayes model
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predict test set
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#%%
# Predicting for a single review
new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)

