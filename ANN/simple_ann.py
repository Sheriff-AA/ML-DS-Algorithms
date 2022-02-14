# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:08:16 2022

@author: SHERIF ATITEBI O
"""

import numpy as np
import pandas as pd
# from tensorflow.keras.model import Sequential
# from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics. import confusion_matrix, accuracy_score
#%%
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#%%
# Encoding categorical data
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])  

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder = "passthrough")
x = np.array(ct.fit_transform(x))
#%%
# Training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#%%
# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
#%%

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=7, activation="relu"))
ann.add(tf.keras.layers.Dense(units=7, activation="relu"))

# if non-binary activation should be "softmax"
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

#%%
# Train the network on the dataset

# For binay classification loss function should be "binary_crossentropy"
# else "categorical_crossentropy"
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

ann.fit(x_train, y_train, batch_size=32, epochs=100)

#%%

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))> 0.5)

#%%
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
#%%

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))










