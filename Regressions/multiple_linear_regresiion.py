# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:42:45 2021

@author: SHERIF ATITEBI O
"""
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# encoding categorical data
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(ct.fit_transform(x))


# training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# training multiple linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)    # set decimal to 2 places

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

# making a single prediction for the stste of california
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# getting coefficients and values
print(regressor.coef_)
print(regressor.intercept_)