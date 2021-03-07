# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:58:05 2021

@author: SHERIF ATITEBI O
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# IMPORTING DATASET
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# TRAINING SET AND TEST SET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON TRAINING SET
regressor = LinearRegression()
regressor.fit(x_train,  y_train)

# PREDICTING THE TEST RESULT
y_pred = regressor.predict(x_test)

# VISUALISING THE TEST RESULTS
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vsExperience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# VISUALISING THE TEST SET RESULTS
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vsExperience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()