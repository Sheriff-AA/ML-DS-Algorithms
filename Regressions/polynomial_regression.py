# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:41:12 2021

@author: SHERIF ATITEBI O
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# training the linear regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# training the polynomial regression model
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# visualising linear regression results
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Linear Reg. model on non-linear")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualising polynomial linear regression results
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg_2.predict(x_poly), color="blue")
plt.title("Polynomial Linear Regression.")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# prdicting new result with linear regression
print(lin_reg.predict([[6.5]]))

# predicting with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
