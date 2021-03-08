# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:29:30 2021

@author: SHERIF ATITEBI O
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# import dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# training the decision tree model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# predict new result
print(regressor.predict([[6.5]]))

# visualising the result
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("Decision Tree Regression.")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
