# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:38:25 2021

@author: SHERIF ATITEBI O
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler   # importing the feature scaler
from sklearn.svm import SVR   # import svr class

# importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values
y = y.reshape(len(y), 1)    # because the standard scaler object does not accept arrays


# Feature scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# training the SVR model on the whole dataset
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

# predicting a new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# visualising the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color="blue")
plt.title("Support Vector Regression.")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualising the SVR results (higher resolution)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color="blue")
plt.title("Support Vector Regression.")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
