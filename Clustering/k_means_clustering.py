# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 10:06:58 2022

@author: SHERIF ATITEBI O
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
#%%

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

#Using the elbow method to find the optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("WCSS Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
#%%
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = kmeans.fit_predict(x)
print(y_pred)

#%%
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = "red", label="Cluster 1")
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = "blue", label="Cluster 2")
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = "green", label="Cluster 3")
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = "cyan", label="Cluster 4")
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = "magenta", label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c = "yellow", label = "Centroids")
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spendong Score (1 - 100)")
plt.legend()
plt.show()
