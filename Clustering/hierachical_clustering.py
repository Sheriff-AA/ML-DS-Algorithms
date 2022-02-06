# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 11:01:11 2022

@author: SHERIF ATITEBI O
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
#%%

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

#Using the dendogram to find the optimum number of clusters
dendogram = sch.dendrogram(sch.linkage(x, method="ward"))

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()
#%%
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(x)
print(y_hc)

#%%
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = "red", label="Cluster 1")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = "blue", label="Cluster 2")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = "green", label="Cluster 3")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = "cyan", label="Cluster 4")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = "magenta", label="Cluster 5")
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spendong Score (1 - 100)")
plt.legend()
plt.show()
