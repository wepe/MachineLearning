"""
Created on Wed Mar 29 21:42:38 2017

@author: Robert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

ds = pd.read_csv('Mall_Customers.csv')
X = ds.iloc[:, 2:5].values

# Choosing the value of k
wcss = []

for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
    
plt.figure(1)
plt.plot(range(1,21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

k = 6

# Clustering
kmeans = KMeans(n_clusters = k)
y_kmeans = kmeans.fit(X)

labels = y_kmeans.labels_

# Making the 3D plot
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:,0], X[:,1], X[:,2], c = labels.astype(np.float))
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
           kmeans.cluster_centers_[:,2], s = 100, c = 'black', 
           label = 'Centroids', marker = 'X' )
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending score')
plt.title('Kmeans cluster plot')
plt.legend()
plt.show()


    