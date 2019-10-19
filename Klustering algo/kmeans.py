"""
Created on Wed Mar 29 21:42:38 2017

@author: Robert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.cluster import KMeans
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')


ds = pd.read_csv('Mall_Customers.csv')
X = ds.iloc[:, [3,4]].values

# Choosing the value of k by the elbow method
wcss = []

for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_transform(X)
    wcss.append(kmeans.inertia_)
    
plt.figure()
plt.plot(range(1,21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Clustering the data
k = 5
kmeans = KMeans(n_clusters = k)
y_kmeans = kmeans.fit_predict(X)

labels = [('Cluster ' + str(i+1)) for i in range(k)]

# Plotting the clusters
plt.figure()
for i in range(k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 20,
                 c = cmap(i/k), label = labels[i]) 
 
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'black', label = 'Centroids', marker = 'X')
plt.xlabel('Age')
plt.ylabel('Spending score')
plt.title('Kmeans cluster plot')
plt.legend()
plt.show()
    