#1) Construct the elbow graph and find the optimal clusters number (k)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from numpy import random

df = pd.read_csv('dummies.csv')
data = df.to_numpy()
k_values = range(1, 11)
wcss_values = [] #sum of distances of the points of a cluster to the centroid

#Perform K-means clustering for different values of K.
for i in k_values:
    features_kcomputers = random.choice(data.shape[0], i, replace = False)
    centroids = data[features_kcomputers]
    distances =  numpy.linalg.norm(data[:, numpy.newaxis, :] - centroids, axis=2) #calculates the Euclidean distance between each data point and each cluster centroid along the third axis (cluster centroids)
    index_closestcentroid = numpy.argmin(distances, axis=1) #indices of the minimum value among each row of the distances array
    
    #Calculate the Within-Cluster-Sum-of-Squares (WCSS) for each value of K.
    wcss = np.sum((data - centroids[index_closestcentroid]) ** 2) #the deviation of each data point from its assigned cluster center (**2 computationally efficient as it avoids the square operation) --> sum of squared deviations
    wcss_values.append(wcss)


#Plot the WCSS values against the number of clusters (K).
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Graph')
plt.grid(True)
plt.ion()
plt.show()
plt.pause(1)
#Identify the "elbow point" where the rate of decrease in WCSS sharply changes. This point represents the optimal number of clusters.