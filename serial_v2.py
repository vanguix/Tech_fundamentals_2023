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


for k in k_values:
    # Randomly select centroid start points, uniformly distributed across the domain of the dataset
    features_kcomputers = random.choice(data.shape[0], k, replace = False)
    centroids = data[features_kcomputers]
    # Iterate, adjusting centroids until converged or until passed max_iter
    iteration = 0
    prev_centroids = None
    while np.not_equal(centroids, prev_centroids).any() and iteration < 50:
        # Sort each datapoint, assigning to nearest centroid
        sorted_points = [[] for _ in k_values]
        for x in data:
            dists = np.sqrt(np.sum((x - centroids)**2, axis=1))
            centroid_idx = np.argmin(dists)
            sorted_points[centroid_idx].append(x)
        # Push current centroids to previous, reassign centroids as mean of the points belonging to them
        prev_centroids = centroids
        centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
        for i, centroid in enumerate(centroids):
            if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                centroids[i] = prev_centroids[i]
        iteration += 1

    #Calculate the Within-Cluster-Sum-of-Squares (WCSS) for each value of K.
    wcss = np.sum((data - centroids[centroid_idx]) ** 2) #the deviation of each data point from its assigned cluster center (**2 computationally efficient as it avoids the square operation) --> sum of squared deviations
    wcss_values.append(wcss)
     
#Plot the WCSS values against the number of clusters (K).
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Graph')
plt.grid(True)
#plt.ion()
plt.show()
#plt.pause(1)
#Identify the "elbow point" where the rate of decrease in WCSS sharply changes. This point represents the optimal number of clusters.