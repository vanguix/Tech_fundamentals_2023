import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

start = time.time()

# Load the dataset
df = pd.read_csv('dummies.csv')
data = df.to_numpy()

k_values = range(1, 11)
wcss_values = []  # WCSS values for different number of clusters (K)

for k in k_values:
    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = None
    
    while np.not_equal(centroids, prev_centroids).any() :
        # Assign each point to the nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        prev_centroids = centroids.copy()
        centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        
        # Handle empty clusters
        for j in range(k):
            if np.isnan(centroids[j]).any():
                centroids[j] = prev_centroids[j]
        
    
    # Calculate WCSS for current K
    wcss = np.sum((data - centroids[labels]) ** 2)
    wcss_values.append(wcss)


end = time.time()
print(end - start)
# Plot the elbow graph
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Graph')
plt.grid(True)
plt.show()
