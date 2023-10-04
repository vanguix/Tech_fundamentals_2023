import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import threading

# Load the dataset
df = pd.read_csv('dummies.csv')
data = df.to_numpy()
k_values = range(1, 11)


start = time.time()

# Implement the k-means algorithm
def k_means(k, data):
    """
    Perform k-means clustering on the given data.

    Parameters:
    - k (int): The number of clusters to create.
    - data (numpy.ndarray): The input data to be clustered.

    Returns:
    - wcss (float): The within-cluster sum of squares.
    - labels (numpy.ndarray): The cluster labels for each data point.
    - centroids (numpy.ndarray): The final centroids of the clusters.
    """
    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = None
    
    while np.not_equal(centroids, prev_centroids).any():
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        prev_centroids = centroids.copy()
        centroids = np.array([data[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else prev_centroids[j] for j in range(k)])
    
    # Calculate WCSS for current K
    wcss = np.sum(np.square(data - centroids[labels]))
    
    return wcss, labels, centroids

if __name__ == "__main__":
    threads = []
    results = {}  # Dictionary to store results for each k value

    for k_value in range(1, 11):
        thread = threading.Thread(target=lambda k=k_value: results.update({k: k_means(k, data)}))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Print and use the results
    for k_value, result_tuple in results.items():
        wcss, labels, centroids = result_tuple
        print(f"Results for k={k_value}: WCSS={wcss}")

    # Extract WCSS values for the elbow graph
    wcss_values = [results[k][0] for k in k_values]


end = time.time()
print("Execution time in seconds: ", end - start)

wcss_4, labels_4, centroids_4 = results[4]

# Plot the first two dimensions of the clusters (price and speed)
plt.scatter(data[:, 1], data[:, 2], c=labels_4, cmap='viridis')
plt.scatter(centroids_4[:, 1], centroids_4[:, 2], s=300, c='red', marker='X', label='Cluster Centers')
plt.xlabel('Price')
plt.ylabel('Speed')
plt.title('Scatter plot with Clusters')
plt.legend()
plt.show()

#Find the cluster with the highest average price and print it.


#Print a heat map using the values of the clusters centroids.