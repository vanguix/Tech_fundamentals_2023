import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from time import time
import multiprocessing as mp
import seaborn as sns


t0 = time()
   
# Load the dataset
df = pd.read_csv('dummies.csv')
data = df.to_numpy()
k_values = range(1, 11)


#Implement the k-means algorithm
def k_means(k,data):
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
        
        while np.not_equal(centroids, prev_centroids).any() :
            # Assign each point to the nearest centroid
            distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            prev_centroids = centroids.copy()
            centroids = np.array([data[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else prev_centroids[j] for j in range(k)])
        
        # Calculate WCSS for current K
        wcss = np.sum(np.square(data - centroids[labels]))
        #print(k, wcss_values, '\n')
        return wcss,labels, centroids
    
    
if __name__ == "__main__":

   pool = mp.Pool(mp.cpu_count())
   #print(mp.cpu_count()) #maximum number of processors in the computer (4)
   
   wcss_values = [] #results
   for k in k_values:
       wcss,labels,centroids=pool.apply(k_means,args=(k,data))
       wcss_values.append(wcss)

   pool.close()
   print(wcss_values)

   t1 = time()
   print("Time: ", str(t1 - t0))
   
   # Construct and plot the elbow graph and find the optimal clusters number (k).
   plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
   plt.xlabel('Number of Clusters (K)')
   plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
   plt.title('Elbow Graph')
   plt.grid(True)
   plt.show()
   
   #Cluster the data using the optimum value using k_means --> k=4
   wcss_4,labels_4, centroids_4= k_means(4,data)


   # Plot the first two dimensions of the clusters (price and speed)
   plt.scatter(data[:, 1], data[:, 2], c=labels_4, cmap='viridis')
   plt.scatter(centroids_4[:, 1], centroids_4[:, 2], s=300, c='red', marker='X', label='Cluster Centers')
   plt.xlabel('Price')
   plt.ylabel('Speed')
   plt.title('Scatter plot with Clusters')
   plt.legend()
   plt.show()

   #Find the cluster with the highest average price and print it.
   # Calculate average price for each cluster
   def calculate_average_price(data, labels, cluster_num):
       cluster_prices = data[labels == cluster_num][:, 1]  # Extract prices for the given cluster
       average_price = np.mean(cluster_prices)
       return average_price

   # Calculate average prices for all clusters
   average_prices_per_cluster = [calculate_average_price(data, labels_4, cluster_num) for cluster_num in range(4)]
   highest_avg_price_cluster = np.argmax(average_prices_per_cluster)
   highest_avg_price = average_prices_per_cluster[highest_avg_price_cluster]

   # Print the cluster with the highest average price
   print("Cluster with the highest average price is cluster", highest_avg_price_cluster)
   print("Average price of the cluster:", highest_avg_price)


   #Print a heat map using the values of the clusters centroids
   plt.figure(figsize=(10, 6))
   sns.heatmap(np.array([centroids_4[:, 1], centroids_4[:, 2]]), annot=True, fmt='.2f', cmap='YlGnBu', cbar=True)
   plt.xlabel('Price')
   plt.ylabel('Speed')
   plt.title('Cluster Centroids Heat Map')
   plt.show()


   
   