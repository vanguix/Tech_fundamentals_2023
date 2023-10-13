import threading
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist


# Define the thread-safe dictionary to store results
results_lock = threading.Lock()
results = {}

# Function to initialize centroids for a given k
def initialize_centroids_thread(k_values, data):
    with results_lock:
        for k in k_values:
            np.random.seed(4)
            in_centroids = data[np.random.choice(data.shape[0], k, replace=False)]
            prev_centroids = None
            results[k] = (in_centroids, prev_centroids)

# Function to perform k-means for a given k
def k_means_thread(k_values, data):
    with results_lock:
        for k in k_values:
            centroids, prev_centroids = results[k]
            while np.not_equal(centroids, prev_centroids).any():
                distances = cdist(data[:, :-1], centroids)#error
                labels = np.argmin(distances, axis=0)
                prev_centroids = centroids.copy()
                centroids = np.array([data[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else prev_centroids[j] for j in range(k)])
            results[k] = (centroids, labels)

# Function to calculate WCSS for a given k
def calculate_wcss_thread(k_values, data):
    with results_lock:
        for k in k_values:
            centroids, labels = results[k]
            wcss = np.sum(np.square(data - centroids[labels]))
            results[k] = (wcss, labels, centroids)

def plot_elbow(k_values, wcss_values):
    '''Function to do an elbow plot
    Inputs:
        -k values (list): set of k values to be tested
        -wcss values: wcss calculation for each k clustering
    Output:
        -Elbow plot'''
    plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Graph')
    plt.grid(True)
    plt.show()

def calculate_average_price(data, labels, cluster_num): 
    '''Calculate average price for each cluster
    Inputs:
        - data (numpy.ndarray): The input data that has been clustered.
        - labels(numpy.ndarray): The cluster labels for each data point.
        - cluster_num: n cluster
    Output:
        - average price of each cluster
'''
    cluster_prices = data[labels == cluster_num][:, 1]  # Extract prices for the given cluster
    average_price = np.mean(cluster_prices)
    return average_price
 
def plot_2D(data,labels_k,centroids_k):
    '''Function to plot the first 2 dimensions of the data, colored by its clusters and with the clusters 
    centroids marked as a red X.
    Inputs:
        - data (numpy.ndarray): The input data that has been clustered.
        - labels_k: labels for the k clustering
        - centroids_k: final centroids of the k clustering
    Output:
        - Scatter plot of the two dimensions with their clusters and centroids.'''
    plt.scatter(data[:, 1], data[:, 2], c=labels_k, cmap='viridis')
    plt.scatter(centroids_k[:, 1], centroids_k[:, 2], s=300, c='red', marker='X', label='Cluster Centers')
    plt.xlabel('Price')
    plt.ylabel('Speed')
    plt.title('Scatter plot with Clusters')
    plt.legend()
    plt.show()

def plot_heatmap(centroids_k):
    '''Plot heatmap of all the features in the dataset for a given clustering
    Input: 
        - centroids_k: final centroids of that given clustering
    Output:
        -Heatmap of all the features in the dataset for a given clustering'''
    plt.figure(figsize=(10, 6))
    y_labels = ['price', 'speed', 'hd','ram', 'screen', 'cores','cd', 'laptop', 'trend']
    std_centroids_k = (centroids_k - np.mean(centroids_k, axis=0)) / np.std(centroids_k, axis = 0)
    sns.heatmap(std_centroids_k.T, annot=True,yticklabels=y_labels, fmt='.2f', cmap='YlGnBu', cbar=True)
    plt.xlabel('Clusters')
    plt.ylabel('Variables')
    plt.title('Cluster Centroids Heat Map')
    plt.show()



if __name__ == "__main__":

    df = pd.read_csv('dummies_5000.csv')
    data = df.to_numpy()
    k_values = range(1, 11)

    start = time.time()

    # Create threads for each function
    initialize_thread = threading.Thread(target=initialize_centroids_thread, args=(k_values, data))
    k_means_thread = threading.Thread(target=k_means_thread, args=(k_values, data))
    calculate_wcss_thread = threading.Thread(target=calculate_wcss_thread, args=(k_values, data))

    # Start the threads
    initialize_thread.start()
    k_means_thread.start()
    calculate_wcss_thread.start()

    # Wait for all threads to finish
    initialize_thread.join()
    k_means_thread.join()
    calculate_wcss_thread.join()

    #Execution time
    end = time.time()
    print("Execution time in seconds: ", end - start)


    # Extract WCSS values for the elbow graph
    wcss_values = [results[k][0] for k in k_values]

plot_elbow(k_values,wcss_values)
