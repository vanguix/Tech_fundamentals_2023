# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:58:28 2023

@author: laram
"""

import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist


def initialize_centroids(k,data):
    '''Initialize same seed for the k centroids, so results do not change
    Inputs:
        - k (int): The number of clusters to create.
        - data (numpy.ndarray): The input data to be clustered.
    Outputs:
        - initialized centroids and previous centroids (None)'''
    np.random.seed(4)
    in_centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids=None
    return in_centroids, prev_centroids


def k_means(k,data,centroids,prev_centroids):
    '''Perform k-means clustering on the given data.

        Inputs:
        - k (int): The number of clusters to create.
        - data (numpy.ndarray): The input data to be clustered.
        - centroids: initialized centroids
        - prev_centroids: None (needed for computation)

        Outputs:
        - labels (numpy.ndarray): The cluster labels for each data point.
        - centroids (numpy.ndarray): The final centroids of the clusters.'''
    while np.not_equal(centroids, prev_centroids).any():
        # Calculate distances using cdist
        distances = cdist(data, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        
        prev_centroids = centroids.copy()
        centroids = np.array([data[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else prev_centroids[j] for j in range(k)])
    
    return centroids, labels


def calculate_wcss(data,centroids,labels):
    '''Calculates the WCSS (within-cluster sums of squares)
    Inputs:
        - data: (numpy.ndarray): The input data that has been clustered.
        - centroids: final centroids
        - labels (numpy.ndarray): The cluster labels for each data point.
    Output:
        -  Within-cluster sums of squares
    '''
    #Calcualte wcss
    wcss = np.sum(np.square(data - centroids[labels]))
    return wcss

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
    y_labels = ['trend','laptop','cd', 'cores', 'screen', 'ram', 'hd','speed','price']
    std_centroids_k = (centroids_k - np.mean(centroids_k, axis=0)) / np.std(centroids_k, axis = 0)
    sns.heatmap(std_centroids_k.T, annot=True,yticklabels=y_labels, fmt='.2f', cmap='YlGnBu', cbar=True)
    plt.xlabel('Clusters')
    plt.ylabel('Variables')
    plt.title('Cluster Centroids Heat Map')
    plt.show()


def start_k_means(k, data):
    in_centroids, prev_centroids = initialize_centroids(k, data)
    centroids, labels = k_means(k, data, in_centroids, prev_centroids)
    wcss = calculate_wcss(data, centroids, labels)
    
    return k, wcss
    
if __name__ == "__main__":
    
    start_total=time.time()
    
    df = pd.read_csv('computers_500000.csv', usecols=lambda column: column != 'id')
    for col in ['cd', 'laptop']:
        df[col].replace(['no', 'yes'], [0, 1], inplace=True)
    data = df.to_numpy()
    k_values = range(1, 11)
    wcss_values = []
    
    pool = mp.Pool(mp.cpu_count()) 
    start_par = time.time()
    
    #1.- Construct the data for the elbow graph and find the optimal clusters number (k).
    for k in k_values:
        #2.- Implement the k-means algorithm
        results = pool.apply(start_k_means,args=(k,data))
        wcss_values.append(results)
            
    pool.close()
    pool.join()

    #3.-Measure time for the parallelized part 
    end_par = time.time()
    print("Execution time for parallelized part in seconds: ", end_par - start_par)

    #4.-Cluster the data using the optimum value using k_means --> k=3
    chosen_k = 3
    centroids_k, prev_centroids_k = initialize_centroids(chosen_k, data)
    centroids_k, labels_k = k_means(chosen_k, data, centroids_k, prev_centroids_k)

    #5.-Find the cluster with the highest average price and print it.
    average_prices_per_cluster = [calculate_average_price(data, labels_k, cluster_num) for cluster_num in range(chosen_k)]
    highest_avg_price_cluster = np.argmax(average_prices_per_cluster)
    highest_avg_price = average_prices_per_cluster[highest_avg_price_cluster]
    print("Cluster with the highest average price is cluster", highest_avg_price_cluster)
    print("Average price of the cluster:", highest_avg_price)

    #6.-Plot the results of the elbow graph.
    # Sort wcss_values by k before plotting (threads may give the result unordered)
    wcss_values.sort(key=lambda x: x[0])
    k_values, sorted_wcss_values = zip(*wcss_values)
    
    #7. Measure time of the total program (without the plots)
    end_total=time.time()
    print("Total execution time: ", end_total - start_total)
    
    #Plot in the elbow graph the sorted values
    plot_elbow(k_values, sorted_wcss_values)

    #8.- Plot the first two dimensions of the clusters (price and speed)
    plot_2D(data,labels_k,centroids_k)


    #9.- Print a heat map using the values of the clusters centroids
    plot_heatmap(centroids_k)
