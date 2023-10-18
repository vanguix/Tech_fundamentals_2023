# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:20:10 2023

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns


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
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)
        # Update centroids
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

start_total = time.time()

# Load the dataset
df = pd.read_csv('computers.csv', usecols=lambda column: column != 'id')
for col in ['cd', 'laptop']:
    df[col].replace(['no', 'yes'], [0, 1], inplace=True)
data = df.to_numpy()
k_values = range(1, 11)
wcss_values = []  

start_par = time.time()

#1.- Construct the data for the elbow graph and find the optimal clusters number (k).
for k in k_values:
    #2.- Implement the k-means algorithm
    in_centroids, prev_centroids = initialize_centroids(k,data)
    centroids, labels = k_means(k,data,in_centroids,prev_centroids)
    wcss = calculate_wcss(data,centroids,labels)
    wcss_values.append(wcss)

end_par = time.time()

#3.-Cluster the data using the optimum value using k_means --> k=3
chosen_k = 3
centroids_k, prev_centroids_k = initialize_centroids(chosen_k,data)
centroids_k, labels_k = k_means(chosen_k,data,centroids_k,prev_centroids_k)

#4.-Find the cluster with the highest average price and print it.
average_prices_per_cluster = [calculate_average_price(data, labels_k, cluster_num) for cluster_num in range(chosen_k)]
highest_avg_price_cluster = np.argmax(average_prices_per_cluster)
highest_avg_price = average_prices_per_cluster[highest_avg_price_cluster]
print("Cluster with the highest average price is cluster", highest_avg_price_cluster)
print("Average price of the cluster:", highest_avg_price)

#5.-Measure total execution time and time of the part that will be parallelized
end_total = time.time()
print("Total execution time in seconds: ", end_total - start_total)
print("Execution time for the part that will be parallelized: ", end_par - start_par)

#6.-Plot the results of the elbow graph.
plot_elbow(k_values,wcss_values)

#7.- Plot the first two dimensions of the clusters (price and speed)
plot_2D(data,labels_k,centroids_k)

#8.- Print a heat map using the values of the clusters centroids
plot_heatmap(centroids_k)
