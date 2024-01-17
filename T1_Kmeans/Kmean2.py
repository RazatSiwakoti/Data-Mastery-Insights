#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Razat Siwakoti (A00046635)
#DMV302 - Assessment 2 
#Kmeans2.ipynb created on Jupyter notebook


# In[2]:


#source code: CihanBosnali (2019)
#https://github.com/CihanBosnali/Machine-Learning-without-Libraries/blob/master/K-Means-Clustering/K-Means-Clustering-without-ML-libraries.ipynb

#Prasanth S N (2020)
#https://ai538393399.wordpress.com/2020/09/29/k-means-clustering-algorithm-without-libraries/


# In[3]:


# Importing necessary libraries
import os
os.getcwd()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Setting the plotting style to 'dark_background'
plt.style.use('dark_background')


# In[4]:


# Reading the CSV file "HouseholdWealth.csv" into a pandas DataFrame
df = pd.read_csv("HouseholdWealth.csv")
# Displaying the first few rows of the DataFrame to get an overview
df.head()


# In[5]:


class KMeansClustering:
    def __init__(self, X, num_clusters):
        # Initialization of parameters
        self.K = num_clusters # Number of clusters
        self.max_iterations = 100 # Maximum number of iterations to avoid running infinitely
        self.num_examples, self.num_features = X.shape # num of examples, num of features
        self.plot_figure = True # Whether to plot figures during the iterations
        self.centroids = self.initialize_random_centroids(X)

    # randomly initialize centroids
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features)) # initialize centroids with zero
        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))] # random centroids
            centroids[k] = centroid
        return centroids # return randomly initialized centroids


    # create cluster Function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)] #initialize clusters
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
            np.sqrt(np.sum((point-centroids)**2,axis=1))
            )
            # Find the closest centroid using Euclidean distance(calculate distance of every point from centroid)
            clusters[closest_centroid].append(point_idx)
        return clusters
    #Calclulate new centroids
    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0) # Calculate the mean for new centroids
            centroids[idx] = new_centroid
        return centroids

    # prediction
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples) # Initialize new centroids with zeros
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred

    # plotinng scatter plot
    def plot_fig(self, X, y):
            fig = px.scatter(X[:, 0], X[:, 1], color=y)
            fig.show() # Visualize the scatter plot
    #fit data
    def fit(self,X):
        centroids = self.initialize_random_centroids(X) # initialize random centroids
        for iteration in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids) # create cluster
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X) # Calculate new centroids based on current clusters
            diff = centroids - previous_centroids # calculate difference
            if not diff.any():
                break   # Break the loop if centroids do not change
            y_pred = self.predict_cluster(clusters, X) # Make predictions
            if iteration % 20 == 0 and self.plot_figure: # If True, plot the scatter plot
                self.plot_fig(X, y_pred)
        return y_pred
   #calculate inertia 
    def inertia(self, X, clusters, centroids):
        total_inertia = 0
         # Calculate inertia within clusters
        for i, cluster in enumerate(clusters):
              # For each point in the cluster
            for point_idx in cluster:
                # Calculate squared distance from the point to its centroid and add it to total inertia
                total_inertia += np.linalg.norm(X[point_idx] - centroids[i])**2  # Squared distance
        return total_inertia
    
#calculate dunn index for corresponding 'k'
    def calculate_dunn_index(self, X, clusters):
        min_inter_cluster_distance = float('inf')  # Set initial value to infinity
        max_intra_cluster_distance = 0.0  # Set initial value to 0

        # Calculate minimum inter-cluster distance
        for i in range(len(self.centroids)):
            for j in range(i + 1, len(self.centroids)):
                 # Calculate Euclidean distance between cluster centroids
                dist = np.linalg.norm(self.centroids[i] - self.centroids[j]) 
                if dist < min_inter_cluster_distance:
                    min_inter_cluster_distance = dist

        # Calculate maximum intra-cluster distance
        for cluster in clusters:
            max_distance = 0
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    # Calculate Euclidean distance between data points within a cluster
                    dist = np.linalg.norm(X[cluster[i]] - X[cluster[j]]) 
                    if dist > max_distance:
                        max_distance = dist
            if max_distance > max_intra_cluster_distance:
                max_intra_cluster_distance = max_distance
 # Calculate Dunn Index by dividing min_inter_cluster_distance by max_intra_cluster_distance
        return min_inter_cluster_distance / max_intra_cluster_distance


# In[6]:


if __name__ == "__main__":
    np.random.seed(10)  # Set seed for reproducibility

    df = pd.read_csv("HouseholdWealth.csv")
    X = df.to_numpy()  # Convert DataFrame to numpy array

    dunn_inertia_results = []  # Initialize a list to store results

    for k in range(2, 11):  # Loop through K values from 2 to 10
        Kmeans = KMeansClustering(X, k) # Initialize KMeansClustering with varying K
        y_pred = Kmeans.fit(X)# Perform K-means clustering on data
        
# Create clusters
        clusters = Kmeans.create_cluster(X, Kmeans.centroids)
# Calculate inertia and Dunn Index for the current clustering
        inertia = Kmeans.inertia(X, clusters, Kmeans.centroids)
        dunn_index = Kmeans.calculate_dunn_index(X, clusters)


        dunn_inertia_results.append((k, dunn_index, inertia))  # Store K, Dunn Index, and inertia
        #Apparently the figure is not showing in the downloaded pdf or html file, hence use the ipynb file in notebook for the clustering figures


# In[7]:


print("K\tDunn Index\tInertia")
for result in dunn_inertia_results:
    print(f"{result[0]}\t{result[1]}\t{result[2]}")
    



# In[8]:


import matplotlib.pyplot as plt

# Assuming you have the results in the 'dunn_inertia_results' list

# Extract Dunn Index and Inertia values
dunn_values = [result[1] for result in dunn_inertia_results]
inertia_values = [result[2] for result in dunn_inertia_results]

# Find the optimal K based on Dunn Index
optimal_k_dunn = dunn_values.index(max(dunn_values)) + 2  # Add 2 to align with K range (from 2 to 10)

# Plot Inertia against K
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia_values, marker='o')
plt.title('Inertia against K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)

# Visual inspection for the "elbow" point in the inertia plot
# Suggests the optimal K where increasing clusters doesn't add much explanatory power


# Let's say the "elbow" point is found at K = 5
optimal_k_inertia = 5  # Replace this with the identified "elbow" K value from the plot

# Display the optimal K values based on both Dunn Index and Inertia
print(f'Optimal K based on Dunn Index: {optimal_k_dunn}')
print(f'Optimal K based on Inertia "elbow" method: {optimal_k_inertia}')


plt.show()  # Display the inertia plot


# In[9]:


# Highlight optimal K values in the plot
plt.axvline(x=optimal_k_dunn, color='r', linestyle='--', label=f'Optimal K (Dunn Index): {optimal_k_dunn}')
plt.axvline(x=optimal_k_inertia, color='g', linestyle='--', label=f'Optimal K (Inertia): {optimal_k_inertia}')
plt.legend()


# In[ ]:




