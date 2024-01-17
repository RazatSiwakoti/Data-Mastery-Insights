#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Razat Siwakoti (A00046635)
#DMV302 - Assessment 2 
#Kmeans1.ipynb created on Jupyter notebook


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


k = 3  # Setting the number of clusters to k
# Initializing a dictionary to store clusters, where keys are cluster indices and values are empty lists
clusters = {}
# Looping through the range of k to create empty lists for each cluster
for i in range(k):
    clusters[i] = []


# In[6]:


class KMeansClustering:
    def __init__(self, X, num_clusters):
        # Initialization of parameters
        self.K = num_clusters # Number of clusters
        self.max_iterations = 100 # Maximum number of iterations to avoid running infinitely
        self.num_examples, self.num_features = X.shape # num of examples, num of features
        self.plot_figure = True # Whether to plot figures during the iterations
        self.final_centroids = None  # Variable to store final centroids
        
        
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
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids) # create cluster
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X) # Calculate new centroids based on current clusters
            diff = centroids - previous_centroids # calculate difference
            if not diff.any():
                break   # Break the loop if centroids do not change
            y_pred = self.predict_cluster(clusters, X) # Make predictions
            if self.plot_figure: # If True, plot the scatter plot
                self.plot_fig(X, y_pred)
        self.final_centroids = centroids
        return y_pred
            


# In[7]:


if __name__ == "__main__":
    np.random.seed(10) #  Set seed for reproducibility
    num_clusters = 3  # num of cluster
    df = pd.read_csv("HouseholdWealth.csv")
    
    # Generate synthetic data with make_blobs
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=num_clusters, random_state=10)
        # Initialize and fit KMeans clustering
    Kmeans = KMeansClustering(X, num_clusters)
    y_pred = Kmeans.fit(X)
#Apparently the figure is not showing in the downloaded pdf or html file, hence use the ipynb file in notebook for the clustering figures


# In[8]:


final_centroids = Kmeans.final_centroids  # Retrieve the final centroids
print("Final Centroids:")
print(final_centroids)  # Print or use these centroids for reporting


# In[9]:


from sklearn import preprocessing
from sklearn import cluster
scaler = preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(df)

pd.DataFrame(data_scaled).describe()

kmeans = cluster.KMeans(n_clusters=2, init='k-means++') # ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
kmeans.fit(data_scaled)
kmeans.inertia_

SSE = []
for i in range(1,20):
    kmeans = cluster.KMeans(n_clusters = i, init='k-means++') # iterate from range (1, 20)
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker="*")
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = cluster.KMeans(n_clusters=5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
pred



# In[10]:


# Visualize the clustered data on the original scale if needed
# For example, you can create a scatter plot with colors representing clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['household_total_assets'], df['annual_household_income'], c=pred, cmap='viridis', alpha=0.5)
plt.title('KMeans Clustering Results')
plt.xlabel('household_total_assets')
plt.ylabel('annual_household_income')
plt.show()

