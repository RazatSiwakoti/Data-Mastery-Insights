#!/usr/bin/env python
# coding: utf-8

#Razat Siwakoti (A00046635)
#DMV302 - Assessment 2 
#NN2.ipynb created on Jupyter notebook


#source: Scikit-learn(2015)
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

#Deepika S. (2019)
#https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn


# In[1]:


#importing necessary libraries
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# Load the training dataset
df_train = pd.read_csv("AtRiskStudentsTraining.csv")
df_train.head()


# In[3]:


# Load the training dataset
df_test = pd.read_csv("AtRiskStudentsTest.csv")
df_test.head()


# In[4]:


# Separate features and target variable for both training and test sets
X_train = df_train.drop('at-risk', axis=1)
y_train = df_train['at-risk']
X_test = df_test.drop('at-risk', axis=1)
y_test = df_test['at-risk']

# Initialize a list to store the results
results = []

# Test various configurations of hidden layers and neurons
hidden_layers_list = [1, 2, 3, 4, 5]  # You can extend this list as needed
neurons_list = [50, 60, 70, 80, 90, 100]  # You can extend this list as needed


# In[5]:


for num_layers in hidden_layers_list:
    for num_neurons in neurons_list:
        # Create the neural network model
        model = MLPClassifier(hidden_layer_sizes=(num_neurons,) * num_layers, activation='relu', random_state=42)

        # Train the model on the training set
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate the error using accuracy as the metric
        error = 1 - accuracy_score(y_test, y_pred)
        
        # Calculate and print metrics for the test set
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        
        # Store the results
        results.append({
            'Hidden Layers': num_layers,
            'Neurons per Layer': num_neurons,
            'Error on Test Set': error,
            'Test Accuracy' : test_accuracy,
            'Precision' : test_precision,
            'Recall' : test_recall,
            'F1 score' : test_f1,
            
        })
        


# In[12]:


# Convert results to a DataFrame for better presentation
results_df = pd.DataFrame(results)
results_df


# In[13]:


#sort best results based on 'error io test set' and print the first ten in table 
best_configs = results_df.sort_values(by='Error on Test Set').head(10)
print("Top 10 Configurations:")
print(best_configs)

