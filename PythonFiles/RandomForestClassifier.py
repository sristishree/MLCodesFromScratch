#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 00:06:57 2021

@author: sristi
"""

##########################################################################################################
###################### Code for Decision Tree Classification Code from scratch ###########################
##########################################################################################################


import pandas as pd
import numpy as np

# Get the data
from sklearn.datasets import load_iris
data = load_iris(return_X_y = True, as_frame = True)



# Node class

class Node():
    def __init__(self, feature_key = None, feature_threshold = None, left_child = None, right_child = None, information_gain = None, value = None):
        '''
        Constructor
        '''
        
        # For decision node
        self.feature_key = feature_key
        self.feature_threshold = feature_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.information_gain = information_gain
        
        # For leaf node
        self.value = value
        
        
class TreeClassifier():
    def __init__(self, minimum_samples_split = 2, max_depth = 2):
        '''
        Constructor
        '''
        self.root = None
        
        # Stopping conditions
        self.minimum_samples_split = minimum_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, cur_depth = 0):
        '''
        Function to build the decision tree recursively
        '''
        
        X = dataset[:,:-1]
        y = dataset[:,-1]
        num_samples, num_features = X.shape
        
        # Check for stopping conditions
        if num_samples >= self.minimum_samples_split and cur_depth <= self.max_depth:
             # Find the best split
             best_split = self.get_best_split(dataset, num_samples, num_features)
             # Check if information gain is positive, information gain is 0 for pure split
             if best_split['information_gain'] > 0:
                 # Form the left subtree using recursion
                 left_subtree = self.build_tree(best_split['dataset_left'], cur_depth+1)
                 # Form the right subtree using recursion
                 right_subtree = self.build_tree(best_split['dataset_right'], cur_depth+1)
                 # Return the node on which the current decsion has been made
                 return Node(best_split['feature_key'], best_split['feature_threshold'], left_subtree, right_subtree, best_split['information_gain'])
                 
        # Compute leaf node
        leaf_value = self.calculate_leaf_value(y)
        # Return the leaf node
        return Node(value = leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        '''
        Function to find the best split based on information gain
        '''
        
        # Dictionary to store the best split result
        best_split = {}
        
        max_info_gain = -float('inf')
        
        # Loop over all features
        for feature_index in range(num_features):
            # feature_values =  dataset[:][feature_index]
            feature_values =  dataset[:,feature_index]
            # Possible values of thresholds
            # Could be infinite possible values on the real scale in our range, 
            # but we will take only the ones which are present in the feature set 
            thresholds_possible_values = np.unique(feature_values)
            # Loop over all possible threshold values present for that feature    
            for threshold in thresholds_possible_values:
                # Split the data on given threshold
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # Check if children have size greater than 0
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    # Extract target values
                    y, y_left, y_right = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]
                    # Compute information gain
                    cur_information_gain = self.information_gain(y, y_left, y_right, "gini")
                    # Compare with the max information gain
                    if cur_information_gain > max_info_gain:
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['feature_key'] = feature_index
                        best_split['feature_threshold'] = threshold
                        best_split['information_gain'] = cur_information_gain
                        max_info_gain = cur_information_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        '''
        Function to split the dataset into left and right according to the threshold value
        '''
        #dataset_left = dataset[dataset[:,feature_index] <= threshold]
        #dataset_right = dataset[dataset[:,feature_index] > threshold]
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        
        return dataset_left, dataset_right
    
    def information_gain(self, y, y_left, y_right, mode='entropy'):
        '''
        Function to calculate information gain using gini or entropy
        '''
        weight_l = len(y_left) / len(y)
        weight_r = len(y_right) / len(y)
        
        if mode == "gini":
            return self.gini_impurity(y) - ( weight_l*self.gini_impurity(y_left) + weight_r*self.gini_impurity(y_right))
        else:
            return self.entropy(y) - ( weight_l*self.entropy(y_left) + weight_r*self.entropy(y_right))
        
    def entropy(self, y):
        class_labels = np.unique(y)
        ent = 0
        
        for label in class_labels:
            p_class = len(y[y == label]) / len(y)
            ent += -p_class * np.log2(p_class)
            
        return ent
        
    def gini_impurity(self, y):
        class_labels = np.unique(y)
        gini = 0
        
        for label in class_labels:
            p_class = len(y[y == label]) / len(y)
            gini += p_class ** 2
            
        return 1-gini
        
    def calculate_leaf_value(self, y):
        '''
        Function to compute the leaf node on the basis of max occurring element in target
        '''
        Y = list(y)
        return max(Y, key = Y.count)
    
    def fit(self, X, y):
        '''
        Function to train the model
        '''
        
        dataset = X
        dataset['y'] = y
        dataset = np.array(dataset)
        self.root = self.build_tree(dataset)
        
    def predict(self, X):
        '''
        Function to predict on the basis of the trained model
        '''
        X = np.array(X)
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        '''
        Function to predict a single data point
        '''
        # If you encounter the leaf node, that will be the prediction
        if tree.value is not None:
            return tree.value
        # Else recursively try to reach the leaf
        else:
            if x[tree.feature_key] <= tree.feature_threshold:
                return self.make_prediction(x, tree.left_child)
            else:
                return self.make_prediction(x, tree.right_child)
        
    def print_tree(self, tree=None, spacing=3, depth=1):
        '''
        Function to print the tree
        '''
        if spacing <=0:
            raise ValueError("Spacing must be a positive integer")
        
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing
        
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)
            
        else:
            print(f'\n{indent} Feature {tree.feature_key}  <= {tree.feature_threshold} ? Information gain is {tree.information_gain}')
            print(f'{indent} Left: ', end='')
            self.print_tree(tree.left_child, 3, depth+1)
            print(f'{indent} Right: ', end='')
            self.print_tree(tree.right_child, 3, depth+1)
            
#############################################################################################################
# Train test split
            
X = data[0]
y = data[1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

#############################################################################################################
# Fit the model

classifier = TreeClassifier(minimum_samples_split=3, max_depth=3)
classifier.fit(X_train, y_train)
classifier.print_tree()

#############################################################################################################

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
