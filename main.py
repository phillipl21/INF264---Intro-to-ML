# INF264 Project 1
# Phillip Lei and Ryan Huynh

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

from csv import reader

# Classes
class Node:
    def __init__(self, label, data, left=None, right=None):
        self.data = data
        self.label = label
        self.left = left
        self.right = right
        pass

    def is_leaf(self):
        return self.right == None and self.left == None

class DecisionTree:
    def __init__(self):
        self.root = Node(None, None)
        self.X, self.y = read_data("wine_dataset.csv")

    # TODO: finish function
    def create_tree(self, x, y, node):
        """
        Create a decision tree based on input data
        
        Parameters:
        x and y: a list of data representing input and output
        node: the starting node of the tree (can be a subset of another tree)
        """
        # Base cases:
        
        # If all data points have the same label, return a leaf with that label
        if len(set(y)) == 1:
            return node

        # Elif all data points have identical feature values, return a leaf with the most common label
        elif self.identical_features(x):
            return self.most_common_label(y)

        # Else create decision tree
        # - Choose a feature with the most infomation gain
        # - Split the data based on the feature's value and add a branch for each subset of data
        # - For each branch, call the algorithm recursively on the data points for that specific branch
        else:
            pass

    # TODO: finish function
    def learn(self, X, y, impurity_measure='entropy', prune='False'):
        """
        Create a decision tree based on input data and an impurity measure

        Parameters:
        - X: a data matrix of continuous features
        - y: a label vector of categorical variables
        - impurity_measure: determines how to split the branches
        """
        
        pass

    # TODO: finish function
    def predict(self, x):
        """
        Predicts the class label based on a single data point x

        Parameters:
        - x: a data point

        Return value:
        - class_label
        """
        pass

    # Helper methods
    # TODO: finish function
    def identical_features(self, X):
        """
        Return True if all the features in each column of X are identical
        """
        if X is None or X[0] is None: # Null array
            print("identical_features() - No values found in array.")
            return False
        
        # Get horizontal length of 2D array
        for i in range(len(X[0])):
            # Iterate through the columns of the array
            column = [row[i] for row in X]
            # Check for value uniqueness by comparing with the first value 
            for val in column:
                if val is not column[0]:
                    return False
            
        return True

    def calculate_entropy(self, x, i):
        """
        Return a single value for entropy from a column of interest from X
        """
        # Get the column of interest from the table X
        column = [row[i] for row in x]
        
        # Count the frequency of each unique value in the data
        values_dict = {}
        for count in column:
            if count in values_dict:
                values_dict[count] += 1
            else:
                values_dict[count] = 1
        
        entropy = 0.0
        total_count = len(column)
        
        # Iteratively calculate the entropy for a different of x and sum it.
        for count in values_dict.values():
            probability = count / total_count
            if probability != 0:
                entropy += probability * math.log2(probability)
        return -entropy

    # TODO: figure out calculate_entropy function
    # TODO: determine how to split data based off information gain
    def calculate_optimal_entropy_split(self, X, y):
        """
        Return the best feature to split at and what value to split at
        """
        # Get total entropy for y
        proportion_white = y.count(0) / len(y)
        proportion_red = y.count(1) / len(y)

        total_entropy = -proportion_white * np.log2(proportion_white) - proportion_red * np.log2(proportion_red)

        # Calculate information gain for each feature
        optimal_info_gain, optimal_col_index = 0, 0

        for col_index in range(len(X[0])):
            col_entropy = self.calculate_entropy(X, col_index)
            information_gain = total_entropy - col_entropy

            print("col_entropy =", col_entropy)
            print("information_gain =", information_gain)

            # Update optimal information gain and column index
            if information_gain > optimal_info_gain:
                optimal_info_gain = information_gain
                optimal_col_index = col_index

        # Return feature with the best information gain
        return optimal_col_index

    def most_common_label(self, y):
        """
        Return 0 or 1 depending on whichever is the most common label
        """
        if y.count(0) > len(y) / 2:
            return 0
        else:
            return 1

    def has_same_label(self, y):
        """
        Check whether the labels in a collection are the same or not
        """
        for label in y:
            # compare current label with first label
            if label is not y[0]: 
                return False

        return True

    def print_subtree(self, node, depth=0):
        """
        Prints the subtree starting at the specified node
        """
        if node:
            print(f"Depth: {node.depth} Data: {node.data}")
            self.print_subtree(node.left, depth + 1)
            self.print_subtree(node.right, depth + 1)

    def train_validation_split(self, X, y, pct_validation):
        """
        Split X and y into train and validation sets. 
        pct_validation determines how much of the data should be validation set

        Return the train set as the first two values
        and the validation set as the last two values
        """
        X_validation, y_validation = [], []
        validation_set_size = round(len(X) * pct_validation)

        # Pick validation_set_size elements and add to validation arrays
        while len(X_validation) < validation_set_size:
            random_index = random.randrange(len(X))
            X_validation.append(X.pop(random_index))
            y_validation.append(y.pop(random_index))
        
        train_set = X, y
        validation_set = X_validation, y_validation
        return train_set, validation_set

# Other functions
def read_data(filename):
    """
    Read data from a file and return a populated feature (X) and label matrix (y)
    """
    # Feature and label matrices
    X, y = [], []

    df = pd.read_csv("wine_dataset.csv")

    for i in range(len(df)):
        # Add labels to y
        row = df.iloc[i]
        y.append(float(row.pop('type')))

        # Add labels to X
        features = [float(x_val) for x_val in row]
        X.append(features)

    return X, y

# Main
if __name__ == "__main__":
    print("INF264 Project 1")
    csv_file = "wine_dataset.csv"
    # X, y = read_data(csv_file)

    tree = DecisionTree()
    calc_entropy_result = tree.calculate_optimal_entropy_split(tree.X, tree.y)
    print("calc_entropy_result =", calc_entropy_result)
    print("entropy sample: ", tree.calculate_entropy(X, 0))