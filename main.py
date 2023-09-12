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
            node.label = y[0]
            return node

        # Elif all data points have identical feature values, return a leaf with the most common label
        elif self.identical_features(x):
            node.label = self.most_common_label(y)
            return node

        # Else create decision tree
        # - Choose a feature with the most infomation gain
        # - Split the data based on the feature's value and add a branch for each subset of data
        # - For each branch, call the algorithm recursively on the data points for that specific branch
        
        # For pruning, need to run a majority_label function
        # Split data in 2 - split either using Gini Index or Entropy based on 
        # chosen parameter. Will likely require helper function to figure
        # out the best way to split.
        optimal_column = self.calculate_optimal_tree_split(X, y)
        x_1, y_1, x_2, y_2 = self.split_data(X, y, optimal_column)
        
        # Run create_tree recursively right and left
        # IF Condition
            # node.left = Node()
            # self.create_tree(x_1, y_1, node.left)
        # ELSE IF Condition
            # node.right = Node()
            # self.create_tree(x_2, y_2, node.right)
        
        
    # TODO: finish function
    def learn(self, X, y, impurity_measure='entropy', prune='False'):
        """
        Create a decision tree based on input data and an impurity measure

        Parameters:
        - X: a data matrix of continuous features
        - y: a label vector of categorical variables
        - impurity_measure: determines how to split the branches
        """
        # Store this for later functions in the class
        self.impurity_measure = impurity_measure
        
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
    
    def split_data(self, X, y, optimal_column):
        # Create lists for splitting the data into 2 parts
        x_1, y_1 = [], []
        x_2, y_2 = [], []
        
        # Get split_threshold
        column_values = [row[optimal_column] for row in X]
        split_threshold = np.median(optimal_column)
        
        # Divide features and corresponding labels into two sets
        for i, row in enumerate(X):
            feature = row[optimal_column]
            
            if feature > split_threshold:
                x_1.append(row)
                y_1.append(y[optimal_column])
            elif feature <= split_threshold:
                x_2.append(row)
                y_2.append(y[optimal_column])
        
        return x_1, y_1, x_2, y_2
        

    def calculate_entropy(self, x, y, col_index):
        """
        Return a single value for entropy from a column of interest from X
        0 signifies white wine and 1 signifies red wine
        """

        # Get the column of interest from the table X
        column = [row[col_index] for row in x]
        total_dataset_size = len(column)
        
        # Choose the splitting threshold
        split_threshold = np.median(column)

        # Get counts for each subset
        below_white, below_red, above_white, above_red = 0, 0, 0, 0

        for i in range(len(column)):
            feature = column[i]

            if feature <= split_threshold:
                if y[i] == 0:
                    below_white += 1
                else:
                    below_red += 1
            else:
                if y[i] == 0:
                    above_white += 1
                else:
                    above_red += 1
        
        # Calculate entropy for below subset
        total_below = below_white + below_red
        p_below_white = below_white / total_below
        p_below_red = below_red / total_below
        entropy_below = -p_below_white * np.log2(p_below_white) - p_below_red * np.log2(p_below_red)

        # Calculate entropy for above subset
        total_above = above_white + above_red
        p_above_white = above_white / total_above
        p_above_red = above_red / total_above
        entropy_above = -p_above_white * np.log2(p_above_white) - p_above_red * np.log2(p_above_red)

        # Calculate total weighted entropy
        proportion_below = total_below / total_dataset_size
        proportion_above = total_above / total_dataset_size

        total_entropy = proportion_below * entropy_below + proportion_above * entropy_above
        return total_entropy
    
    # TODO: add option for both Gini and Entropy based on earlier paraameters
    def calculate_optimal_tree_split(self, X, y):
        if self.impurity_measure == 'gini':
            pass
        elif self.impurity_measure == 'entropy':
            return self.calculate_optimal_entropy_split(X, y)
        
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
            col_entropy = 100000
            # col_entropy = self.calculate_entropy(X, y, col_index)
            information_gain = total_entropy - col_entropy

            # print("col_entropy =", col_entropy)
            # print("information_gain =", information_gain)

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

    # 1.3 - Pruning
    def prune_tree(self, X, y, tree):
        pass
    
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
    X, y = read_data(csv_file)

    tree = DecisionTree()
    calc_entropy_result = tree.calculate_optimal_entropy_split(tree.X, tree.y)
    print("calc_entropy_result =", calc_entropy_result)

    for i in range(5):
        print("entropy sample: ", tree.calculate_entropy(X, y, i))