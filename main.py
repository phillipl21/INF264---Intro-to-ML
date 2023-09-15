# INF264 Project 1
# Phillip Lei and Ryan Huynh

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# TODO:
# - implement prune function
# - finish implementing gini index
# - do we need to return split_threshold from calculate_optimal_entropy_split?
# - test the code

# Classes
class Node:
    def __init__(self, feature_index=None, split_threshold=None, class_label=None, left=None, right=None):
        """
        Parameters:
         - feature_index: int specifying which feature to split on
         - split_threshold: float specifying where to split
         - class_label: int specifying the result (0 for white, 1 for red). Only stored for leaf nodes
         - left: left node child
         - right: right node child
        """
        self.feature_index = feature_index
        self.split_threshold = split_threshold
        self.class_label = class_label
        self.left = left
        self.right = right

    def is_leaf(self):
        return not self.left and not self.right

class DecisionTree:
    def __init__(self):
        """
        root: the root Node for the tree
        X and y: features and classes for each data point as Python lists
        max_depth: the maximum depth to build the tree
        """
        self.root = Node()
        self.X, self.y = read_data("wine_dataset.csv")
        self.max_depth = 7

    # TODO: finish function
    def create_tree(self, x, y, node, current_depth):
        """
        Create a decision tree based on input data
        Modifies the node passed in directly

        Parameters:
        x and y: a list of data representing input and output
        node: the starting node of the tree (can be a subset of another tree)
        """
        # Base cases:
        # If all data points have the same label, return a leaf with that label
        if len(set(y)) == 1:
            node.class_label = y[0]
            node.left = None
            node.right = None
            return

        # Elif all data points have identical feature values
        # or max depth is reached
        # return a leaf with the most common label
        elif self.identical_features(x) or current_depth == max_depth:
            node.class_label = self.most_common_label(y)
            node.left = None
            node.right = None
            return

        # Else create decision tree
        # - Choose a feature with the most infomation gain
        # - Split the data based on the feature's value and add a branch for each subset of data
        # - For each branch, call the algorithm recursively on the data points for that specific branch
        
        # For pruning, need to run a majority_label function
        # Split data in 2 - split either using Gini Index or Entropy based on 
        # chosen parameter. Will likely require helper function to figure
        # out the best way to split.
        optimal_col_index = self.calculate_optimal_tree_split(X, y)
        x_1, y_1, x_2, y_2 = self.split_data(X, y, optimal_col_index)

        # Set feature to split on and its threshold
        node.feature_index = optimal_col_index
        feature_col = np.array([row[i] for row in X])
        node.split_threshold = np.median(feature_col)

        # Run create_tree recursively right and left
        if len(y_1) != 0: # If data set exists
            node.left = Node()
            self.create_tree(x_1, y_1, node.left)
            
        if len(y_2) != 0:
            node.right = Node()
            self.create_tree(x_2, y_2, node.right)

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

        # Split data into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

        # Create the decision tree
        self.create_tree(X_train, y_train, self.root)

        # Prune the tree if needed
        if prune:
            self.prune(X_test, y_test, self.root)

        return self

    # TODO: finish function
    def predict(self, x):
        """
        Predicts the class label based on a single data point x

        Parameters:
        - x: a data point

        Return value:
        - class_label
        """
        # Basically just a wrapper for traverse
        return self.traverse(x, self.tree)

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

    def split_data(self, X, y, feature_index):
        """
        Split the data into two sections based off the feature_index
        """
        feature_col = X[:,feature_index]
        split_threshold = np.median(column_values)

        below_threshold_mask = feature_col <= split_threshold
        x_below, y_below = X[below_threshold_mask], y[below_threshold_mask]
        x_above, y_above = X[~below_threshold_mask], y[~below_threshold_mask]

        return x_below, y_below, x_above, y_above

    def calculate_optimal_tree_split(self, X, y):
        if self.impurity_measure == 'gini':
            return self.calculate_optimal_gini_index_split(X, y)
        elif self.impurity_measure == 'entropy':
            return self.calculate_optimal_entropy_split(X, y)
        else:
            print("Error in calculate_optimal_tree_split() - invalid impurity measure")

    def weighted_subset_entropy(self, df, split_threshold, dataset_size, which_half):
        """
        Return the weighted subset entropy of either the upper or lower half of the dataset
        Returns -1 on error
        """
        if dataset_size == 0:
            print("Error in weighted_subset_entropy: dataset_size is 0")
            return -1

        # Create subset
        if which_half == 'below':
            subset = df[df['feature'] <= split_threshold]

        elif which_half == 'above':
            subset = df[df['feature'] > split_threshold]
        
        else:
            print("Error in weighted_subset_entropy: invalid which_half value")
            return -1

        # Calculate weighted entropy value
        subset_counts = subset['label'].value_counts()
        class_proportions = subset_counts / len(subset)
        unweighted_entropy = -(class_proportions * np.log2(class_proportions)).sum()
        subset_weight = len(subset) / dataset_size

        return subset_weight * unweighted_entropy

    def calc_entropy(self, X, y, feature_index):
        """
        Return a single value for entropy from a column of interest from X
        0 signifies white wine and 1 signifies red wine
        """
        # Get the column of interest from the table X
        column = np.array([row[feature_index] for row in X])
        dataset_size = len(column)
        df = pd.DataFrame({'feature': column, 'label': y})

        # Choose the splitting threshold
        split_threshold = np.median(column)

        # Get entropy for each subset
        weighted_below_entropy = self.weighted_subset_entropy(df, split_threshold, dataset_size, 'below')
        weighted_above_entropy = self.weighted_subset_entropy(df, split_threshold, dataset_size, 'above')

        return weighted_below_entropy + weighted_above_entropy

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
            col_entropy = self.calc_entropy(col_index)
            information_gain = total_entropy - col_entropy

            # Update optimal information gain and column index
            if information_gain > optimal_info_gain:
                optimal_info_gain = information_gain
                optimal_col_index = col_index

        # Return feature with the best information gain
        return optimal_col_index

    def weighted_subset_gini(self, df, split_threshold, dataset_size, which_half):
        """
        Returns the weighted gini index for a given subset
        Returns -1 on error
        """
        if dataset_size == 0:
            print("Error in weighted_subset_gini: dataset_size is 0")
            return -1

        # Create subset
        if which_half == 'below':
            subset = df[df['feature'] <= split_threshold]

        elif which_half == 'above':
            subset = df[df['feature'] > split_threshold]

        else:
            print("Error in weighted_subset_gini: invalid which_half value")
            return -1

        # Calculate weighted gini value
        subset_counts = subset['label'].value_counts()
        class_proportions = subset_counts / len(subset)
        subset_weight = len(subset) / dataset_size
        unweighted_gini = 1 - (class_proportions**2).sum()

        return subset_weight * unweighted_gini

    def calculate_gini_index(self, X, y, feature_index):
        """
        Return a single value for Gini index from a column of interest from X.
        Gini Index is another impurity measurement. Will be similar to entropy 
        calculation function.
        """
        # Get the column of interest from the table X
        column = np.array([row[feature_index] for row in X])
        dataset_size = len(column)
        df = pd.DataFrame({'feature': column, 'label': y})

        # Choose the splitting threshold
        split_threshold = np.median(column)

        weighted_below_gini = self.weighted_subset_gini(df, split_threshold, dataset_size, 'below')
        weighted_above_gini = self.weighted_subset_gini(df, split_threshold, dataset_size, 'above')

        return weighted_below_gini + weighted_above_gini
    
    # TODO: finish function!
    def calculate_optimal_gini_index_split(self, X, y):
        """
        Return best feature and index to split at
        """
        pass
        
    def most_common_label(self, y):
        """
        Return 0 or 1 depending on whichever is the most common label
        """
        return max(y, key=y.count)

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

    def traverse(self, x, node):
        """
        Traverse the tree starting at the specified node

        Parameters:
         - x: an input data point represented as a row of feature values
         - node: the starting node of the tree or subtree

        Return value: a int representing the leaf node class label
        """
        # Return label if the node is a leaf
        if node.is_leaf() == True:
            return node.class_label

        x_feature = x[node.feature_index]

        if x_feature <= node.split_threshold:
            return self.traverse(x, node.left)
        else:
            return self.traverse(x, node.right)

    # 1.3 - Pruning
    # TODO: finish
    def prune(self, X, y, tree):
        """
        Post-pruning: should prune leaves/subtrees of tree to 
        reduce overfitting
        """
        # Return 0 if there's no data
        if X == None:
            return 0
        # Return error of a single leaf
        if tree.is_leaf():
            return len(y) - y.count(tree.class_label)
        
        # Get majority label of tree
        majority_label = self.most_common_label(y)
        
        #TODO: is feature_index the correct patameter?
        x_1, y_1, x_2, y_2 = self.split_data(X, y, tree.feature_index)
        
        # Process: We need to get the accuracy of the left and right subtrees
        # and compare that to the accuracy of the majority label. If replacing
        # the subtree with a node that predicts the ML doesn't reduce the 
        # accuracy of the tree, then go through with it. Pruning is a 
        # bottom-up process from the leaves.
        
        # Calculate the label inaccuracies for ML, left, and right
        majority_label_inaccuracy = len(y) - y.count(tree.class_label)
        left_subtree_inaccurate_labels = self.prune(x_1, y_1, tree.left)
        right_subtree_inaccurate_labels = self.prune(x_2, y_2, tree.right)
        subtree_inaccuracy_total = left_subtree_inaccurate_labels + right_subtree_inaccurate_labels
        
        # Compare inaccuracies and replace w/ a node or not 
        if subtree_inaccuracy_total > majority_label_inaccuracy:
            # Prune the subtree
            tree.left = None
            tree.right = None
            tree.class_label = majority_label
            
            # Return majority label up a recursive level if pruned
            return majority_label_inaccuracy
        
        return subtree_inaccuracy_total

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
    # tree.learn(X, y, impurity_measure='entropy', prune='False')
    tree.split_data(X, y, 0)

    # calc_entropy_result = tree.calculate_optimal_entropy_split(tree.X, tree.y)
    # print("calc_entropy_result =", calc_entropy_result)

    tree.calculate_gini_index(X, y, 0)