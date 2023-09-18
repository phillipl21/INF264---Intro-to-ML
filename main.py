# INF264 Project 1
# Phillip Lei and Ryan Huynh

# Import libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # for 1.5 Comparison
from sklearn.metrics import accuracy_score

# TODO:
# - compare decision tree against DecisionTreeClassifier
#   - get accuracy comparison
#   - get speed comparison

# Classes
class Node:
    def __init__(self, depth, feature_index=None, split_threshold=None, class_label=None, left=None, right=None):
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
        self.depth = depth

    def __str__(self):
        """
        Return a string representation of the node
        """
        feature_string = ""

        # Get feature name
        feature_name = ""
        if self.feature_index == 0:
            feature_name = 'citric acid'
        elif self.feature_index == 1:
            feature_name = 'residual sugar'
        elif self.feature_index == 2:
            feature_name = 'pH'
        elif self.feature_index == 3:
            feature_name = 'sulphates' 
        elif self.feature_index == 4:
            feature_name = 'alcohol'

        # Get class label
        class_label = ""
        if self.class_label == 0:
            class_label = 'white'
        elif self.class_label == 1:
            class_label = 'red'

        if self.is_leaf():
            return f"Depth: {self.depth}, Class label: {class_label}"
        else:
            return f"Depth: {self.depth}, Feature: {feature_name}, Split threshold: {self.split_threshold}"

    def is_leaf(self):
        return not self.left and not self.right

class DecisionTree:
    def __init__(self, max_depth=20):
        """
        root: the root Node for the tree
        X and y: features and classes for each data point as Python lists
        max_depth: the maximum depth to build the tree
        """
        self.root = Node(0)
        self.X, self.y = read_data("wine_dataset.csv")
        self.num_features = len(self.X[0])
        self.total_dataset_size = len(self.y)
        self.impurity_measure = None
        self.max_depth = max_depth

    def create_tree(self, X, y, node, current_depth):
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
        elif self.identical_features(X) or current_depth == self.max_depth:
            node.class_label = self.most_common_label(y)
            node.left = None
            node.right = None
            return

        # Else create decision tree
        optimal_col_index = self.calculate_tree_split(X, y)
        x_1, y_1, x_2, y_2 = self.split_data(X, y, optimal_col_index)

        # Set feature to split on and its threshold
        node.feature_index = optimal_col_index
        feature_col = np.array([row[node.feature_index] for row in X])
        node.split_threshold = np.median(feature_col)

        # Run create_tree recursively right and left
        if len(y_1) != 0: # If data set exists
            node.left = Node(current_depth + 1)
            self.create_tree(x_1, y_1, node.left, current_depth + 1)
            
        if len(y_2) != 0:
            node.right = Node(current_depth + 1)
            self.create_tree(x_2, y_2, node.right, current_depth + 1)

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

        # Create the decision tree
        self.create_tree(X_train, y_train, self.root, 1)

        # Prune the tree if needed
        if prune == True:
            self.prune(X_test, y_test, self.root)

        return self

    def predict(self, x):
        """
        Predicts the class label based on a single data point x

        Parameters:
        - x: a data point

        Return value:
        - class_label: the predicted class label for that data point
        """
        # Basically just a wrapper for traverse
        return self.traverse(x, self.root)

    # Helper methods
    def identical_features(self, X):
        """
        Return True if all the features in each column of X are identical
        """
        if X is None or X[0] is None: # Null array
            print("identical_features() - No values found in array.")
            return False

        all_rows_same = np.all(X == X[0, :])
        return all_rows_same

    def split_data(self, X, y, feature_index):
        """
        Split the data into two sections based off the feature_index
        """
        feature_col = X[:,feature_index]
        split_threshold = np.median(feature_col)

        below_threshold_mask = feature_col <= split_threshold
        x_below, y_below = X[below_threshold_mask], y[below_threshold_mask]
        x_above, y_above = X[~below_threshold_mask], y[~below_threshold_mask]

        return x_below, y_below, x_above, y_above

    def calculate_tree_split(self, X, y):
        if self.impurity_measure == 'gini':
            return self.calculate_col_gini_split(X, y)
        elif self.impurity_measure == 'entropy':
            return self.calculate_entropy_split(X, y)
        else:
            print("Error in calculate_tree_split() - invalid impurity measure")

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

    def calc_col_entropy(self, X, y, feature_index):
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

    def calculate_entropy_split(self, X, y):
        """
        Return the best feature to split at and what value to split at
        """
        # Get total entropy for y
        proportion_white, proportion_red = np.sum(y == 0.0) / len(y), np.sum(y == 1.0) / len(y)
        total_entropy = -proportion_white * np.log2(proportion_white) - proportion_red * np.log2(proportion_red)

        # Calculate information gain for each feature
        optimal_info_gain, optimal_col_index = 0, 0

        for col_index in range(self.num_features):
            col_entropy = self.calc_col_entropy(X, y, col_index)
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

    def calculate_col_gini(self, X, y, feature_index):
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
    
    def calculate_col_gini_split(self, X, y):
        """
        Return best feature and index to split at
        """
        # Calculate information gain for each feature
        best_gini , best_col = 999, 0

        for col_index in range(self.num_features):
            gini = self.calculate_col_gini(X, y, col_index)

            # Update optimal information gain and column index
            if gini < best_gini:
                best_gini, best_col = gini, col_index

        return best_col
        
    def most_common_label(self, y):
        """
        Return 0 or 1 depending on whichever is the most common label
        """
        proportion_white = np.sum(y == 0) / len(y)
        return 0 if proportion_white > 0.5 else 1

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
            print(node)
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
        if not node:
            print("Error in traverse: null node")
            return None

        # Return label if the node is a leaf
        if node.is_leaf() == True:
            return node.class_label

        x_feature = x[node.feature_index]

        if x_feature <= node.split_threshold:
            return self.traverse(x, node.left)
        else:
            return self.traverse(x, node.right)

    # 1.3 - Pruning
    def prune(self, X, y, tree):
        """
        Post-pruning: should prune leaves/subtrees of tree to 
        reduce overfitting
        """
        # Return 0 if there's no data
        if X == None or len(X) == 0:
            return 0

        # Return error of a single leaf
        if tree.is_leaf():
            return len(y) - y.count(tree.class_label)
        
        # Get majority label of tree
        most_common_label = self.most_common_label(y)
        
        # Split data into two halves
        x_1, y_1, x_2, y_2 = self.split_data(X, y, tree.feature_index)
        
        # Process: We need to get the accuracy of the left and right subtrees
        # and compare that to the accuracy of the majority label. If replacing
        # the subtree with a node that predicts the ML doesn't reduce the 
        # accuracy of the tree, then go through with it. Pruning is a 
        # bottom-up process from the leaves.
        
        # Calculate the label inaccuracies for ML, left, and right
        most_common_label_inaccuracy = len(y) - y.count(tree.class_label)
        left_subtree_inaccurate_labels = self.prune(x_1, y_1, tree.left)
        right_subtree_inaccurate_labels = self.prune(x_2, y_2, tree.right)
        subtree_inaccuracy_total = left_subtree_inaccurate_labels + right_subtree_inaccurate_labels
        
        # Compare inaccuracies and replace w/ a node or not 
        if subtree_inaccuracy_total > most_common_label_inaccuracy:
            # Prune the subtree
            tree.left = None
            tree.right = None
            tree.class_label = most_common_label
            
            # Return majority label up a recursive level if pruned
            return most_common_label_inaccuracy
        
        return subtree_inaccuracy_total

    def tree_accuracy(self, X_test, y_test):
        """
        Return the accuracy of the decision tree
        """
        # Get all of the predictions for each data point
        predicted_labels = np.empty((0,0))
        test_size = len(X_test)

        for i in range(test_size):
            data_point = X_test[i]
            prediction = self.predict(data_point)
            predicted_labels = np.append(predicted_labels, prediction)

        pct_correct = (predicted_labels == y_test).sum() / test_size
        return pct_correct
    
def graph_accuracy(max_depth):
    """
    Graph the accuracies of decision trees from depths 1 to max_depth (inclusive)
    """
    depths = []
    accuracies = []


    for depth in range(1, max_depth + 1):
        print("depth: ", depth)
        tree = DecisionTree(depth)
        tree.learn(X, y, 'entropy')
        accuracy = tree.accuracy()

        depths.append(depth)
        accuracies.append(accuracy)
    
    plt.plot(depths, accuracies)
    plt.xlabel('Decision Tree Depth')  # Set the label for the x-axis
    plt.ylabel('Accuracy')  # Set the label for the y-axis
    plt.title('The Effect of Decision Tree Depth on Observed Dataset Accuracy')  # Set the title of the chart
    plt.grid(True)  # Display a grid
    plt.ylim(0.25, 1)
    plt.show()  # Display the chart


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

    return np.array(X), np.array(y)

# Main
if __name__ == "__main__":
    print("INF264 Project 1")

    # Import data and preprocess it
    csv_file = "wine_dataset.csv"
    X, y = read_data(csv_file)
    
    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    # Create and test a tree
    impurity_measure = 'entropy'
    
    start_time = time.time()
    tree = DecisionTree()
    tree.learn(X_train, y_train, impurity_measure)
    end_time = time.time() - start_time
    
    # tree.print_subtree(tree.root)
    print("Our tree implementation accuracy:", tree.tree_accuracy(X_test, y_test))
    print(f"Time to create and learn our decision tree: {end_time} seconds")

    # Test DecisionTreeClassifier on the data
    start_time = time.time()
    sklearn_tree = DecisionTreeClassifier(criterion=impurity_measure)
    sklearn_tree.fit(X_train, y_train)
    predictions = sklearn_tree.predict(X_test)
    end_time = time.time() - start_time

    sklearn_accuracy = accuracy_score(y_test, predictions)

    print("sklearn accuracy:", sklearn_accuracy)
    print("sklearn tree depth:", sklearn_tree.tree_.max_depth)
    print(f"Time to create and learn sklearn decision tree: {end_time} seconds")