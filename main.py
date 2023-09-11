# INF264 Project 1
# Phillip Lei and Ryan Huynh

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random

from csv import reader

# Classes
class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        pass

class DecisionTree:
    def __init__(self):
        self.root = Node()

    # TODO: finish function
    def create_tree(self, x, y, node):
        """
        Create a decision tree based on input data
        
        Parameters:
        x and y: a list of data representing input and output
        node: the starting node of the tree (can be a subset of another tree)
        """
        # If all data points have the same label, return a leaf with that label
        if len(set(y)) == 1:
            return node

        # Elif all data points have identical feature values, return a leaf with the most common label
        elif identical_features(x):
            return most_common_label(y)

        # Else create decision tree
        # - Choose a feature with the most infomation gain
        # - Split the data based on the feature's value and add a branch for each subset of data
        # - For each branch, call the algorithm recursively on the data points for that specific branch
        else:
            pass

    # TODO: finish function
    def learn(self, X, y, impurity_measure='entropy'):
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
        pass

    # TODO: finish function
    def calculate_entropy(self, x, i):
        """
        Return a single value for entropy from a column of interest from X
        """
        # Get the column of interest from the table X
        column = [row[i] for row in x]
        
        # Count the frequency of each unique value in the data
        values_dict = {}
        for i in column:
            if i in values_dict:
                values_dict[i] += 1
            else:
                values_dict[i] = 1
        
        entropy = 0.0
        total_count = len(column)
        
        # Iteratively calculate the entropy for a different of x and sum it.
        # entropy -= since it's the same as multiplying the overal sum by -1
        for count in values_dict.values():
            probability = count / total_count
            entropy -= probability * np.log2(probability)
        
        return entropy

    def most_common_label(self, y):
        """
        Return 0 or 1 depending on whichever is the most common label
        """
        if y.count(0) > len(y) / 2:
            return 0
        else:
            return 1

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

    # Read data from CSV file line by line
    with open(filename, newline='') as csvfile:
        csv_reader = reader(csvfile, delimiter=' ', quotechar='|')
        csv_as_list = list(csv_reader)

        for i in range(len(csv_as_list)):
            if i > 0:
                # Add labels to y
                row = csv_as_list[i][0].split(',')
                y.append(float(row.pop()))

                # Add labels to X
                features = [float(x_val) for x_val in row]
                X.append(features)

    return X, y

# Main
if __name__ == "__main__":
    print("INF264 Project 1")
    csv_file = "wine_dataset.csv"
    X, y = read_data(csv_file)
    print(X)