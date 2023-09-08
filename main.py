# INF264 Project 1
# Phillip Lei and Ryan Huynh

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

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

    def create_tree(self, x, y, node):
        """
        Create a decision tree based on input data
        
        Parameters:
        x and y: a list of data representing input and output
        node: the starting node of the tree (can be a subset of another tree)
        """
        # If all data points have the same label, return a leaf with that label
        if _all_same_value(y):
            return node

        # Elif all data points have identical feature values, return a leaf with the most common label

        # Else create decision tree
        # - Choose a feature with the most infomation gain
        # - Split the data based on the feature's value and add a branch for each subset of data
        # - For each branch, call the algorithm recursively on the data points for that specific branch

    def learn(self, X, y, impurity_measure='entropy'):
        """
        Create a decision tree based on input data and an impurity measure

        Parameters:
        - X: a data matrix of continuous features
        - y: a label vector of categorical variables
        - impurity_measure: determines how to split the branches
        """
        pass

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
    def _all_same_value(self, input_list):
        """
        Returns true if the input list has all of the same value
        """
        return len(set(input_list)) == 1

    def print_subtree(self, node=self.root, depth=0):
        """
        Prints the subtree starting at the specified
        """
        if node:
            print(f"Depth: {node.depth} Data: {node.data}")
            self.print_subtree(node.left, depth + 1)
            self.print_subtree(node.right, depth + 1)

# Main
if __name__ == "__main__":
    print("INF264 Project 1")