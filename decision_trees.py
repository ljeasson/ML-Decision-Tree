import numpy as np
import math

print("CS 691 - Project 1 - Decision Trees")
print("Lee Easson")
print("Kurtis Rodrigue")
print()

class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

class Binary_Tree:
    def __init__(self):
        self.root = None

    def add(self, value):
        if self.root == None:
            self.root = Node(value)
        else:
            self.add_node(value, self.root)

    def add_node(self, value, node):
        if value < node.value:
            if node.left is not None:
                self.add_node(value, node.left)
            else:
                node.left = Node(value)
        else:
            if node.right is not None:
                self.add_node(value, node.right)
            else:
                node.right = Node(value)

    def print_tree(self):
        if self.root is not None:
            self.print_nodes(self.root)

    def print_nodes(self, node):
        if node is not None:
            self.print_nodes(node.left)
            print(str(node.value) + ' ')
            self.print_nodes(node.right)

def calculate_entropy(Y):
    # Set of all labels
    labels = (0, 1)
    # Number of training labels
    length = Y.shape[0]

    # Calculate number of each label
    num_labels_0 = np.count_nonzero(Y == labels[0])
    num_labels_1 = np.count_nonzero(Y == labels[1])
    
    # Calculate proportion of each label
    prop_labels_0 = num_labels_0 / length
    prop_labels_1 = num_labels_1 / length
    
    # Calculate entropy
    entropy = (-1 * prop_labels_0 * math.log(prop_labels_0)) \
            + (-1 * prop_labels_1 * math.log(prop_labels_1))
    return entropy

def DT_train_binary(X, Y, max_depth):
    # Check if X any Y have same number of rows
    # Return 0 if unequal row number
    if X.shape[0] != Y.shape[0]:
        return 0

    # Create new Binary Tree for DT
    DT = Binary_Tree()

    # Calculate entropy of entire dataset
    H = calculate_entropy(Y)

    # Split training data at each feature
    for row in X:
        print(row[0])

        # Calculate Information Gain at each split
        
        # Choose split based on maximum IG
        
    # Return DT as list of lists, numpy array, or class object
    return DT

def DT_test_binary(X, Y, DT):
    return 0


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
    return

def DT_train_real(X, Y, max_depth):
    return

def DT_test_real(X, Y, DT):
    return

def DT_train_real_best(X_train, Y_train, X_val, Y_val):
    return

# Test Data
'''
F1 F2 F3 F4  L
0  1  0  1 | 1
1  1  1  1 | 1
0  0  0  1 | 0
'''
training_features = np.array([ [0,1,0,1], [1,1,1,1], [0,0,0,1] ])
training_labels = np.array([ [1], [1], [0] ])
max_depth = 2

# Test DT_train_binary() and DT_test_binary()
DT = DT_train_binary(training_features, training_labels, max_depth)
test_acc = DT_test_binary(training_features, training_labels, DT)