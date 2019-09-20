import numpy as np
import math

print("CS 691 - Project 1 - Decision Trees")
print("Lee Easson")
print("Kurtis Rodrigue")
print()

class Node:
    def __init__(self, value, depth):
        self.left = None
        self.right = None
        self.value = value
        self.depth = depth

class Binary_Tree:
    def __init__(self):
        self.root = None
        self.depth = 0

    def add(self, value):
        if self.root == None:
            self.root = Node(value, 0)
        else:
            self.add_node(value, self.root)

    def add_node(self, value, node):
        if value < node.value:
            if node.left is not None:
                self.depth += 1
                self.add_node(value, node.left)
            else:
                node.left = Node(value, self.depth)
        else:
            if node.right is not None:
                self.depth += 1
                self.add_node(value, node.right)
            else:
                node.right = Node(value, self.depth)

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
    entropy = (-1 * prop_labels_0 * math.log2(prop_labels_0)) \
            + (-1 * prop_labels_1 * math.log2(prop_labels_1))
    return entropy

def calculate_feature_entropy(feature, labels, binary_value):
    # Count of 0s and 1s respectively
    num_0 = 0
    num_1 = 0
    # Proportion of 0s and 1s respectively
    prop_0 = 0
    prop_1 = 1

    if binary_value == 0:
        # Calculate H(Feature == 0)
        H_0 = 0  

        No = np.where(feature == 0)[0]
        for i in No:
            if feature[i] == labels[i]: num_0 += 1
            else: num_1 += 1
        if (len(No) == 0): return H_0
        else:    
            prop_0 = num_0 / len(No) 
            prop_1 = num_1 / len(No)

        if prop_0 == 1 or prop_1 == 1:
            H_0 = 0     
        else: H_0 = -(prop_0 * math.log2(prop_0)) + -(prop_1 * math.log2(prop_1))

        return H_0

    else:
        # Calculate H(Feature == 1)
        H_1 = 0

        Yes = np.where(feature == 1)[0]
        for i in Yes:
            if feature[i] == labels[i]: num_0 += 1
            else: num_1 += 1
        if (len(Yes) == 0): return H_1
        else:
            prop_0 = num_0 / len(Yes) 
            prop_1 = num_1 / len(Yes)

        if prop_0 == 1 or prop_1 == 1:
            H_1 = 0
        else: H_1 = -(prop_0 * math.log(prop_0)) + -(prop_1 * math.log(prop_1))

        return H_1

def DT_train_binary(X, Y, max_depth):
    # Number of samples in training data
    num_samples = X.shape[0]
    # Number of features in training data
    num_features = X.shape[1]

    # Check if X any Y have same number of samples
    # Return 0 if unequal sample number
    if X.shape[0] != Y.shape[0]:
        return 0

    # Create new Binary Tree for DT
    DT = Binary_Tree()

    # Calculate entropy of entire dataset
    H = calculate_entropy(Y)

    # Create numpy array for Labels
    labels = np.zeros(shape=(num_samples))
    for sample in Y:
        labels[sample] = Y[sample]

    # Greedy algorithm
    for feature in range(num_features):
        # Create empty numpy array for storing
        # the samples of the current feature
        current_feature = np.zeros(shape=(num_samples))

        # Iterate through the sample in the current feature
        # adding each sample to current feature numpy array
        for sample in range(num_samples):
            current_feature[sample] = X[sample][feature]
        print(current_feature)

        # Calculate entropy of the split
        # where feature is 0
        H_0 = calculate_feature_entropy(current_feature, labels, 0)
        print("H_0:",H_0)
        # where feature is 1
        H_1 = calculate_feature_entropy(current_feature, labels, 1)
        print("H_1:",H_1)

        # Calculate proportion of 0s and 1s in labels
        prop_labels_0 = int(np.count_nonzero(labels == 0)) / num_samples
        print("Prop_0:",prop_labels_0)
        prop_labels_1 = int(np.count_nonzero(labels == 1)) / num_samples
        print("Prop_1:",prop_labels_1)

        # Calculate IG for current feature
        IG = H - ( (prop_labels_0 * H_0) + (prop_labels_1 * H_1) )
        print("IG:",IG)
        print()

    # Find maximum IG
        
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