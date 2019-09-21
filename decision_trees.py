import numpy as np
import math

print("CS 691 - Project 1 - Decision Trees")
print("Lee Easson")
print("Kurtis Rodrigue")
print()

class Node:
    def __init__(self, value, IG):
        self.left = None
        self.right = None
        self.value = value
        self.IG = IG

class Binary_Tree:
    def __init__(self):
        self.root = None
        self.depth = 0

    def add(self, value, IG):
        if self.root == None:
            self.root = Node(value, IG)
        else:
            self.add_node(value, IG, self.root)

    def add_node(self, value, IG, node):
        if value < node.value:
            if node.left is not None:
                self.add_node(value, IG, node.left)
            else:
                node.left = Node(value, IG)
        else:
            if node.right is not None:
                self.add_node(value, IG, node.right)
            else:
                node.right = Node(value, IG)

    def print_tree(self):
        if self.root is not None:
            self.print_nodes(self.root)

    def print_nodes(self, node):
        if node is not None:
            self.print_nodes(node.left)
            print(str(node.value) + ' ' + str(node.IG) + ' ')
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
    H_0 = 0 
    H_1 = 0
    num_0 = 0
    num_1 = 0
    prop_0 = 0
    prop_1 = 0 
    
    if binary_value == 0:
        # Calculate H(Feature == 0)
        No = np.where(feature == 0)[0]

        for i in No:
            if feature[i] != labels[i]: num_0 += 1
            else: num_1 += 1

        if (len(No) == 0): return H_0
        else:    
            prop_0 = num_0 / len(No)
            prop_1 = num_1 / len(No)

        if prop_0 == 1 or prop_1 == 1: H_0 = 0     
        else: H_0 = -(prop_0 * math.log2(prop_0)) + -(prop_1 * math.log2(prop_1))

        return H_0

    else:
        # Calculate H(Feature == 1)
        Yes = np.where(feature == 1)[0]

        for i in Yes:
            if feature[i] != labels[i]: num_0 += 1
            else: num_1 += 1

        if (len(Yes) == 0): return H_1
        else:
            prop_0 = num_0 / len(Yes)
            prop_1 = num_1 / len(Yes)

        if prop_0 == 1 or prop_1 == 1: H_1 = 0
        else: H_1 = -(prop_0 * math.log2(prop_0)) + -(prop_1 * math.log2(prop_1))

        return H_1


def DT_train_binary(X, Y, max_depth):
    # Number of samples and features in training data
    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Check if X any Y have same number of samples
    # Return 0 if unequal sample number
    if X.shape[0] != Y.shape[0]: return 0

    # Create new Binary Tree for DT
    DT = Binary_Tree()

    # Calculate entropy of entire dataset
    H = calculate_entropy(Y)

    # Create numpy array for Labels
    labels = np.zeros(shape=(num_samples))
    for sample in range(len(Y)): labels[sample] = Y[sample]

    # GREEDY ALGORITHM

    # Initialize dictionary for storing IG 
    # of each feature and current index of feature
    Feature_IG = dict.fromkeys(list(range(1, num_features+1, 1)))
    current_index = 1

    # Iterate through features and calculate IG for each
    for feature in range(num_features):
        # Create empty numpy array for storing
        # the samples of the current feature
        current_feature = np.zeros(shape=(num_samples))

        # Iterate through the sample in the current feature
        # adding each sample to current feature numpy array
        for sample in range(num_samples): current_feature[sample] = X[sample][feature]

        # Calculate entropy of the split
        # where feature is 0 and 1 respectively
        H_0 = calculate_feature_entropy(current_feature, labels, 0)
        H_1 = calculate_feature_entropy(current_feature, labels, 1)

        # Calculate proportion of 0s and 1s in feature
        prop_features_0 = int(np.count_nonzero(current_feature == 0)) / num_samples
        prop_features_1 = int(np.count_nonzero(current_feature == 1)) / num_samples

        # Calculate IG for current feature
        IG = H - ( (prop_features_0 * H_0) + (prop_features_1 * H_1) )

        # Update feature IG dictionary and current index
        Feature_IG[current_index] = IG
        current_index += 1

    print(Feature_IG)

    # Find feature with maximum IG
    Max_IG = max(Feature_IG.values())
    Max_feature = [k for k,v in Feature_IG.items() if v == Max_IG]

    # Add feature with maximum IG to DT
    DT.add("F"+str(Max_feature), Max_IG)

    # If IG = 0, stop
    # else, continue

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
Test 1
F1 F2 F3 F4  L
0  1  0  1 | 1
1  1  1  1 | 1
0  0  0  1 | 0

Training Set 1
F1 F2   L
0  1  | 1
0  0  | 0
1  0  | 0
0  0  | 0
1  1  | 1

Training Set 2
F1 F2 F3 F4  L
0  1  0  0 | 0
0  0  0  1 | 1
1  0  0  0 | 0
0  0  1  1 | 0
1  1  0  1 | 1
1  1  0  0 | 0
1  0  0  1 | 1
0  1  0  1 | 1
0  1  0  0 | 1

Trainin Set Example
    F1 F2 F3 F4  L
1   1  1  0  0 | 0
2   1  1  1  1 | 1
3   1  1  1  1 | 1
4   0  0  0  1 | 0
5   0  0  1  1 | 0
6   0  0  1  0 | 1
7   0  0  0  0 | 0
8   1  0  1  0 | 0
9   1  1  1  0 | 1
10  0  0  1  1 | 0
'''
training_features_1 = np.array([ [0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1] ])
training_labels_1 = np.array([ [1], [1], [0] ])

training_features_2 = np.array([ [0,1], [0,0], [1,0], [0,0], [1,1] ])
training_labels_2 = np.array([ [1], [0], [0], [0], [1] ])

training_features_3 = np.array([ [0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0] ])
training_labels_3 = np.array([ [0], [1], [0], [0], [1], [0], [1], [1], [1] ])


X = np.array([ [1,1,0,0], [1,1,1,1], [1,1,1,1], [0,0,0,1], [0,0,1,1], [0,0,1,0], [0,0,0,0], [1,0,1,0], [1,1,1,0], [0,0,1,1] ])
Y = np.array([ [0], [1], [1], [0], [0], [1], [0], [0], [1], [0] ])
max_depth = 2

# Test DT_train_binary() and DT_test_binary()
DT = DT_train_binary(X, Y, max_depth)
DT.print_tree()
test_acc = DT_test_binary(training_features_1, training_labels_1, DT)