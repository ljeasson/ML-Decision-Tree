import numpy as np
import math

class Node:
    def __init__(self, value, IG):
        self.no = None
        self.yes = None
        self.value = value
        self.IG = IG

class Binary_Tree:
    def __init__(self):
        self.root = None

    def get_root(self):
        return self.root

    def add(self, value, IG):
        if self.root == None:
            self.root = Node(value, IG)
        else:
            self.add_node(value, IG, self.root)

    def add_node(self, value, IG, node):
        if value < node.value:
            if node.no is not None:
                self.add_node(value, IG, node.no)
            else:
                node.no = Node(value, IG)
        else:
            if node.yes is not None:
                self.add_node(value, IG, node.yes)
            else:
                node.yes = Node(value, IG)

    def print_tree(self):
        if self.root is not None:
            self.print_nodes(self.root)

    def print_nodes(self, node):
        if node is not None:
            self.print_nodes(node.no)
            print(str(node.value) + ' ' + str(node.IG))
            self.print_nodes(node.yes)

# Decision Tree - Binary Training
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

def build_tree(X, DT, Feature_IG, labels, num_samples):
    # Find feature with maximum IG
    Max_IG = max(Feature_IG.values())
    Max_feature = [k for k,v in Feature_IG.items() if v == Max_IG]

    # Add feature with maximum IG to DT
    DT.add("F"+str(Max_feature), Max_IG)

    # Calculate entopy of maximum IG feature
    feature = np.zeros(shape=(num_samples))
    for sample in range(num_samples): feature[sample] = X[sample][int(Max_feature[0])-1]
    
    H_0 = calculate_feature_entropy(feature, labels, 0)
    H_1 = calculate_feature_entropy(feature, labels, 1)
    DT.add("_", H_0)
    DT.add("_", H_1)

    # Remove max key, value pair
    del Feature_IG[Max_feature[0]]
    
    # Recursively build tree as long as Feature_IG dict
    # is not empty
    if len(Feature_IG) > 0:
        build_tree(X, DT, Feature_IG, labels, num_samples)

def DT_train_binary(X, Y, max_depth):
    # Check if X any Y have same number of samples
    # Return 0 if unequal sample number
    if X.shape[0] != Y.shape[0]: return 0
    
    # Number of samples and features in training data
    num_samples = X.shape[0]
    num_features = X.shape[1]

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

    # Build tree
    build_tree(X, DT, Feature_IG, labels, num_samples)

    # Return DT
    return DT


# Decision Tree - Binary Testing
def DT_predict_recursive(X, DT):
    '''
    if X[node.feature] <= node.value:
        if node.left:
            DT_predict(X, node.left)
        else:
            return node.left_prediction
    else:
        if node.right:
            DT_predict(X, node.right)
        else:
            return node.right_prediction
    '''
    return

def DT_predict(X, DT):
    return DT_predict_recursive(X, DT)

def DT_test_binary(X, Y, DT):
    num_samples = X.shape[0]
    predictions = []
    for i in range(num_samples):
        print(X[i])
        predictions.append(DT_predict(X[i], DT))
    
    diff = 0
    same = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            same += 1
        else:
            diff += 1

    accuracy = same / (same + diff)
    return accuracy


# Decision Tree - Binary Best
def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
    # Intialize list of decision trees and depth    
    DTs = []
    depth = 0

    # make unique trees, iterative depth
    while True:
        # Train binary tree with current depth
        DTs.append(DT_train_binary(X_train, Y_train, depth))
        depth += 1
        
        # Test accuracy of current tree
        if DT_test_binary(X_val, Y_val, DTs[depth-1]) == 1:
            return DTs[depth-1]

        # If the accuracy does not change 
        # return the tree
        if len(DTs) > 1:
            if DT_test_binary(X_val, Y_val, DTs[depth-1]) == DT_test_binary(X_val, Y_val, DTs[depth-2]):
                return DTs[depth-1]


# Decision Tree - Real Training

# Decision Tree - Real Testing

# Decision Tree - Real Test Best