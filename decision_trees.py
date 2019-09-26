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

class Real_Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = 0
        self.feature = 0
        self.depth = 0
        self.left_prediction = 0
        self.right_prediction = 1

class Real_Binary_Tree:
    def __init__(self):
        self.root = None

    def print_tree(self):
        if self.root is not None:
            self.print_nodes(self.root)

    def print_nodes(self, node):
        if node is not None:
            self.print_nodes(node.left)
            print(str(node.value) + ' ' + str(node.feature) + ' ' + str(node.depth) + ' ' + str(node.left_prediction) + ' ' + str(node.right_prediction))
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

    H0 = 0
    H1 = 0

    if prop_labels_0 > 0:
        H0 = -1 * prop_labels_0 * math.log2(prop_labels_0)

    if prop_labels_1 > 0:
        H1 = -1 * prop_labels_1 * math.log2(prop_labels_1)
    
    # Calculate entropy
    entropy = H0 + H1
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
            if feature[i] != labels[i]:
                num_0 += 1
            else:
                num_1 += 1

        if (len(No) == 0):
            return H_0
        else:
            prop_0 = num_0 / len(No)
            prop_1 = num_1 / len(No)

        if prop_0 == 1 or prop_1 == 1:
            H_0 = 0
        else:
            H_0 = -(prop_0 * math.log2(prop_0)) + -(prop_1 * math.log2(prop_1))

        return H_0

    else:
        # Calculate H(Feature == 1)
        Yes = np.where(feature == 1)[0]

        for i in Yes:
            if feature[i] != labels[i]:
                num_0 += 1
            else:
                num_1 += 1

        if (len(Yes) == 0):
            return H_1
        else:
            prop_0 = num_0 / len(Yes)
            prop_1 = num_1 / len(Yes)

        if prop_0 == 1 or prop_1 == 1:
            H_1 = 0
        else:
            H_1 = -(prop_0 * math.log2(prop_0)) + -(prop_1 * math.log2(prop_1))

        return H_1

def build_tree(X, DT, Feature_IG, labels, num_samples):
    # Find feature with maximum IG
    Max_IG = max(Feature_IG.values())
    Max_feature = [k for k, v in Feature_IG.items() if v == Max_IG]

    # Add feature with maximum IG to DT
    DT.add("F" + str(Max_feature), Max_IG)

    # Calculate entopy of maximum IG feature
    feature = np.zeros(shape=(num_samples))
    for sample in range(num_samples): feature[sample] = X[sample][int(Max_feature[0]) - 1]

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
    Feature_IG = dict.fromkeys(list(range(1, num_features + 1, 1)))
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
        IG = H - ((prop_features_0 * H_0) + (prop_features_1 * H_1))

        # Update feature IG dictionary and current index
        Feature_IG[current_index] = IG
        current_index += 1

    # Build tree
    build_tree(X, DT, Feature_IG, labels, num_samples)

    # Return DT
    return DT

def DT_predict_recursive(X, node):
    if node.no is not None:
        DT_predict(X, node.no)
    else:
        return node.IG

    return

def DT_predict(X, DT):
    return DT_predict_recursive(X, DT)

def DT_test_binary(X, Y, DT):
    num_samples = X.shape[0]
    predictions = []
    for i in range(num_samples):
        predictions.append(X[i][0])
        #predictions.append(DT_predict(X[i], DT))

    diff = 0
    same = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            same += 1
        else:
            diff += 1

    accuracy = same / (same + diff)
    return accuracy

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



def DT_train_real(X, Y, max_depth):
    # Check if X any Y have same number of rows
    # Return 0 if unequal row number
    if X.shape[0] != Y.shape[0]:
        return 0

    # Create new Binary Tree for DT
    DT = Real_Binary_Tree()
    # Calculate entropy of entire dataset
    H = calculate_entropy(Y)
    # Split training data at each feature
    DT.root = Real_Node()
    DT.root.depth = 1
    Y = Y.flatten()
    samples = []
    for i in range(Y.shape[0]):
        samples.append(i)
    # Call recursive function
    DT_train_real_recursive(DT.root, samples, X, Y, max_depth, H)

    # Return DT as list of lists, numpy array, or class object
    return DT

def DT_train_real_recursive(node, sample_nums, X, Y, max_depth, H):
    if node.depth > max_depth:
        if max_depth != -1:
            return

    IG = []
    best_split = {}

    # Loop through features, calculate best IG
    for i in range(X.shape[1]):
        temp_IG = []

        for x in sample_nums:
            samples_0 = []
            samples_1 = []
            labels_0 = []
            labels_1 = []
            value = X[x][i]
            for y in sample_nums:
                if X[y][i] <= value:
                    samples_0.append(X[y][i])
                    labels_0.append(Y[y])
                else:
                    samples_1.append(X[y][i])
                    labels_1.append(Y[y])

            # Calculate probability on each split
            p_L = len(samples_0) / len(sample_nums)
            p_R = len(samples_1) / len(sample_nums)

            # Calculate entropy after split at each side
            l_entropy = 0
            if len(labels_0) > 0:
                labels_0 = np.array(labels_0)
                l_entropy = calculate_entropy(labels_0)

            r_entropy = 0
            if len(labels_1) > 0:
                labels_1 = np.array(labels_1)
                r_entropy = calculate_entropy(labels_1)

            temp_IG.append(H - (p_L * l_entropy) - (p_R * r_entropy))

        best = 0
        for z in range(len(temp_IG)):
            if temp_IG[z] > temp_IG[best]:
                best = z
        # dict feature# -> best sample to split on
        best_split[i] = best
        IG.append(temp_IG[best])


    # Choose best feature to split on based off IG
    feature = 0
    for i in range(len(IG)):
        if IG[i] > IG[feature]:
            feature = i

    # name feature for node
    node.feature = feature
    value = X[sample_nums[best_split[feature]]][feature]
    node.value = value
    left_samples = []
    right_samples = []
    right_labels = []
    left_labels = []
    for i in sample_nums:
        if X[i][feature] <= value:
            left_samples.append(i)
            left_labels.append(Y[i])
        else:
            right_samples.append(i)
            right_labels.append(Y[i])

    # Create new nodes and give them the appropriate samples and depth
    test = 0
    if len(left_labels) >= 1:
        test = left_labels[0]
        same = True
        for i in range(len(left_labels)):
            if left_labels[i] != test:
                same = False
        if same:
            node.left_prediction = left_labels[0]
        else:
            node.left = Real_Node()
            node.left.depth = node.depth + 1
            DT_train_real_recursive(node.left, left_samples, X, Y, max_depth, H)

    if len(right_labels) >= 1:
        test = right_labels[0]
        same = True
        for i in range(len(right_labels)):
            if right_labels[i] != test:
                same = False
        if same:
            node.right_prediction = right_labels[0]
        else:
            node.right = Real_Node()
            node.right.depth = node.depth + 1
            DT_train_real_recursive(node.right, right_samples, X, Y, max_depth, H)

    if len(right_samples) == 0:
        node.right_prediction = not node.left_prediction
    elif len(left_samples) == 0:
        node.left_prediction = not node.right_prediction
    return

def DT_predict_real_recursive(X, node):
    if X[node.feature] <= node.value:
        if node.left:
            return DT_predict_real_recursive(X, node.left)
        else:
            return node.left_prediction
    else:
        if node.right:
            return DT_predict_real_recursive(X, node.right)
        else:
            return node.right_prediction

def DT_predict_real(X, DT):
    return DT_predict_real_recursive(X, DT.root)

def DT_test_real(X, Y, DT):
    predictions = []
    for i in range(X.shape[0]):
        predictions.append(DT_predict_real(X[i], DT))

    diff = 0
    same = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            same += 1
        else:
            diff += 1

    accuracy = same / (same + diff)
    return accuracy

def DT_train_real_best(X_train, Y_train, X_val, Y_val):
    DTs = []
    i = 0
    # make unique trees, iterative depth
    while True:
        # train a new tree of depth i
        DTs.append(DT_train_real(X_train, Y_train, i))
        i += 1
        # test accuracy of tree
        if DT_test_real(X_val, Y_val, DTs[i-1]) == 1:
            return DTs[i-1]
        # if the accuracy does not change with an increase in depth, return the tree
        if len(DTs) > 2:
            if DT_test_real(X_val, Y_val, DTs[i-1]) == DT_test_real(X_val, Y_val, DTs[i-2]):
                return DTs[i-1]