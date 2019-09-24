import numpy as np
import math

print("CS 691 - Project 1 - Decision Trees")
print("Lee Easson")
print("Kurtis Rodrigue")
print()

class Real_Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = 0
        self.feature = 0
        self.depth = 0
        self.left_prediction = False
        self.right_prediction = True


class Real_Binary_Tree:
    def __init__(self):
        self.root = None

    def print_tree(self):
        if self.root is not None:
            self.print_nodes(self.root)

    def print_nodes(self, node):
        if node is not None:
            self.print_nodes(node.left)
            print(str(node.value) + ' ' + str(node.feature))
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

    #print('prop_lables_1: {}'.format(prop_labels_1))
    #print('prop_lables_0: {}'.format(prop_labels_0))

    H0 = 0
    H1 = 0

    if prop_labels_0 > 0:
        H0 = -1 * prop_labels_0 * math.log2(prop_labels_0)

    if prop_labels_1 > 0:
        H1 = -1 * prop_labels_1 * math.log2(prop_labels_1)
    
    # Calculate entropy
    entropy = H0 + H1
    return entropy

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

            #print('p_r/l {}, {}'.format(p_L, p_R))

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
    #print('temp IG @ {}: {}'.format(i, IG))


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
            DT_train_real_recursive(node.right, left_samples, X, Y, max_depth, H)

    if len(right_samples) == 0:
        node.right_prediction = not node.left_prediction
    elif len(left_samples) == 0:
        node.left_prediction = not node.right_prediction
    return

def DT_test_real(X, Y, DT):
    predictions = []
    for i in range(X.shape[0]):
        predictions.append(DT_predict_real(X[i], DT.root))

    diff = 0
    same = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            same += 1
        else:
            diff += 1

    accuracy = same / (same + diff)
    return accuracy

def DT_predict_real(X, DT):
    return DT_predict_real_recursive(X, DT.root)

def DT_predict_real_recursive(X, node):
    if X[node.feature] <= node.value:
        if node.left:
            DT_predict_real(X, node.left)
        else:
            return node.left_prediction
    else:
        if node.right:
            DT_predict_real(X, node.right)
        else:
            return node.right_prediction
    return

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
        if len(DTs) > 1:
            if DT_test_real(X_val, Y_val, DTs[i-1]) == DT_test_real(X_val, Y_val, DTs[i-2]):
                return DTs[i-1]

# Test Data
'''
F1 F2 F3 F4  L
0  1  0  1 | 1
1  1  1  1 | 1
0  0  0  1 | 0
'''
training_features = np.array([ [0,1,0,1], [1,0,1,1], [0,0,0,1] ])
training_labels = np.array([ [1], [1], [0] ])
max_depth = 3

# Test DT_train_binary() and DT_test_binary()
#DT = DT_train_binary(training_features, training_labels, max_depth)
#test_acc = DT_test_binary(training_features, training_labels, DT)

#DT.print_tree()

#real test data
real_training_features = np.array([
    [4.8, 3.4, 1.9, 0.2],
     [5, 3, 1.6, 1.2],
     [5, 3.4, 1.6, 0.2],
     [5.2, 3.5, 1.5, 0.2],
     [5.2, 3.4, 1.4, 0.2],
     [4.7, 3.2, 1.6, 0.2],
     [4.8, 3.1, 1.6, 0.2],
     [5.4, 3.4, 1.5, 0.4],
     [7, 3.2, 4.7, 1.4],
     [6.4, 3.2, 4.7, 1.5],
     [6.9, 3.1, 4.9, 1.5],
     [5.5, 2.3, 4, 1.3],
     [6.5, 2.8, 4.6, 1.5],
     [5.7, 2.8, 4.5, 1.3],
     [6.3, 3.3, 4.7, 1.6],
     [4.9, 2.4, 3.3, 1]
])
real_training_labels = np.array([[1], [1], [1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [0], [0], [0], [0]])

DT2 = DT_train_real(real_training_features, real_training_labels, max_depth)

#DT2.print_tree()
