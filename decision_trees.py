import numpy as np
import math

print("CS 691 - Project 1 - Decision Trees")
print("Lee Easson")
print("")

def calculate_entropy(Y):
    # Set of all labels
    labels = (0,1)
    # Length of training labels
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

def calculate_information_gain(X, Y):
    # Calculate entropy of entire dataset
    H = calculate_entropy(Y)
    print(H)
    return 0

def DT_train_binary(X, Y, max_depth):
    # Check if X any Y have same number of rows
    # Return 0 if unequal row number
    if X.shape[0] != Y.shape[0]:
        return 0
    # If max_depth is -1, create most accurate tree
    if (max_depth == -1):
        return
    # Otherwise, create tree with max_depth
    

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
#DT = DT_train_binary(training_features, training_labels, max_depth)
#test_acc = DT_test_binary(training_features, training_labels, DT)

calculate_information_gain(training_features, training_labels)