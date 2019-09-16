import numpy as np

print("CS 691 - Project 1 - Decision Trees")
print("Lee Easson")
print("")

def DT_train_binary(X, Y, max_depth):
    # Check if X any Y have same number of rows
    # Return 0 if unequal row number
    if X.shape[0] != Y.shape[0]:
        return 0

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
X = np.array([ [0,1,0,1], [1,1,1,1], [0,0,0,1] ])
Y = np.array([ [1], [1], [0] ])
max_depth = 2

# Test DT_train_binary() and DT_test_binary()
DT = DT_train_binary(X, Y, max_depth)
test_acc = DT_test_binary(X, Y, DT)