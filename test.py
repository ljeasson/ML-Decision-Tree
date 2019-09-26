import numpy as np

# Binary Test Data
import decision_trees as dt
'''
Training Set Example
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

# Test DT_train_binary(X, Y, max_depth) and DT_test_binary(X, Y, DT)
X = np.array([ [1,1,0,0], [1,1,1,1], [1,1,1,1], [0,0,0,1], [0,0,1,1], [0,0,1,0], [0,0,0,0], [1,0,1,0], [1,1,1,0], [0,0,1,1] ])
Y = np.array([ [0], [1], [1], [0], [0], [1], [0], [0], [1], [0] ])
max_depth = 2

#DT = dt.DT_train_binary(X, Y, max_depth)
#test_acc = dt.DT_test_binary(X, Y, DT)


# Real Test Data
import decision_trees_real as dtr

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

# Test DT_train_real(X, Y, max_depth) and DT_test_real(X, Y, DT)
DTR = dtr.DT_train_real(real_training_features, real_training_labels, max_depth)
test_acc_real = dtr.DT_test_real(real_training_features, real_training_labels, DTR)