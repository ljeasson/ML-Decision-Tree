import numpy as np
import decision_trees as dt

#Training Set 1:
X_train = np.array([ [0,1], [0,0], [1,0], [0,0], [1,1] ])
Y_train = np.array([ [1], [0], [0], [0], [1] ])
#Validation Set 1:
X_val = np.array([ [0,0], [0,1], [1,0], [1,1] ])
Y_val = np.array([ [0], [1], [0], [1] ])
#Testing Set 1:
X = np.array([ [0,0], [0,1], [1,0], [1,1] ])
Y = np.array([ [1], [1], [0], [1] ])

DT = dt.DT_train_binary(X_train, Y_train, 2)
#DT.print_tree()
test_acc = dt.DT_test_binary(X, Y, DT)
print(test_acc)

DT = dt.DT_train_binary_best(X_train, Y_train, X_val, Y_val)
#DT.print_tree()
test_acc = dt.DT_test_binary(X, Y, DT)
print(test_acc)

print()

#Training Set 2:
X_train = np.array([ [0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0] ])
Y_train = np.array([ [0], [1], [0], [0], [1], [0], [1], [1], [1] ])
#Validation Set 1:
X_val = np.array([ [1,0,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,0], [1,0,0,1], [0,1,0,0] ])
Y_val = np.array([ [0], [0], [1], [0], [1], [1] ])
#Testing Set 1:
X = np.array([ [0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0] ])
Y = np.array([ [1], [1], [0], [0], [1], [0], [1], [1], [1] ])

DT = dt.DT_train_binary(X_train, Y_train, 2)
#DT.print_tree()
test_acc = dt.DT_test_binary(X, Y, DT)
print(test_acc)

DT = dt.DT_train_binary_best(X_train, Y_train, X_val, Y_val)
#DT.print_tree()
test_acc = dt.DT_test_binary(X, Y, DT)
print(test_acc)


# Real Test Data
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
DTR = dt.DT_train_real(real_training_features, real_training_labels, 4)
DTR.print_tree()
test_acc_real = dt.DT_test_real(real_training_features, real_training_labels, DTR)

DTR2 = dt.DT_train_real_best(real_training_features, real_training_labels, real_training_features, real_training_labels)
test_acc_real = dt.DT_test_real(real_training_features, real_training_labels, DTR2)
DTR2.print_tree()

