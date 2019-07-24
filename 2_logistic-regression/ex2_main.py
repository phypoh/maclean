import numpy as np
import matplotlib.pyplot as plt
from ex2_functions import *
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

#-------------------------------------------------------------------------
#Initialisation
#-------------------------------------------------------------------------

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

X_train = []
y_train = []

X_test = []
y_test = []

num_of_runs = 100
beta = np.array([0,0,0])
eta = 0.001

ll_avg_train = []
ll_avg_test = []

#plot_data(X, y)

#-------------------------------------------------------------------------
# Split data into train and test sets
#-------------------------------------------------------------------------
for i in range(X.shape[0]):
    if i%5 == 0:        
        X_test.append(X[i])
        y_test.append(y[i])
    else:
        X_train.append(X[i])
        y_train.append(y[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
        
X_test = np.array(X_test)
y_test = np.array(y_test)


x_train = np.concatenate((np.ones((X_train.shape[0], 1 )), X_train), 1)
x_test = np.concatenate((np.ones((X_test.shape[0], 1 )), X_test), 1)

#-------------------------------------------------------------------------
#Training
#-------------------------------------------------------------------------

for run in range(num_of_runs):
    y_mod = []
    for i in range(len(y_train)):
        y_mod.append(y_train[i] - logistic(np.dot(beta,x_train[i])))
    
    y_mod = np.array(y_mod)
    
    beta = beta + eta*np.matmul(x_train.transpose(),y_mod)
        
    ll_avg_train.append(compute_average_ll(x_train, y_train, beta))
    ll_avg_test.append(compute_average_ll(x_test, y_test, beta))


#print(beta)

#-------------------------------------------------------------------------
#Plot Log-likelihood mean and predictive distribution
#-------------------------------------------------------------------------

plot_ll2(ll_avg_train,ll_avg_test)

plot_predictive_distribution(X_train, y_train, beta)

plot_predictive_distribution(X_test, y_test, beta)


#-------------------------------------------------------------------------
#Predictions
#-------------------------------------------------------------------------


ll_train = logistic(np.dot(x_train, beta))

x_test = np.concatenate((np.ones((X_test.shape[ 0 ], 1 )), X_test), 1)

ll_test = logistic(np.dot(x_test, beta))

#print(ll_train)
#print(ll_test)


test_predict = []

for i in ll_test:
    if i>0.5 and i <= 1:
        test_predict.append(1)
    elif i <= 0.5 and i >= 0:
        test_predict.append(0)
        
#print(test_predict)

false_pos, false_neg, true_pos, true_neg = 0, 0, 0, 0

for i in range(len(test_predict)):
    actual_class = y_test[i]
    predicted_class = test_predict[i]

    if actual_class == 1:
        if predicted_class == 1:
            true_pos += 1
        elif predicted_class == 0:
            false_neg += 1
    elif actual_class == 0:
        if predicted_class == 0:
            true_neg += 1
        elif predicted_class == 1:
            false_pos += 1
            
print(false_pos, false_neg, true_pos, true_neg)

C = confusion_matrix(y_test, test_predict)

print(C/C.astype(np.float).sum(axis=1))

    
    


