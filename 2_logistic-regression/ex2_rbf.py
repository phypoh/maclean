import numpy as np
# import matplotlib.pyplot as plt
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

num_of_runs_expanded = 600

l = 1
eta_expanded = 0.0001

ll_avg_train = []
ll_avg_test = []


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


x_train = np.concatenate((np.ones((X_train.shape[ 0 ], 1 )), X_train), 1)
x_test = np.concatenate((np.ones((X_test.shape[ 0 ], 1 )), X_test), 1)



#-------------------------------------------------------------------------
#RADIAL BASIS FUNCTION
#-------------------------------------------------------------------------



X_expanded = expand_inputs(l, X_train, X_train)

x_expanded = np.column_stack((np.ones(X_expanded.shape[0]),X_expanded))

beta_expanded = np.zeros(X_expanded.shape[0]+1)

ll_expanded = []

#-------------------------------------------------------------------------
#Training
#-------------------------------------------------------------------------

for run in range(num_of_runs_expanded):
    y_mod = []
    for i in range(len(y_train)):
        y_mod.append(y_train[i] - logistic(np.dot(beta_expanded,x_expanded[i])))
    
    y_mod = np.array(y_mod)
    
    beta_expanded = beta_expanded + eta_expanded*np.matmul(x_expanded.transpose(),y_mod)
        
    ll_avg_train.append(compute_average_ll(x_expanded, y_train, beta_expanded))
    
    X_test_expanded = expand_inputs(l, x_test, x_train)
    x_test_expanded = np.column_stack((np.ones(X_test_expanded.shape[0]),X_test_expanded))

    
    ll_avg_test.append(compute_average_ll(x_test_expanded, y_test, beta_expanded))

    
#plot_ll(ll_avg_train)
plot_ll2(ll_avg_train,ll_avg_test)

x_test = np.concatenate((np.ones((X_test.shape[0], 1 )), X_test), 1)

plot_predictive_distribution_expanded(x_train, X_train, y_train, beta_expanded, l)

plot_predictive_distribution_expanded(x_train, X_test, y_test, beta_expanded, l)


#-------------------------------------------------------------------------
#Confusion Matrix
#-------------------------------------------------------------------------

x_predict = predict_for_plot_expanded_features(l, X_test, x_train, beta_expanded)
    
test_predict = []

for i in x_predict:
    if i>0.5 and i <= 1:
        test_predict.append(1)
    elif i <= 0.5 and i >= 0:
        test_predict.append(0)
        

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




