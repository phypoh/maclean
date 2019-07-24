#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:52:10 2019

@author: stage_pphyllis
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from ex3_functions import *

import scipy.optimize as opt

"""
-------------------------------------------------------------------------------
Hyperparameters
-------------------------------------------------------------------------------
"""

eta = 0.00001
cycles = 2000



data = pd.read_csv('ex2data1.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# =============================================================================
# """
# -------------------------------------------------------------------------------
# Visualising Raw Data
# -------------------------------------------------------------------------------
# """
# 
# positive = data[data['Admitted'].isin([1])]
# negative = data[data['Admitted'].isin([0])]
# 
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# =============================================================================


"""
-------------------------------------------------------------------------------
Preprocessing Data
-------------------------------------------------------------------------------
"""


# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.matrix(np.array(X.values))
y = np.matrix(np.array(y.values))



"""
-------------------------------------------------------------------------------
Optimise using SciPy
-------------------------------------------------------------------------------
"""

theta = np.zeros(3)
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

theta_min = np.matrix(result[0])
predictions = predict(theta_min.T, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

plot_predictive_distribution(X[:, 1:], y, theta_min)



