#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:43:49 2019

@author: stage_pphyllis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex3_functions import *

import scipy.optimize as opt

"""
----------------------------------------------------------------------------
Hyperparameters
-------------------------------------------------------------------------------
"""

learningRate = 1



data2 = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Accepted'])


"""
----------------------------------------------------------------------------
Visualising Raw Data
-------------------------------------------------------------------------------
"""

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

"""
----------------------------------------------------------------------------
Fitting Data to Polynomial
-------------------------------------------------------------------------------
"""



cols = data2.shape[1]
pre_X2 = data2.iloc[:,1:cols]
pre_X2 = np.array(pre_X2.values)

data2.insert(3, 'Ones', 1)

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']


for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

"""
----------------------------------------------------------------------------
Preprocesing Data
-------------------------------------------------------------------------------
"""

# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)

"""
----------------------------------------------------------------------------
Optimisation
-------------------------------------------------------------------------------
"""

result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))

theta_min = np.matrix(result2[0])
predictions = predict(theta_min.T, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

