#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:18:49 2019

@author: stage_pphyllis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ex1_functions import *

""" 
-------------------------------------------------------------------------------
Initialise hyperparameters
-------------------------------------------------------------------------------
"""

eta = 0.0001
cycles = 5000


""" 
-------------------------------------------------------------------------------
Preprocessing Data
-------------------------------------------------------------------------------
"""
data = pd.read_csv("ex1data1.txt", header=None, names=["Population", "Profit"])

# Prints the first few data entries of the dataframe
# print(data.head(10))

# Prints statistical descriptions of the dataframe
# print(data.describe())

# data.plot(kind = 'scatter', x = 'Population', y = 'Profit')

# Add a column of ones to make matrix X
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# Initialising values
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([[0],[0]]))


""" 
-------------------------------------------------------------------------------
Gradient Descent
-------------------------------------------------------------------------------
"""

cost_col = []

for i in range(cycles):
    theta, cost = grad_desc(X, y, theta, eta)
    cost_col.append(cost)
    

plot_pred(X, y, theta, data)
plot_cost(cost_col)


""" 
-------------------------------------------------------------------------------
Testing Second Set of Data (Multivariate case)
-------------------------------------------------------------------------------
"""


eta = 0.0001
cycles = 1000

data2 = pd.read_csv("ex1data2.txt", header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std() # Normalising features
data2.insert(0, 'Ones', 1)

cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([[0],[0],[0]]))

cost_col2 = []

for i in range(cycles):
    theta2, cost2 = grad_desc(X2, y2, theta2, eta)
    cost_col2.append(cost2)
    
plot_cost(cost_col2)







