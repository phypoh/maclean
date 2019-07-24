#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:10:10 2019

@author: stage_pphyllis
"""

import numpy as np
import matplotlib.pyplot as plt

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

"""
Plotting
-------------------------------------------------------------------------------
"""

def plot_data_internal(X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    plt.figure(figsize=(12,8))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    ax = plt.gca()
    X = np.array(X)
    y = np.array(y).ravel()
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Admitted')
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'rx', label = 'Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc = 'upper right', scatterpoints = 1, numpoints = 1)
    return xx, yy
    

def predict_for_plot(x, beta):
    x_tilde = np.concatenate((np.ones((x.shape[ 0 ], 1 )), x), 1)
    return logistic(np.dot(x_tilde, beta))


def plot_predictive_distribution(X, y, beta):
    beta = np.array(beta).ravel()
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_predict = np.concatenate((xx.ravel().reshape((-1, 1)),
                                yy.ravel().reshape((-1, 1))), 1)
    Z = predict_for_plot(X_predict, beta)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()


"""
----------------------------------------------------------------------------
Logistic Regression
-------------------------------------------------------------------------------
"""

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(logistic(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - logistic(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = logistic(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

def predict(theta, X):
    probability = logistic(X * theta)
    return [1 if x >= 0.5 else 0 for x in probability]

"""
----------------------------------------------------------------------------
Regularised Logistic Regression
-------------------------------------------------------------------------------
"""

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(logistic(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - logistic(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = logistic(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    
    return grad
