#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:33:08 2019

@author: phypoh
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_data_internal(X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()

##
# x: input to the logistic function
#
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# w: current parameter values
#
    
def log_sigmoid(X, w):
    dot_prod = np.dot(X, w)
    output = np.zeros(dot_prod.size)
    for i in range(len(dot_prod)):
        if dot_prod[i] < -10**100:
            output[i] = dot_prod[i]
            print("estimate")
        else:
            output[i] = np.log(logistic(dot_prod[i]))
    return output

def neg_log_sigmoid(X, w):
    dot_prod = -np.dot(X, w)
    output = np.zeros(dot_prod.size)
    for i in range(len(dot_prod)):
        if dot_prod[i] > 10**100:
            output[i] = dot_prod[i]
            print("estimate")
        else:
            output[i] = np.log(logistic(dot_prod[i]))
    return output

def compute_average_ll(X, y, w):
    return np.mean(y * log_sigmoid(X, w)
                   + (1 - y) * neg_log_sigmoid(X, w))

# =============================================================================
#     
# def compute_average_ll(X, y, w):
#     output_prob = logistic(np.dot(X, w))
#     return np.mean(y * np.log(output_prob)
#                    + (1 - y) * np.log(1.0 - output_prob))
# =============================================================================

##
# ll: 1d array with the average likelihood per data point, for each training
# step. The dimension of this array should be equal to the number of training
# steps.
#

def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()
    
    
def plot_ll2(ll1, ll2):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll1) + 2)
    plt.ylim(min(ll1) - 0.1, max(ll1) + 0.1)
    ax.plot(np.arange(1, len(ll1) + 1), ll1, 'r-', label = "Training")
    ax.plot(np.arange(1, len(ll2) + 1), ll2, 'b-', label = "Test")
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()

##
# X: 2d array with input features at which to compute predictions.
#
# (uses parameter vector w which is defined outside the function's scope)
#

def predict_for_plot(x, beta):
    x_tilde = np.concatenate((np.ones((x.shape[ 0 ], 1 )), x), 1)
    return logistic(np.dot(x_tilde, beta))

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# predict: function that recives as input a feature matrix and returns a 1d
#          vector with the probability of class 1.

def plot_predictive_distribution(X, y, beta):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_predict = np.concatenate((xx.ravel().reshape((-1, 1)),
                                yy.ravel().reshape((-1, 1))), 1)
    Z = predict_for_plot(X_predict, beta)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()

##
# l: hyper-parameter for the width of the Gaussian basis functions
# Z: location of the Gaussian basis functions - training points
# X: points at which to evaluate the basis functions - training or test points

def expand_inputs(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)

##
# x: 2d array with input features at which to compute the predictions
# using the feature expansion
#
# (uses parameter vector w and the 2d array X with the centers of the basis
# functions for the feature expansion, which are defined outside the function's
# scope)
#

def predict_for_plot_expanded_features(l, x, X, w):
    x = np.concatenate((np.ones((x.shape[ 0 ], 1 )), x), 1)
    x_expanded = expand_inputs(l, x, X)
    #print(x.shape, X.shape, x_expanded.shape)
    
    x_tilde = np.concatenate((np.ones((x_expanded.shape[ 0 ], 1 )),x_expanded), 1)
    return logistic(np.dot(x_tilde, w))

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# predict: function that recives as input a feature matrix and returns a 1d
#          vector with the probability of class 1.

def plot_predictive_distribution_expanded(X_train, X_test, y, beta, l):
    xx, yy = plot_data_internal(X_test, y)
    ax = plt.gca()
    X_predict = np.concatenate((xx.ravel().reshape((-1, 1)),
                                yy.ravel().reshape((-1, 1))), 1)
    Z = predict_for_plot_expanded_features(l, X_predict, X_train, beta)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2, vmin=0, vmax=1)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()

