#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:06:06 2019

@author: stage_pphyllis
"""
import numpy as np
import matplotlib.pyplot as plt


""" 
-------------------------------------------------------------------------------
Functions
-------------------------------------------------------------------------------
"""

def comp_cost(X, y, theta):
    mult = y - X*theta
    cost = mult.T * mult
    return float(cost)/len(X)/2

def comp_grad(X, y, theta):
    grad = 2 * (X.T * X * theta - X.T * y)
    return grad

def grad_desc(X, y, theta, eta):
    new_theta = theta - eta * comp_grad(X, y, theta)
    cost = comp_cost(X, y, new_theta)
    return new_theta, cost

def plot_pred(X, y, theta, data):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0, 0] + (theta[1, 0] * x)
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    return

def plot_cost(cost):
    fig, ax = plt.subplots()
    ax.plot(range(len(cost)), cost)
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost over Number of Iterations')