#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:00:44 2019

@author: stage_pphyllis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from ex4_functions import *

eta = 1


data = loadmat('ex3data1.mat')

# =============================================================================
# # To visualise a random picture
# visualise_rand(data)
# =============================================================================

all_theta = one_vs_all(data['X'], data['y'], 10, eta)

y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))