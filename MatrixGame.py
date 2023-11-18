#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:21:13 2023

@author: sayantanchoudhury
"""

import numpy as np
from ProximalOperator import projection_simplex_sort
from model import MatrixGame
import matplotlib.pyplot as plt

### Iteration and communication
n_comm = 5000         # number of communication rounds

### Initialize paramters
n_node = 20          # number of nodes
n_row = 5            # dimension of x
n_col = 5            # dimension of y
n_feature = n_row + n_col       # dimension of data = dim(x) + dim(y)
n_data = 100         # size of dataset

### Initialize based on model
game = MatrixGame(n_row, n_col, n_node)
dF = game.grad             # computes operator
measure = game.dualgap     # measure duality gap

### Initialize hyperparameters
gamma = 1e-2       # stepsize \gamma
prob_comm = .1      # communication probability

### Initialize variables
x = np.random.normal(0, 1, (n_node, n_feature))
control, xhat, xdash = np.zeros((3, n_node, n_feature))    # local var, control var
ProxSkipVIPhist = []                                       # stores the distance to optimality
comm = 0                                                   # number of communication
x_sum = 0                                                  # stores the sum of x from all previous communications

while comm < n_comm:
    theta = np.random.binomial(1, prob_comm)           # generate \theta from bernoulli(prob_comm)
    if theta:
        comm += 1                                      # for \theta = 1, take a communication step
        for node in range(n_node):
            xhat[node] = x[node] - gamma * (dF(x[node], node) - control[node])      # compute local xhat for each node
            xdash[node] = xhat[node] - (gamma/ prob_comm) * control[node]
        x_comm = np.mean(xdash, axis = 0)              # communication step: take mean of vectors from all nodes
        x_comm = np.concatenate((projection_simplex_sort(x_comm[:n_row], 1), projection_simplex_sort(x_comm[n_row:], 1)))    # project the parts of the vectors on the probability simplex
        for node in range(n_node):
            x[node] = x_comm
            control[node] += (prob_comm/ gamma) * (x[node] - xhat[node])            # update the local control variates
        x_sum += x_comm
        ProxSkipVIPhist.append(measure(x_sum/comm))                       # store the duality gap for average iterate
        # ProxSkipVIPhist.append(measure(x_comm))                             # store the duality gap for last iterate
    else:
        for node in range(n_node):
            x[node] -= gamma * (dF(x[node], node) - control[node])                  # update local variable with gradient step and keep control variate same
ProxSkipVIPhist = np.array(ProxSkipVIPhist)

"""
Implement Local GDA.
"""

### Initialize hyperparameters
local_step = 10                                    # local steps
gamma = 1e-2

### Initialize variables
x = np.random.normal(0, 1, (n_node, n_feature))                               # local var
global_x = np.random.normal(0, 1, n_feature)                                  # global var
LocalGDAhist = []                                                   # stores the distance to optimality
comm = 0                                                            # number of communication
global_xsum = 0

while comm < n_comm:
    for node in range(n_node):
        x[node] = global_x
        for step in range(local_step):
            x[node] -= gamma * dF(x[node], node)
    comm += 1                                                        # update communication
    global_x = np.mean(x, axis = 0)
    global_x = np.concatenate((projection_simplex_sort(global_x[:n_row], 1), projection_simplex_sort(global_x[n_row:], 1)))    # project the parts of the vectors on the probability simplex
    global_xsum += global_x
    LocalGDAhist.append(measure(global_xsum/comm))                          # store distance to optimality for global variable 
LocalGDAhist = np.array(LocalGDAhist)                                                       

"""
Implement Local EG.
"""

### Initialize Hyperparameters
local_step = 10
gamma1 = 1e-2
gamma2 = 1e-2

### Initialize variables
x = np.random.normal(0, 1, (n_node, n_feature))                               # local var
global_x = np.random.normal(0, 1, n_feature)                                  # global var
LocalEGhist = []                                                 # stores the distance to optimality
comm = 0

while comm < n_comm:
    for node in range(n_node):
        x[node] = global_x
        for _ in range(local_step):
            temp = x[node] - gamma1 * dF(x[node], node)
            x[node] -= gamma2 * dF(temp, node)
    global_x = np.mean(x, axis = 0)
    global_x = np.concatenate((projection_simplex_sort(global_x[:n_row], 1), projection_simplex_sort(global_x[n_row:], 1)))    # project the parts of the vectors on the probability simplex
    LocalEGhist.append(measure(global_x))                        # store distance to optimality for global variable
    comm += 1         
LocalEGhist = np.array(LocalEGhist)

### Plot the Trajectories
marker = np.append(np.arange(0, n_comm, n_comm/10, dtype='int'), n_comm - 1)
plt.plot(marker, LocalEGhist[marker], '-r>', label = 'Local EG')
plt.plot(marker, LocalGDAhist[marker], '-bo', label = 'Local GDA')
plt.plot(marker, ProxSkipVIPhist[marker], '-gd', label = 'ProxSkip-GDA-FL')


### Plot Formatting
plt.grid(True)
plt.yscale('log')
plt.ylabel(r'Duality Gap')
plt.xlabel('Communication Rounds')
plt.title('Policemen Burglar Game')
plt.legend()



