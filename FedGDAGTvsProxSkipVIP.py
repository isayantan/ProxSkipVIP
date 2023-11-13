#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:55:28 2023

@author: sayantanchoudhury
"""

import numpy as np
import Quad
import matplotlib.pyplot as plt

### Iteration and communication
n_comm = 50          # number of communication rounds

### Initialize paramters
n_node = 16          # number of nodes
n_feature = 10       # dimension of data
n_data = 100         # size of dataset

### Initialize based on model
QuadraticGame = Quad.QuadGame(n_feature, n_node, n_data, method = 'mini batch', mini_batch=100)
dF = QuadraticGame.grad
measure = QuadraticGame.optdist             # distance of current update from x*

### ProxSkipVIP

### Initialize hyperparameters
gamma = 1e-1         # stepsize \gamma
prob_comm = .01      # communication probability

### Initialize variables
x, control, xhat = np.zeros((3, n_node, n_feature))    # local var, control var
ProxSkipVIPhist = []                                   # stores the distance to optimality
comm = 0                                               # number of communication

while comm < n_comm:
    theta = np.random.binomial(1, prob_comm)           # generate \theta from bernoulli(prob_comm)
    if theta:
        comm += 1                                      # for \theta = 1, take a communication step
        for node in range(n_node):
            xhat[node] = x[node] - gamma * (dF(x[node], node) - control[node])      # compute local xhat for each node
        for node in range(n_node):
            x[node] = np.mean(xhat, axis = 0)                                       # update local variable with the mean of all variable
            control[node] += (prob_comm/ gamma) * (x[node] - xhat[node])            # update the local control variates
        ProxSkipVIPhist.append(measure(np.mean(x, axis = 0)))                       # store the distance to optimality
    else:
        for node in range(n_node):
            x[node] -= gamma * (dF(x[node], node) - control[node])                  # update local variable with gradient step and keep control variate same
ProxSkipVIPhist = np.array(ProxSkipVIPhist)

### FedGDA-GT

### Initialize hyperparameters
gamma = 1e-1
local_step = int(np.ceil(1/prob_comm))

### Initialize variables
x = np.zeros((n_node, n_feature))                       # local var
global_x = np.zeros(n_feature)                          # global var
FedGDAGThist = []                                       # stores the distance to optimality
comm = 0                                                # number of communication

while comm < n_comm:
    comm += 1                                           # update number of communication
    global_grad = 0                                     # store global gradient computed at global variable
    for node in range(n_node):
        global_grad += dF(global_x, node)
    global_grad = global_grad/n_node
    
    for node in range(n_node):
        corr_term = dF(global_x, node) - global_grad    # store gradient-tracking correction term
        x[node] = global_x                              # initialize local steps with local variable = global variable
        for _ in range(local_step):
            x[node] -= gamma * (dF(x[node], node) - corr_term)  # update local variables with gradient tracking step
    global_x = np.mean(x, axis = 0)                     # communication step, update global variable
    FedGDAGThist.append(measure(global_x))              # store distance to optimality for global variable
FedGDAGThist = np.array(FedGDAGThist)

### Plot the Trajectories
marker = np.arange(0, n_comm, n_comm/10, dtype='int')
plt.plot(np.arange(n_comm), ProxSkipVIPhist, '-gd', markevery = marker, label = 'ProxSkipVIP')
plt.plot(np.arange(n_comm), FedGDAGThist, '-rs', markevery = marker, label = 'FedGDA-GT')

### Plot Formatting
plt.grid(True)
plt.yscale('log')
plt.ylabel(r'Relative Error')
plt.xlabel('Communication Rounds')
plt.title('Quadratic Game')
plt.legend()
    



