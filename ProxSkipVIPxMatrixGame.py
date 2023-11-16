#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:21:13 2023

@author: sayantanchoudhury
"""

import numpy as np
from ProximalOperator import projection_simplex_sort
from MatrixGame import MatrixGame
import matplotlib.pyplot as plt

### Iteration and communication
n_comm = 100         # number of communication rounds

### Initialize paramters
n_node = 16          # number of nodes
n_row = 5            # dimension of x
n_col = 5            # dimension of y
n_feature = n_row + n_col       # dimension of data = dim(x) + dim(y)
n_data = 100         # size of dataset

### Initialize based on model
game = MatrixGame(n_row, n_col, n_node)
dF = game.grad             # computes operator
measure = game.dualgap     # measure duality gap

### Initialize hyperparameters
gamma = 1e-10       # stepsize \gamma
prob_comm = 1      # communication probability

### Initialize variables
x, control, xhat, xdash = np.zeros((4, n_node, n_feature))    # local var, control var
ProxSkipVIPhist = []                                   # stores the distance to optimality
comm = 0                                               # number of communication

while comm < n_comm:
    theta = np.random.binomial(1, prob_comm)           # generate \theta from bernoulli(prob_comm)
    if theta:
        comm += 1                                      # for \theta = 1, take a communication step
        for node in range(n_node):
            xhat[node] = x[node] - gamma * (dF(x[node], node) - control[node])      # compute local xhat for each node
            xdash[node] = xhat[node] - (gamma/ prob_comm) * control[node]
        x_comm = np.mean(xdash, axis = 0)
        x_comm = np.concatenate((projection_simplex_sort(x_comm[:n_row], 1), projection_simplex_sort(x_comm[n_row:], 1)))
        for node in range(n_node):
            x[node] = x_comm
            control[node] += (prob_comm/ gamma) * (x[node] - xhat[node])            # update the local control variates
        ProxSkipVIPhist.append(measure(np.mean(x, axis = 0)))                       # store the distance to optimality
    else:
        for node in range(n_node):
            x[node] -= gamma * (dF(x[node], node) - control[node])                  # update local variable with gradient step and keep control variate same
ProxSkipVIPhist = np.array(ProxSkipVIPhist)

### Plot the Trajectories
marker = np.arange(0, n_comm, n_comm/10, dtype='int')
plt.plot(np.arange(n_comm), ProxSkipVIPhist, '-gd', markevery = marker, label = 'ProxSkipVIP')

### Plot Formatting
plt.grid(True)
plt.yscale('log')
plt.ylabel(r'Duality Gap')
plt.xlabel('Communication Rounds')
plt.title('Policemen Burglar Game')
plt.legend()



