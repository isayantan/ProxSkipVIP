#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:55:28 2023

@author: sayantanchoudhury
"""

import numpy as np
from model import QuadGame 
import matplotlib.pyplot as plt

### Iteration and communication
n_comm = 400         # number of communication rounds

### Initialize paramters
n_node = 20         # number of nodes
n_feature = 20       # dimension of data
n_data = 100         # size of dataset

### Initialize based on model
game = QuadGame(n_feature, n_node, n_data, method = 'full batch')
dF = game.grad
measure = game.optdist             # relative distance of current update from x*

"""
Implement ProxSkipVIP algorithm.
"""

### Initialize hyperparameters
gamma = 1/max(game.cocoercive_node)                  # stepsize \gamma
prob_comm = np.sqrt(gamma * min(game.mu_node))       # communication probability

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
        for node in range(n_node):
            x[node] = np.mean(xdash, axis = 0)                                       # update local variable with the mean of all variable
            control[node] += (prob_comm/ gamma) * (x[node] - xhat[node])            # update the local control variates
        ProxSkipVIPhist.append(measure(np.mean(x, axis = 0)))                       # store the distance to optimality
    else:
        for node in range(n_node):
            x[node] -= gamma * (dF(x[node], node) - control[node])                  # update local variable with gradient step and keep control variate same
ProxSkipVIPhist = np.array(ProxSkipVIPhist)

"""
Implement Local GDA.
"""

### Initialize hyperparameters
local_step = int(n_comm/n_node)                                     # local steps
a = 2048 * local_step * ((max(game.L_node)/min(game.mu_node))**2)   # constant for step-size computation

### Initialize variables
x = np.zeros((n_node, n_feature))                                   # local var
global_x = np.zeros(n_node)                                         # global var
LocalGDAhist = []                                                   # stores the distance to optimality
comm = 0                                                            # number of communication

while comm < n_comm:
    for node in range(n_node):
        x[node] = global_x
        for step in range(local_step):
            gamma = 8/(min(game.mu_node) * (a + (comm * local_step + step + 1)))
            x[node] -= gamma * dF(x[node], node)
    global_x = np.mean(x, axis = 0)
    LocalGDAhist.append(measure(global_x))                          # store distance to optimality for global variable 
    comm += 1                                                       # update communication 

"""
Implement Local EG.
"""

### Initialize Hyperparameters
local_step = int(n_comm/n_node)
gamma1 = 1/(21 * local_step * max(game.L_node))
gamma2 = 1/(21 * local_step * max(game.L_node))

### Initialize variables
x = np.zeros((n_node, n_feature))                                # local var
global_x = np.zeros(n_feature)                                   # global var
LocalEGhist = []                                                 # stores the distance to optimality
comm = 0

while comm < n_comm:
    for node in range(n_node):
        x[node] = global_x
        for _ in range(local_step):
            temp = x[node] - gamma1 * dF(x[node], node)
            x[node] -= gamma2 * dF(temp, node)
    global_x = np.mean(x, axis = 0)
    LocalEGhist.append(measure(global_x))                        # store distance to optimality for global variable
    comm += 1                                                    # update communication
 
"""
We implement the FedGDA-GT algorihtm from https://arxiv.org/pdf/2206.01132.pdf, which uses gradient tracking technique with local steps.

We use the theoretical step size choices from the page 20 of that paper.
"""

### Initialize hyperparameters
# local_step = int(np.ceil(1/prob_comm))
local_step = int(n_comm/n_node)
root = np.roots([-max(game.L_node**4)*local_step**4, 0, -2*max(game.L_node**2)*local_step**2, min(game.mu_node)*local_step, 0])
gamma = .5 * min((2 * min(game.mu_node))/(max(game.L_node)**2), 1/(2 * min(game.mu_node) * local_step), min(root[np.where((root.imag == 0) & (root.real > 0))].real))

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

"""
Plot the Trajectories
"""

marker = np.arange(0, n_comm, n_comm/10, dtype='int')
plt.plot(np.arange(n_comm), LocalEGhist, '-r>', markevery = marker, label = 'Local EG')
plt.plot(np.arange(n_comm), LocalGDAhist, '-bo', markevery = marker, label = 'Local GDA')
plt.plot(np.arange(n_comm), FedGDAGThist, '-cs', markevery = marker, label = 'FedGDA-GT')
plt.plot(np.arange(n_comm), ProxSkipVIPhist, '-gd', markevery = marker, label = 'ProxSkip-GDA-FL')

### Plot Formatting
plt.grid(True)
plt.yscale('log')
plt.ylabel(r'Relative Error')
plt.xlabel('Communication Rounds')
plt.title('Quadratic Game')
plt.legend()

    



