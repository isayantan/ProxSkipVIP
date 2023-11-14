#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:23:55 2023

@author: sayantanchoudhury
"""

import numpy as np
from scipy.optimize import linprog

class MatrixGame():
    def __init__(self, row, col, n_node, game = 'PolicemenBurglar'):
        self.row = row
        self.col = col
        self.node = n_node
        self.game = game
        
        if self.game == 'PolicemenBurglar':
            """
            A_ij = w_i * (1 - exp(-theta * |i - j|)) where w_i = |w_i^'| with w_i^' ~ N(0, 1)
            """
            np.random.seed(0)
            theta = .8       # set \theta = 0.8
            M_node = []      # stores full block matrix for each node
            for _ in range(self.node):
                A = np.zeros((self.row, self.col))
                for i in range(self.row):
                    for j in range(self.col):
                        A[i, j] = np.abs(np.random.normal(0, 1)) * (1 - np.exp(-theta * np.abs(i - j)))
                M_node.append(np.block([[np.zeros((self.row, self.row)), A], [-np.transpose(A), np.zeros((self.col, self.col))]]))
        self.M_node = M_node
        self.M = np.mean(M_node, axis = 0)
        self.A = self.M[:self.row, self.row:]
    
    def generate_mx(self):
        return self.M_node
    
    def grad(self, x, node):
        return self.M_node[node]@x
    
    def optdist(self, x):
        """
        Compute the duality gap max_y f(x, y) - min_x f(x, y)
        """
        
        x1 = x[:self.row]
        x2 = x[self.row:]
        primal = linprog(c = self.A @ x2, A_eq = np.ones(self.row), b_eq = 1)
        dual = linprog(c = , A_eq = np.ones(self.col), b_eq = 1)
        return primal - dual
    
        
        
        
        