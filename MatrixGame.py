#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:23:55 2023

@author: sayantanchoudhury
"""

import numpy as np

class MatrixGame():
    def __init__(self, row, col, n_node, game = 'PolicemenBurglar'):
        self.row = row
        self.col = col
        self.node = n_node
        self.game = game
        
        if self.game == 'PolicemenBurglar':
            np.random.seed(0)
            theta = .8
            M_node = []
            for _ in range(self.node):
                A = np.zeros((self.row, self.col))
                for i in range(self.row):
                    for j in range(self.col):
                        A[i, j] = np.abs(np.random.normal(0, 1)) * (1 - np.exp(-theta * np.abs(i - j)))
                M_node.append(np.block([[np.zeros((self.row, self.row)), A], [-np.transpose(A), np.zeros((self.col, self.col))]]))
        self.M_node = M_node
                    
    
    def grad(self, x, node):
        return self.M_node[node]@x
        