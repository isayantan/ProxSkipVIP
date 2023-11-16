#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:53:38 2023

@author: sayantanchoudhury
"""

import numpy as np

class QuadGame():
    def __init__(self, n_feature, n_node, n_data, L = 1, mu = 0.01, method = 'full batch', q = 1, mini_batch = 1):
        self.n_feature = int(n_feature/2)
        self.n_node = n_node
        self.n_data = n_data
        self.L = L
        self.mu = mu
        self.method = method
        self.q = q
        self.mini_batch = mini_batch
        
        ### Generate Data
        A_node, B_node, C_node, M_node, z_node = [],[],[],[],[]
        a_node, c_node = [], []
        M_node_data, z_node_data = [], []
        mu_node, L_node = [],[]
        cocoercive_data = []
        cocoercive_node = []
        for node in range(self.n_node):
            A_node_data, B_node_data, C_node_data = [], [], []
            a_node_data, c_node_data = [], []
            for _ in range(n_data):
                evalues = np.random.uniform(self.mu, self.L, self.n_feature)
                rndm_mx = np.random.normal(0, 1, (self.n_feature, self.n_feature))
                _, Q = np.linalg.eig(rndm_mx.T @ rndm_mx)
                A_node_data.append(Q @ np.diag(evalues) @ Q.T)
                
                evalues = np.random.uniform(self.mu, self.L, self.n_feature)
                rndm_mx = np.random.normal(0, 1, (self.n_feature, self.n_feature))
                _, Q = np.linalg.eig(rndm_mx.T @ rndm_mx)
                C_node_data.append(Q @ np.diag(evalues) @ Q.T)
                
                evalues = np.random.uniform(0, self.L, self.n_feature)
                rndm_mx = np.random.normal(0, 1, (self.n_feature, self.n_feature))
                _, Q = np.linalg.eig(rndm_mx.T @ rndm_mx)
                B_node_data.append(Q @ np.diag(evalues) @ Q.T)
                
                M_node_data.append(np.block([[A_node_data[-1], B_node_data[-1]],[-B_node_data[-1], C_node_data[-1]]]))
                a_node_data.append(np.random.normal(0,1,self.n_feature))
                c_node_data.append(np.random.normal(0,1,self.n_feature))
                z_node_data.append(np.concatenate((a_node_data[-1], c_node_data[-1])))
                eval_data = np.linalg.eig(M_node_data[-1])[0]                                     # compute eigenvalue of each data
                cocoercive_data.append(1/np.min(np.real(1/eval_data[np.abs(eval_data) > 1e-5])))  # store cocoercive constant for each data  
            
            A_node.append(np.mean(A_node_data, axis = 0))
            B_node.append(np.mean(B_node_data, axis = 0))
            C_node.append(np.mean(C_node_data, axis = 0))
            M_node.append(np.block([[A_node[-1], B_node[-1]],[-B_node[-1], C_node[-1]]]))
            eval_node = np.linalg.eig(M_node[-1])[0]                                              # compute eigenvalue of M
            cocoercive_node.append(1/np.min(np.real(1/eval_node[np.abs(eval_node) > 1e-5])))      # store cocoercive constant for each node
            mu_node.append(np.min((np.linalg.eig(A_node[-1])[0], np.linalg.eig(C_node[-1])[0])))
            L_node.append(np.sqrt(max(np.linalg.eig(M_node[-1].T @ M_node[-1])[0])))
            a_node.append(np.mean(a_node_data, axis=0))
            c_node.append(np.mean(c_node_data, axis=0))
            z_node.append(np.concatenate((a_node[-1], c_node[-1])))
            
        self.x_optimal = - np.linalg.inv(np.mean(M_node_data, axis = 0)) @ np.mean(z_node_data, axis = 0)      # store optimal solution of Quad game
        self.lipschitz = np.sqrt(np.max(np.linalg.eig(np.mean(M_node, axis = 0).T @ np.mean(M_node, axis = 0))[0]))
        self.M_node = np.array(M_node)
        self.z_node = np.array(z_node)
        self.M_node_data = np.array(M_node_data)
        self.z_node_data = np.array(z_node_data)
        
    def grad(self, x, node, last_update = None):
        """
        Computes gradient estimate.
        
        Use last_update for variance-reduced type method. 
        """
        if self.method == 'full batch':
            return self.M_node[node]@x + self.z_node[node]
            # return self.M_node[node]@x + np.mean(self.z_node_data[np.arange(node*self.n_data, (node+1)*self.n_data)], axis=0)
            
        elif self.method == 'mini batch':
            index = np.random.choice(np.arange(node*self.n_data, (node+1)*self.n_data), self.mini_batch, replace=False)
            return np.mean(self.M_node_data[index], axis = 0)@x + np.mean(self.z_node_data[index], axis=0)
        elif self.method == 'variance reduced':
            index = np.random.choice(np.arange(node*self.n_data, (node+1)*self.n_data), self.mini_batch, replace=False)
            result = np.mean(self.M_node_data[index], axis = 0)@x + np.mean(self.z_node_data[index], axis=0) - np.mean(self.M_node_data[index], axis = 0)@last_update - np.mean(self.z_node_data[index], axis=0) + self.M_node[node]@last_update + np.mean(self.z_node_data[np.arange(node*self.n_data, (node+1)*self.n_data)], axis=0)
            return result
    
    def optdist(self, x):
        """
        Computes the relative error.
        
        Note that, here we assume x_0 = 0.
        """
        return np.sum((x - self.x_optimal)**2)/np.sum((self.x_optimal)**2)     