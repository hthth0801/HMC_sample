# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 09:57:36 2015

@author: user
"""

import numpy as np
import scipy.linalg as linalg

import theano
import theano.tensor as T

rng = np.random.RandomState(4321)

class gauss_2d:
    def __init__(self):
        self.mu_np_true = np.array([1,0.25], dtype = theano.config.floatX).reshape((-1,1))
        self.mu_true = theano.shared(self.mu_np_true.ravel()).reshape((-1,1))
        self.cov_np = np.array([[0.8,0.5], [0.5, 0.6]], dtype=theano.config.floatX)
        self.cov_inv = theano.shared(linalg.inv(self.cov_np))
        self.theta = T.vector('theta')
        self.name = 'gaussian_2d'
    def E(self, X):
        return T.sum(T.dot((X-self.theta), self.cov_inv)*(X-self.theta), axis=1)/2.
    def dE_dtheta(self, X, acpt = None):
        if acpt == None:
            return T.grad(T.mean(self.E(X)), self.theta, consider_constant=[X])
        else:
            return T.grad(T.sum(acpt*self.E(X)), self.theta, consider_constant=[X, acpt])
    def generate_training_samples(self, n_sample):
        samples_sd_normal = rng.normal(size = (n_sample, 2)).astype(theano.config.floatX)
        samples_true = (linalg.sqrtm(self.cov_np).dot(samples_sd_normal.T)).T + self.mu_np_true
        return samples_true
	
    
class ICA_soft_laplace:
    def __init__(self):
        # theta here is the receptive field J:[n_dim]* [n_expert], for ica, J is a square matrix
        self.theta = T.matrix('theta')
        self.epsilon_np  = np.array(0.1, dtype = theano.config.floatX)
        self.epsilon = theano.shared(self.epsilon_np)
        self.name = 'ICA_soft_laplace'
    def E(self, X):
        XJ = T.dot(X, self.theta) # [n_sample]*[n_expert]
        XJ2_ep = self.epsilon + XJ**2
        return T.sum(T.sqrt(XJ2_ep), axis=1)
    def dE_dtheta(self, X, acpt=None):
        if acpt == None:
            return T.grad(T.mean(self.E(X)), self.theta, consider_constant=[X])
        else:
            return T.grad(T.sum(acpt*self.E(X)), self.theta, consider_constant=[X, acpt])
    
    
        
        
        
        
        
        
        
        
        
        
        
        
  