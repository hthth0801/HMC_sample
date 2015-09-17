# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:24:46 2015

@author: tian
"""

from Model_match import Gaussian
import numpy as np
import theano
import scipy.optimize
import timeit

#initialize
n_sample = 100
n_dim =2
rng = np.random.RandomState(123)
mu = np.array(rng.rand(n_dim)*5, dtype=theano.config.floatX)
cov = np.array([[0.8, 0.], [0., 0.6]], dtype=theano.config.floatX)

rng1 = np.random.RandomState(444)
random_stepsizes = rng1.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.2
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 
        
rng2 = np.random.RandomState(1)
initial_v = rng2.randn(n_sample, n_dim)

rng3 = np.random.RandomState(125)
initial_params = rng3.randn(n_sample, n_dim)
initial_params_flat = initial_params.flatten()
args_hyper = [initial_v, stepsizes0, 30,n_sample,n_dim]
args_stats = ['first', 'H']
gaussian_2D = Gaussian(mu, cov, *args_stats)
res = gaussian_2D.f_df(initial_params, *args_hyper)
best_samples = scipy.optimize.fmin_l_bfgs_b(gaussian_2D.f_df_wrapper, 
                                            initial_params_flat,
                                            args = args_hyper,
                                            maxfun=100)
res = best_samples[0].reshape(n_sample, n_dim)
print "estimated mean from representative sample= ", res.mean(axis=0)
print "true mean= ", mu                                         
