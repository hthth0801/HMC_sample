# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:07:31 2015
HMC sampling to match specific statistics
@author: tian
"""

import numpy as np
#import theano
import theano.tensor as T
from build_f_df_matching import theano_f_df

class Gaussian:
    def __init__(self, mu, cov, *stats):
       self.num_stats = len(stats)
       self.match_stats=[]
       self.match_stats.append(stats[0])
       if self.num_stats == 2:
           self.match_stats.append(stats[1])    
       self.mu = mu
       self.cov = cov
       from scipy.linalg import inv
       self.cov_inv = inv(cov)
       #self.num_batches = num_batches
       self.f_df_theano = theano_f_df(self.theano_energy, self.match_stats)
       
    def theano_energy(self, x):
        """
        x: theano matrix. [n_samples]*[num_dim] to be used by theano_f_df
        """
        return 0.5 * (T.dot((x - self.mu), self.cov_inv) *
                      (x - self.mu)).sum(axis=1)
    def f_df(self, params, *args):
        """
        params:initial params
        
        args1:initial_v
        args2:stepsizes
        args3:num_steps
        args4:n_sample
        args5:n_dim
        """
       # print "length of args", len(args)
        initial_v = args[0]
        stepsizes0 = args[1]
        n_step = args[2]
        theano_args = [params] + [initial_v, stepsizes0, n_step]
        #theano_args = params + [initial_v, stepsizes0, self.n_step]
        results = self.f_df_theano(*theano_args)
        return results[0], results[1]
    def f_df_wrapper(self, params, *args):
        #input is the flattened version of params
        n_sample = args[3]
        n_dim = args[4]
        params_original = params.reshape(n_sample, n_dim)
        f1, df1 = self.f_df(params_original, *args)
        f = 0.
        df = 0.
        f+=f1
        df+=df1.flatten()
        return f, df
        
                      
     
         
       
       
       