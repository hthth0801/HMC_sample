# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:33:39 2015
Optimation package that contains different solvers for the task
borrow from Jascha Sohl-dickstein, modify some names and simplified

@author: tian
"""

import numpy as np
import scipy.optimize
#from sfo import SFO
class train:
    def __init__(self, model):
        self.model = model
        self.param_init_flatten = self.theta_list_to_flat(self.model.initial_params)
        self.param_init = self.model.initial_params
        self.n_batches = len(self.model.mini_batches)
        
    def f_df_flat(self, params_flatten, mini_batches):
        params = self.theta_flat_to_list(params_flatten)
        f=0.
        df=0.
       # print "mini_batches shape = ", mini_batches[0].shape
        for ibatch in mini_batches:
            print "ibatch"
        #    print "ibatch = ", ibatch.shape
            f_i, df_i = self.model.f_df(params, ibatch)
            df_i_flatten = self.theta_list_to_flat(df_i)
            print "cost= ", f_i
            f += (f_i[0]+f_i[1])
            df += df_i_flatten     
        return f, df.ravel()
        #return f_i, df_i
    
    def SGD(self, n_passes = 20):
        N_batches = self.n_batches
        param = self.param_init_flatten.copy()
        for iter in range(n_passes*N_batches):
            idx = np.random.randint(N_batches)
            r_batch = self.model.mini_batches[idx]
            fc,dfc = self.f_df_flat(param, [r_batch,])
            param -= dfc.reshape(param.shape) * 0.2
        return param
        
        
    def LBFGS(self, n_passes=5000):
        best_results = scipy.optimize.fmin_l_bfgs_b(self.f_df_flat, 
                                                    self.param_init_flatten.copy(),
                                                    disp=1,
                                                    args = (self.model.mini_batches,), 
                                                    maxiter=n_passes)
        return best_results
        
    def LBFGS_minibatch(self, n_passes = 20, fraction=0.1, n_lbfgs_step=10):
        param = self.param_init_flatten.copy()
        N_batches = self.n_batches
        for iter in range(n_passes):
            perm = np.random.permutation(N_batches)
            k = int(fraction*N_batches)
            idx = perm[:k]
            batch_idx = []
            for i in idx:
                batch_idx.append(self.model.mini_batches[i])
            param, _, _ = scipy.optimize.fmin_l_bfgs_b(self.f_df_flat,
                                                       param, 
                                                       args=(batch_idx,),
                                                       maxfun = n_lbfgs_step)
        return param
    
   # def SFO (self, n_passes = 20):
   #     self.optimizer = SFO(self.f_df_flat, self.model.initial_params, self.model.mini_batches)
   #     x = self.optimizer.optimize(num_passes = n_passes)
   #     return x
        
        
    def theta_list_to_flat(self, theta_list):
        """
        Convert from a list of numpy arrays into a 1d numpy array.
        """
        num_el = 0
        for el in theta_list:
            num_el += np.prod(el.shape)
        theta_flat = np.zeros((num_el, 1))
        start_indx = 0
        for el in theta_list:
            stop_indx = start_indx + np.prod(el.shape)
            theta_flat[start_indx:stop_indx, 0] = el.ravel()
            start_indx = stop_indx
        return theta_flat
        
    def theta_flat_to_list(self, theta_flat):
        """
        Convert from a 1d numpy arfray into a list of numpy arrays.
        """
        if len(theta_flat.shape) == 1:
            # make it Nx1 rather than N, for consistency
            theta_flat = theta_flat.reshape((-1,1))
        theta_list = []
        start_indx = 0
        for el in self.param_init:
            stop_indx = start_indx + np.prod(el.shape)
            theta_list.append(theta_flat[start_indx:stop_indx,0].reshape(el.shape))
            start_indx = stop_indx
        return theta_list

    
