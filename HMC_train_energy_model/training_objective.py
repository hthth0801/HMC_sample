# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 14:10:34 2015

@author: user
"""
import theano
import theano.tensor as T
import hmc_sampling as HMC
import numpy as np
from datasets import load_van_hateren, load_pixel_sparse
import timeit
"""
dataset is the dictory. e.g., {'mnist': load_mnist, 'van_hateren': load_van_hateren}
The key is the name of the dataset, the value is the loading function, all necessary pre-processing are defined there. 
"""
data_dict = {
            # 'van_hateren': load_van_hateren,
             'pixel_sparse': load_pixel_sparse,
            # 'mnist': load_mnist
            # 'gaussian_2d': load_gaussian_2d
             }

def theano_funcs(theta,energy, dE_dtheta):
    n_step = T.iscalar('n_step')
    stepsizes = T.vector('stepsizes')
    initial_pos = T.matrix('initial_pos') #parameter for the representative samples     
    initial_vel = T.matrix('initial_vel')
    training = T.matrix('training')
     
    # params includes two parts: one from the energy based model. one from the representive samples
    params = [initial_pos, theta]
    # do one-step HMC sampling
    [accept, initial_pos_vec, final_pos_vec, ndeltaH, final_pos] = HMC.hmc_move(initial_vel, initial_pos, energy, stepsizes,n_step)
    accept_flatten = accept.flatten()
    # get sampler_cost
    sampler_cost = dE_dtheta(initial_pos_vec, accept_flatten) - dE_dtheta(final_pos_vec, accept_flatten)
    #sampler_cost = dE_dtheta(initial_pos_vec, accept_flatten)
    sampler_cost = 1./(final_pos.shape[0]) * sampler_cost
    sampler_cost = T.mean(sampler_cost**2)
    # get param_cost
    param_cost = dE_dtheta(training) - dE_dtheta(initial_pos)
    param_cost = T.mean(param_cost**2)
        
    total_cost = param_cost + sampler_cost
    costs = [param_cost, sampler_cost]
    gparams = []
    for param in params:
        gparam = T.grad(total_cost, param)
        gparams.append(gparam)
    
    f_df=theano.function(params + [training, initial_vel,stepsizes,n_step], costs+gparams, name='func_f_df', allow_input_downcast=True)
    f_samples = theano.function(params + [initial_vel, stepsizes, n_step], [initial_pos_vec, final_pos_vec, final_pos], name='func_samples', allow_input_downcast=True)
    
    return f_df, f_samples
    
class training_objective:
    """
    data_name : string. Specify which dataset we wanna use
    """
    def __init__(self, energy_model, data_name, n_batches =100):
        start_time = timeit.default_timer()
        self.f_df_theano, self.f_samples = theano_funcs(energy_model.theta, energy_model.E, energy_model.dE_dtheta)
        end_time = timeit.default_timer()
        print "compiling time = ", end_time-start_time
        #For now, consider we only have training data, but not the training labels.
        start_time = timeit.default_timer()
        self.training_X = data_dict[data_name](5,100000)
        end_time = timeit.default_timer()
        print "loading van hateren time = ", end_time-start_time
        self.mini_batches=[]
        for ibatch in range(n_batches):
            self.mini_batches.append(self.training_X[ibatch::n_batches,:])
        
    def f_df(self, params, training_X, *args):
        """
        params:initial params, list contains the parameters for samples and parameters for energy-based model
        training_X: the training batches
        args: hyper-parameters used by HMC       
          args1:initial_v
          args2:stepsizes
          args3:num_steps
          args4:n_sample
          args5:n_dim
        """
        initial_v = args[0]
        stepsizes0 = args[1]
        n_step = args[2]
        theano_args = params + [training_X] + [initial_v, stepsizes0, n_step]
        results = self.f_df_theano(*theano_args)
        """
        rval 1: param_cost and sampler_cost (list)
        rval 2: grad_sampler and grad_param (list)
        """
        return results[:2], results[2:]
        
    def f_df_wrapper(self, params, *args):
        #input is the flattened version of params
        """
        args: training_X and the hyper-parameters
        args[0]: training_X
        args[1:]: hyper-parameters
        """
        training_X = args[0]
        n_sample = args[-2]
        n_dim = args[-1]
        """
        flat to list
        """
        params_original = []
        params_original.append(params[:(n_sample*n_dim)].reshape(n_sample, n_dim)) # for representative samples
        params_original.append(params[(n_sample*n_dim):].reshape(n_dim, n_dim)) # for parameters
        #params_original = params.reshape(n_sample, n_dim)
        f1, df1 = self.f_df(params_original, training_X, *args[1:])
        f = 0.
        df = 0.
        f += (f1[0] + f1[1])
        print "param_cost=", f1[0]
        print "sampler_cost=", f1[1]
        # df1 is the list, so need to convert it to flat representation
        """
        list to flat, use the handy func from SFO
        """
        df1_flat = theta_list_to_flat(df1)
        df+=df1_flat.flatten()
        return f, df
        
def theta_list_to_flat(theta_list):
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
        
    
        


    
    
    
    
