# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:26:35 2015
optimizer for for matching statistics
@author: tian
"""
import numpy as np
import scipy.optimize
def RMSprop(objective, alg_params, initial_params, args_hyper):
    """
    objective: training objective, contains funtion to compute f_df
    alg_params: hyper parameters for the RMSprop algorithm [decay_rate, learning_rate, num_passes]
    initial_params: initialization for the parameters which we want to optimize, vectorized version
    args_hyper: set of hyper-parameters for the theano function. 
                [initial_v, decay_rates, stepsizes, num_step, n_sample, n_dim]
    """
    n_sample = args_hyper[-2]
    n_dim = args_hyper[-1]
    decay_rate = alg_params[0]
    eta = alg_params[1] # learning_rate
   # N = len(mini_batches_x)
    num_passes = alg_params[2]
  #  f_eval = np.ones(N)*np.nan
    params_i = initial_params.copy()
    cache = 0.0
    for ipass in range(num_passes):
        f_i, df_i = objective.f_df_wrapper(params_i, *args_hyper)
        cache = decay_rate * cache + (1.0 - decay_rate) * df_i **2
        params_i += -eta * df_i / np.sqrt(cache + 1e-8)
        #print "cost = ", f_i
        if not np.isfinite(f_i):
            print("Non-finite func")
            break
    optimal_params = params_i
    best_samples = optimal_params.reshape(n_sample, n_dim)
    return best_samples
    
def LBFGS(objective, alg_params, initial_params, args_hyper):
    """
    Here, alg_params just the number of func. calls for the l_bfgs algorithms
    """
    n_sample = args_hyper[-2]
    n_dim = args_hyper[-1]
    num_passes = alg_params[0]
    
    best_samples_list = scipy.optimize.fmin_l_bfgs_b(objective.f_df_wrapper, 
                                    initial_params.copy(),
                                    args = args_hyper,
                                    maxfun=num_passes,
                                    # disp=1,
                                    )
    best_samples = best_samples_list[0].reshape(n_sample, n_dim)
    return best_samples

