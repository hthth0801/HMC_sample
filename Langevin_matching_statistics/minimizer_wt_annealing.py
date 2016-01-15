# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:56:37 2016
minimizer that returns the gradient evaluation numbers and the estimated parameters along the optimation process
@author: tian
"""
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
def RMSprop(objective, alg_params, initial_params, args_hyper):
    """
    objective: training objective, contains funtion to compute f_df
    alg_params: hyper parameters for the RMSprop algorithm [decay_rate, learning_rate, num_passes]
    initial_params: initialization for the parameters which we want to optimize, vectorized version
    args_hyper: set of hyper-parameters for the theano function. 
                [initial_v, decay_rates, stepsizes, num_step, n_sample, n_dim]
            or[objective.data, objective.label, initial_v, decay_rates, stepsizes, num_step, n_sample, n_dim]
    """
    n_step = args_hyper[-3]
    n_sample = args_hyper[-2]
    n_dim = args_hyper[-1]
    decay_rate = alg_params[0]
    """
    initial learning rate
    """
    eta_0 = alg_params[1] # learning_rate
   # N = len(mini_batches_x)
    num_passes = alg_params[2]
    """
    Learning rate schedule: O(1/t), epsilon_t = eta_0 * tao / max(t, tao)
    or exponentially. eta_t = eta_0 * 10^(-t/tao)
    """
    tao = num_passes / 3.
   
    
    """
    We also try to annealing the step of the leapfrog integrator
    Similar to inverse annealing wrt the iteration number: ceil(100*ipass/num_pass)
    """
    #n_step_anneal = args_hyper[-3]
    #args_hyper_anneal =  list(args_hyper)
    #tao_step = n_step_anneal / 10.
   # tao_step = 10.
    
    """
    get a dict contains the #GE and the corresponding esimated parameters along the optimization process
    """
    num_GE = 0
    sample_GE = {}
  #  f_eval = np.ones(N)*np.nan
    params_i = initial_params.copy()
    #num_GE += n_sample*n_step
    sample_GE[num_GE] = params_i.copy().reshape(n_sample, n_dim)    
    
    cache = 0.0
    cost = []
    #eta_min = eta_0 * 10**(-num_passes / tao)
    for ipass in range(num_passes):
        num_GE += n_sample* n_step
        """
        We inversly annealing the number of leapfrog steps
        Just use the new args_hyper_anneal list to store the new hyper-parameters 
        Only change the [-3] item, which the n_step
        """
        #args_hyper_anneal[-3] = np.ceil(np.max([tao_step, ipass+1.])*float(n_step_anneal)/num_passes)
        #args_hyper_anneal[-3] = np.ceil((ipass+1.)*float(n_step_anneal)/num_passes) # from small to large
        #args_hyper_anneal[-3] = np.ceil(float(4.)*num_passes/(ipass+1.))
        #print args_hyper_anneal[-3]         
        """
        linearly, uncomment the following line
        """
        #eta_t = eta_0 * tao / np.max([ipass, tao])
        """
        exponentially
        """
        eta_t = eta_0 * 10**(-ipass/tao)
        f_i, df_i = objective.f_df_wrapper(params_i, *args_hyper)
        cost.append(f_i)
        cache = decay_rate * cache + (1.0 - decay_rate) * df_i **2
        params_i += -eta_t * df_i / np.sqrt(cache + 1e-8)
        #current_params = params_i.copy()
        #sample_GE[num_GE] = params_i.reshape(n_sample, n_dim)
        sample_GE.update({num_GE: params_i.copy().reshape(n_sample, n_dim)})
        #print "dict value is ", sample_GE[num_GE]
        #print "cost = ", f_i
        if not np.isfinite(f_i):
            print("Non-finite func")
            break
    cost_np = np.asarray(cost)
    optimal_params = params_i
    best_samples = optimal_params.reshape(n_sample, n_dim) 
    """
    uncomment the following if we want to plot the convergence curve
    """

    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.title('convergence')
    plt.plot(cost_np)
    plt.ylabel('cost')
    plt.xlabel('iteration')


    return best_samples, cost_np[-1], sample_GE
    


