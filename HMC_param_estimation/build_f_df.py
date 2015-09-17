# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 19:13:51 2015
Use theano to build the cost and the corresponding gradient. 


@author: tian
"""
import theano
import theano.tensor as T
import hmc_sampling as HMC


def theano_f_df(theta, energy, dE_dtheta):
    n_step = T.iscalar('n_step')
    stepsizes = T.vector('stepsizes')
    initial_pos = T.matrix('initial_pos') #parameter for the representative samples     
    initial_vel = T.matrix('initial_vel')
    training = T.matrix('training')
   # theta = T.vector('theta')
    
   
     
    # params includes two parts: one from the energy based model. one from the representive samples
    params = [initial_pos, theta]
    # do one-step HMC sampling
    [accept,accept1, final_pos, final_pos1, ndeltaH] = HMC.hmc_move(initial_vel, initial_pos, energy, stepsizes,n_step)
    initial_pos_vec = T.tile(initial_pos, [final_pos.shape[0],1])
    accept_flatten = accept.flatten()
    final_pos_flatten = T.reshape(final_pos, (final_pos.shape[0]*final_pos.shape[1],final_pos.shape[2]))
    # get sampler_cost
    sampler_cost = dE_dtheta(initial_pos_vec, accept_flatten) - dE_dtheta(final_pos_flatten, accept_flatten)
    #sampler_cost = dE_dtheta(initial_pos_vec, accept_flatten)
    sampler_cost = 1./(final_pos.shape[0]) * sampler_cost
    sampler_cost = T.mean(sampler_cost**2)
    # get param_cost
    param_cost = dE_dtheta(training) - dE_dtheta(initial_pos)
    param_cost = (final_pos.shape[1])*T.mean(param_cost**2)
        
    total_cost = param_cost + sampler_cost
    costs = [param_cost, sampler_cost]
    gparams = []
    for param in params:
        gparam = T.grad(total_cost, param)
        gparams.append(gparam)
    
    f_df=theano.function(params+[training, initial_vel,stepsizes,n_step], costs+gparams, name='func_f_df', allow_input_downcast=True)
    return f_df
