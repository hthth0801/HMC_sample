# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:29:06 2015

@author: tian
"""

import theano
import theano.tensor as T
import hmc_sampling as HMC


def theano_f_df(energy, stats):
    n_step = T.iscalar('n_step')
    stepsizes = T.vector('stepsizes')
    initial_pos = T.matrix('initial_pos') #parameter for the representative samples     
    initial_vel = T.matrix('initial_vel')
    num_stats = len(stats)
   
    params = initial_pos
    # do one-step HMC sampling
    [accept,accept1, final_pos, final_pos1, ndeltaH] = HMC.hmc_move(initial_vel, initial_pos, energy, stepsizes,n_step)
    accept_matrix = accept.dimshuffle(0,1, 'x')
    if stats[0] == 'first':
        sampler_cost = accept_matrix*(initial_pos-final_pos)
    if stats[0] == 'second':
        sampler_cost = accept_matrix*(initial_pos**2-final_pos**2)
    if stats[0] == 'third':
        sampler_cost=accept_matrix*(initial_pos**3-final_pos**3)
    if stats[0] == 'exp':
        sampler_cost = accept_matrix*(T.exp(initial_pos)-T.exp(final_pos))
    if stats[0] =='sin':
        sampler_cost = accept_matrix*(T.sin(initial_pos)-T.sin(final_pos))
    sampler_cost= T.mean(sampler_cost, axis=0)
    sampler_cost = T.sum(sampler_cost, axis=0)
    sampler_cost = T.mean(sampler_cost**2)
    # add the possible Hamiltonian energy matching
    if num_stats == 2:
        sampler_cost_H = T.mean(accept*ndeltaH, axis=0)
        sampler_cost_H = T.sum(sampler_cost_H)
        sampler_cost = sampler_cost + sampler_cost_H**2
    total_cost = sampler_cost   
    costs = [total_cost]
    gparams = [T.grad(total_cost, params)]     
      
    f_df=theano.function([params,initial_vel,stepsizes,n_step], costs+gparams, name='func_f_df', allow_input_downcast=True)
    return f_df
