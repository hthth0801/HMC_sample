# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:29:06 2015

@author: tian
"""

import theano
import theano.tensor as T
import hmc_sampling as HMC
import numpy as np

def theano_funcs(energy, stats_dict):
    n_step = T.iscalar('n_step')
    stepsizes = T.vector('stepsizes')
    initial_pos = T.matrix('initial_pos') #parameter for the representative samples     
    initial_vel = T.matrix('initial_vel')
   
    sampler_cost = 0.

    # do one-step HMC sampling
    [accept, initial_pos_vec, final_pos_vec, ndeltaH, final_pos] = HMC.hmc_move(initial_vel, initial_pos, energy, stepsizes,n_step)
    nsamp = accept.shape[0].astype(theano.config.floatX)
    accept_matrix = accept.dimshuffle(0,'x')
    for stat in stats_dict.itervalues():
        initial_stat = stat(initial_pos_vec, T.ones_like(accept_matrix)/nsamp)
        final_stat = stat(
            T.concatenate((initial_pos_vec, final_pos_vec), axis=0),
            T.concatenate(
                (T.ones_like(accept_matrix)-accept_matrix, accept_matrix),
                axis=0)/nsamp,
            )
        sampler_cost = sampler_cost + T.sum((final_stat - initial_stat)**2)


    ## DEBUG
    #[accept, initial_pos_vec, final_pos_vec, ndeltaH, final_pos] = HMC.hmc_move(-initial_vel, initial_pos, energy, stepsizes,n_step)
    #nsamp = accept.shape[0].astype(theano.config.floatX)
    #accept_matrix = accept.dimshuffle(0,'x')
    #for stat in stats_dict.itervalues():
    #    initial_stat = stat(initial_pos_vec, T.ones_like(accept_matrix)/nsamp)
    #    final_stat = stat(
    #        T.concatenate((initial_pos_vec, final_pos_vec), axis=0),
    #        T.concatenate(
    #            (T.ones_like(accept_matrix)-accept_matrix, accept_matrix),
    #            axis=0)/nsamp,
    #        )
    #    sampler_cost = sampler_cost + T.sum((final_stat - initial_stat)**2)



        # difference = stat(initial_pos_vec) - stat(final_pos_vec)
        # weighted_difference = T.mean(accept_matrix*difference, axis=0)
        # sampler_cost = sampler_cost + T.sum(weighted_difference**2)





    # we want the gradient per-sample to stay large -- so scale by the number of samples!
    # this is # initial conditions * #steps
    sampler_cost *= nsamp

    ## and actually, let's make it really large -- see if this helps convergence
    #sampler_cost *= 1e10

    total_cost = sampler_cost   
    costs = [total_cost]
    gparams = [T.grad(total_cost, initial_pos)]
      
    f_df      = theano.function([initial_pos,initial_vel,stepsizes,n_step], costs+gparams, name='func_f_df', allow_input_downcast=True)
    f_samples = theano.function([initial_pos,initial_vel,stepsizes,n_step], [initial_pos_vec, final_pos_vec, final_pos], name='func_samples', allow_input_downcast=True)
    return f_df, f_samples


class training_objective:
    def __init__(self, energy, stats_dict):
       #self.num_batches = num_batches
       self.f_df_theano, self.f_samples = theano_funcs(energy.E, stats_dict)
       
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

        # DEBUG
        df = df.astype(float)

        return f, df
       
