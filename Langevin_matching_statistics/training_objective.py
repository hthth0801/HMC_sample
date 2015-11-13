# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:29:06 2015

@author: tian
"""

import theano
import theano.tensor as T
import lmc_sampling as LMC
import numpy as np

def theano_funcs(energy, stats_dict):
    n_step = T.iscalar('n_step')
    stepsizes = T.vector('stepsizes')
    initial_pos = T.matrix('initial_pos') #parameter for the representative samples     
    initial_vel = T.matrix('initial_vel')
    decay_rates = T.vector('decay_rates')
   
    sampler_cost = 0.
    # do Langevin sampling
    [all_pos, all_vel, all_noise, updates] = LMC.langevin_move(initial_pos, initial_vel, decay_rates, stepsizes, n_step, energy) 
        
    """
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
    """
    
    
    # get the samples in the end of the trajectory.
    final_pos = all_pos[-1] 
    nsamp = initial_pos.shape[0].astype(theano.config.floatX)
    for stat in stats_dict.itervalues():
        initial_stat = stat(initial_pos, T.ones_like(initial_pos)/nsamp)
        final_stat = stat(final_pos, T.ones_like(final_pos)/nsamp)
        sampler_cost = sampler_cost + T.sum((final_stat-initial_stat)**2)
    
    """
    # get the samples along the whole trajectory
    
    final_pos = all_pos[-1]
    initial_pos_vec = T.tile(initial_pos, [all_pos.shape[0],1])
    final_pos_vec = T.reshape(all_pos, (all_pos.shape[0]*all_pos.shape[1],all_pos.shape[2]))
    nsamp = final_pos_vec.shape[0].astype(theano.config.floatX)
    for stat in stats_dict.itervalues():
        initial_stat = stat(initial_pos_vec, T.ones_like(initial_pos_vec)/nsamp)
        final_stat = stat(final_pos_vec, T.ones_like(final_pos_vec)/nsamp)           
        sampler_cost = sampler_cost + T.sum((final_stat - initial_stat)**2)
    """
    
    # we want the gradient per-sample to stay large -- so scale by the number of samples!
    # this is # initial conditions * #steps
    sampler_cost *= nsamp

    ## and actually, let's make it really large -- see if this helps convergence
    #sampler_cost *= 1e10

    total_cost = sampler_cost   
    costs = [total_cost]
    gparams = [T.grad(total_cost, initial_pos)]
    report_scalars = costs
      
    f_df      = theano.function([initial_pos,initial_vel,decay_rates, stepsizes,n_step], report_scalars+gparams, name='func_f_df',updates = updates,  allow_input_downcast=True)
    f_samples = theano.function([initial_pos,initial_vel,decay_rates, stepsizes,n_step], [all_pos, all_vel, all_noise, final_pos], name='func_samples', updates = updates, allow_input_downcast=True)
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
        args2:decay_rates
        args3:stepsizes
        args4:num_steps
        args5:n_sample
        args6:n_dim
        """
       # print "length of args", len(args)
        initial_v = args[0]
        decay = args[1]
        stepsizes0 = args[2]
        n_step = args[3]
        theano_args = [params] + [initial_v, decay, stepsizes0, n_step]
        #theano_args = params + [initial_v, stepsizes0, self.n_step]
        results = self.f_df_theano(*theano_args)
        return results[0], results[1]
    def f_df_wrapper(self, params, *args):
        #input is the flattened version of params
        n_sample = args[4]
        n_dim = args[5]
        params_original = params.reshape(n_sample, n_dim)
        f1, df1 = self.f_df(params_original, *args)
        f = 0.
        df = 0.
        f+=f1
        print "tot_cost=%g, grad_pos = %g"%(
            f1,
            np.sqrt(np.mean(df1**2))
            )
        df+=df1.flatten()

        # DEBUG
        df = df.astype(float)

        return f, df
       
