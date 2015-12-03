# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:17:47 2015
performance plot for different # of samples
@author: tian
"""
import numpy as np
import timeit
import scipy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from collections import defaultdict
import itertools

import energies
import training_objective
from minimizer import RMSprop, LBFGS 
from theano import config

def generate_plot(energy, stats_dict, ndim=2, true_init=False,
    num_samplecounts=25, max_samplecount=20000,
    # num_samplecounts=10, max_samplecount=50,
    n_steps=300,
    # n_steps=10,
    ):
    # TODO break each subplot into its own function.

    rng = np.random.RandomState(12)


    plt.figure(figsize=(17,17))
#    plt.figure(figsize=(10,7))
    plt.subplot(2,2,3)

    """
    draw the 2D contour
    """
    delta = 0.025
    gaussian_x = np.arange(-5.0, 5.0, delta)
    gaussian_y = np.arange(-5.0, 5.0, delta)
    mesh_X, mesh_Y = np.meshgrid(gaussian_x, gaussian_y)
    mesh_xy = np.concatenate((mesh_X.reshape((-1,1)), mesh_Y.reshape((-1,1))), axis=1)
    x = T.matrix()
    E_func = theano.function([x], energy.E(x), allow_input_downcast=True)
    mesh_Z = E_func(mesh_xy).reshape(mesh_X.shape)

    plt.subplot(2,2,4)
    gaussian_Contour =plt.contour(mesh_X,mesh_Y, mesh_Z, 14, alpha=0.3)
    plt.subplot(2,2,3)
    gaussian_Contour =plt.contour(mesh_X,mesh_Y, mesh_Z, 14, alpha=0.3)

    color_map = itertools.cycle(['b','r', 'k', 'g', 'c', 'm', 'y'])
        
    
    """
    set up x_axis, i.e., number of samples
    """
    n_sample_list = np.exp(np.linspace(np.log(20), np.log(max_samplecount), num_samplecounts)).astype(int)

    estimated_samples = defaultdict(list)
    independent_samples=defaultdict(list)
    train_objective = []
    # compile the training objective
    objective = training_objective.training_objective(energy, stats_dict)
    for n_sample in n_sample_list:
        # print "processing sample = ", n_sample
        n_dim=2
        random_stepsizes = rng.rand(n_sample)
        random_interval = 1.5*random_stepsizes-1
        stepsize_baseline = 0.2
        # stepsize_baseline = 0.1
        noise_level = 2
        decay_rates0 = 0.5*np.ones(n_sample)
        stepsizes0 = stepsize_baseline*noise_level**random_interval 
       
        initial_v = rng.randn(n_sample, n_dim)
        samples_true = energy.generate_samples(n_sample)

        initial_params = rng.randn(n_sample, n_dim)
        # initial_params = rng3.uniform(size=(n_sample, n_dim))*10. - 5.
        if true_init:
           initial_params = samples_true.copy()
        initial_params_flat = initial_params.flatten()
        num_passes = 500
        decay_alg = 0.9
        learning_rate_alg = 0.5
        alg_params = [decay_alg, learning_rate_alg, num_passes]    
        """
        args_hyper is the set of hyperparameters: initial momentum, stepsizes, number of samplers, n_sample and n_dim
        """
        args_hyper = [initial_v, decay_rates0, stepsizes0, n_steps, n_sample,2]
        
        #best_samples_list = scipy.optimize.fmin_l_bfgs_b(objective.f_df_wrapper, 
         #                           initial_params_flat,
         #                           args = args_hyper,
         #                           maxfun=500,
                                    # disp=1,
          #                          )
        best_samples, final_cost = RMSprop(objective, alg_params, initial_params_flat.copy(), args_hyper)
        train_objective.append(final_cost)

        # DEBUG again, with new velocity and step sizes
        for iii in range(0): #10): # DEBUG
            initial_v = rng.randn(n_sample, n_dim)
            random_stepsizes = rng.rand(n_sample)
            random_interval = 1.5*random_stepsizes-1
            stepsize_baseline = 0.2
            noise_level = 2
            stepsizes0 = stepsize_baseline*noise_level**random_interval 
            args_hyper = [initial_v, stepsizes0, n_steps, n_sample,2]
            
            best_samples_list = scipy.optimize.fmin_l_bfgs_b(objective.f_df_wrapper, 
                                    best_samples_list[0],
                                    args = args_hyper,
                                    maxfun=200,
                                    disp=1,
                                    )
            
            

        #best_samples = best_samples_list[0].reshape(n_sample, n_dim)
        
        #train_objective.append(best_samples_list[1])
        """
        we only draw scatter plot for last run
        TODO this is a super hacky way to do this. one better option would be to store outputs for all numbers of samples in an array, and then call this as a function with the last element of that array.
        """
        if n_sample==n_sample_list[-1]:
            nps = 100
            initial_draw = initial_params
            plt.subplot(2,2,3)
            plt.scatter(samples_true[:nps,0], samples_true[:nps,1], s=10, marker='+', color='blue', alpha=0.6, label='Independent')
            plt.scatter(best_samples[:nps,0], best_samples[:nps,1],s=10, marker='*', color='red', alpha=0.6, label='Characteristic' )
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.legend(loc='upper right')
            plt.title('Samples')

            all_pos, all_vel, all_noise, final_pos = objective.f_samples(best_samples, *args_hyper[:4])
            plt.subplot(2,2,4)
            for traj_i in range(4):
                color_current = next(color_map)
                plt.plot(all_pos[:,traj_i,0], all_pos[:,traj_i,1], markersize=10, marker='x', label='trajectory %d'%traj_i, color = color_current)
                # add the starting point
                plt.plot(all_pos[0,traj_i,0], all_pos[0,traj_i,1], markersize=10, marker='o', label='trajectory %d'%traj_i, color = 'black')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title('Example LMC trajectories')
     
            # tmp = np.concatenate((initial_pos_vec, final_pos_vec), axis=1)
            # tmp = tmp.reshape((n_steps-1, int(n_sample), 4))
            # tmp = np.transpose(tmp, [1,0,2])
            # tmp = tmp.reshape(-1, 4)
            # plt.figure()
            # plt.imshow(tmp[:150], interpolation='nearest', aspect='auto')
            # # plt.colorbar()
            # plt.savefig('delme.pdf')
            # plt.close()
        
        for stat_name in sorted(stats_dict.keys()):
            xx = T.fmatrix()
            yy = stats_dict[stat_name](xx, T.ones_like(xx)/xx.shape[0].astype(theano.config.floatX))
            stat_func = theano.function([xx], yy, allow_input_downcast=True)

            # mean here is over dimensions of stats output, NOT over samples
            # mean over samples is taken internally in the stat
            estimated_samples[stat_name].append(np.mean(stat_func(best_samples)))
            independent_samples[stat_name].append(np.mean(stat_func(samples_true)))

    for stat_name in sorted(stats_dict.keys()):
        estimated_samples[stat_name] = np.asarray(estimated_samples[stat_name])
        independent_samples[stat_name] = np.asarray(independent_samples[stat_name])
    train_objective = np.asarray(train_objective)

    if true_init:
        print "true init, ",

    plt.subplot(2,2,1)
    plt.xscale('log')
    plt.yscale('log')
    for stat_name in sorted(stats_dict.keys()):
        color_current = next(color_map)
        plt.plot(n_sample_list, estimated_samples[stat_name], '.', markersize=12, marker='o', label='Characteristic ' + stat_name, color = color_current)
        plt.plot(n_sample_list, independent_samples[stat_name], '.', markersize=12, marker='+', label='Independent ' + stat_name, color = color_current)
        print "%s char %g ind %g, "%(stat_name, estimated_samples[stat_name][-1], independent_samples[stat_name][-1]),
    print
    plt.xlabel('# samples')
    plt.ylabel('Value')
    plt.title('Target statistic')
    plt.legend(loc='upper left')
    #plt.ylim([0,4]) # TODO don't hardcode this
    #plt.legend(bbox_to_anchor=(0.,1.02, 1.,.102),loc=3,ncol=1,mode='expand', borderaxespad=0.)
           
    true_values = estimated_samples[-1]

    plt.subplot(2,2,2)
    plt.xscale('log') 
    plt.yscale('log')
    # plt.plot(n_sample_list, ((estimated_samples-true_values)**2), color='red', label='true')  
    plt.plot(n_sample_list, train_objective, color='black', label="object.")
    plt.xlabel('# samples')
    plt.ylabel('RMS error')
    plt.title('Objective')
    # plt.legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(0.,1.02, 1.,.102),loc=3,ncol=1,mode='expand', borderaxespad=0.)
    
    #plt.subplot(3,1,3)
    plt.tight_layout()
    plt.show() 
    plt_name = 'long_' + energy.name + '-' + '_'.join(str(elem) for elem in sorted(stats_dict.keys())) 
    if true_init:
       plt_name += "_true-init"
    plt_name += '.pdf'
    plt.savefig(plt_name)
    plt.close()
    
energy_2d = energies.gauss_2d()
base_stats = {
    'mean': lambda x:x,
    'sqr': lambda x: x**2,
    'third': lambda x: x**3,
    'exp': lambda x:T.exp(x),
    'sin': lambda x:T.sin(x),
    'sqrt': lambda x: T.sqrt(x**2 + 1e-5),
    'log2': lambda x: T.log(1. + x**2),
    'abs': lambda x: T.abs_(x),
    'sqrt_inv': lambda x: x/T.sqrt(x**2 + 1e-5),
    'inv_sqr': lambda x: 1./(x**2),
    'inv_abs': lambda x: 1./T.abs_(x)
}
start_time = timeit.default_timer()
for base_stat_name in base_stats:
    stat_dict = {}
    stat_dict[base_stat_name] = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)
    generate_plot(energy_2d, stat_dict, 2)
end_time = timeit.default_timer()
print "compiling time = ", end_time-start_time    