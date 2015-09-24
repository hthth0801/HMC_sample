# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:29:32 2015

@author: tian
"""
import numpy as np
import timeit
import scipy.optimize
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from collections import defaultdict

import energies
import training_objective

"""
comments:

General comments:
The code is wonderfully well commented, and easy to read. Nice job! Good variable names too.
Make the code much, much, more modular! Functions should be very short, and only do a single thing.
To the extent possible, avoid special cases and long strings of if statements.
The more the code can be written with simple short functions that make few assumptions about their input, the easier it will be to extend and run new experiments, and the less likely bugs are.
For instance, it's much easier to run experiments if both the energy and statistics are passed as functions.
Really bad is when special cases have to be handled separately in multiple locations. This makes the code very brittle, and invites bugs where a change is made in one place but not the other.
examples in the code included:
- the various if statements evaluating strings listing the different target statistics to include
- that constraining the Hamiltonian was a special case, which could only occur in the second position, and had to be correctly handled in all locations
- that the Gaussian energy function was defined using two different functions (numpy and theano) that were in two widely separated regions of code. Much better to have a single implementation. If that was impossible though, the multiple implementations should be clustered together.
Don't give different variables the same name. One specific example was gaussian_2D, and gaussian_2d (this was even worse, because it was *almost* but not quite the same name)
Imports should all go at the top of the file, to make dependencies clear.
Always use the same number of spaces for indentation (4 spaces for indent level is good). I **strongly** recommend using an editor that does this for you. Sublime Text is a very popular and powerful recent one.
"""



def generate_plot(energy, stats_dict, ndim=2, true_init=False, num_samplecounts=25, max_samplecount=50000):
    # TODO break each subplot into its own function.

    rng = np.random.RandomState(1234)


    plt.figure(figsize=(13,5))
    plt.subplot(1,3,3)

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
    gaussian_Contour =plt.contour(mesh_X,mesh_Y, mesh_Z, 14, alpha=0.3)
    color_map = iter(['b','r', 'k', 'g', 'c', 'm', 'y'])
        
    
    """
    set up x_axis, i.e., number of samples
    """
    n_sample_list = np.exp(np.linspace(0, np.log(max_samplecount), num_samplecounts))

    estimated_samples = defaultdict(list)
    independent_samples=defaultdict(list)
    train_objective = []
    for n_sample in n_sample_list:
        print "processing sample = ", n_sample
        n_dim=2
        random_stepsizes = rng.rand(n_sample)
        random_interval = 1.5*random_stepsizes-1
        stepsize_baseline = 0.2
        noise_level = 2
        stepsizes0 = stepsize_baseline*noise_level**random_interval 
       
        initial_v = rng.randn(n_sample, n_dim)
        samples_true = energy.generate_samples(n_sample)

        initial_params = rng.randn(n_sample, n_dim)
        # initial_params = rng3.uniform(size=(n_sample, n_dim))*10. - 5.
        if true_init:
           print "initializing at true samples"
           initial_params = samples_true.copy()
        initial_params_flat = initial_params.flatten()
        """
        args_hyper is the set of hyperparameters: initial momentum, stepsizes, number of samplers, n_sample and n_dim
        """
        n_steps = 100
        args_hyper = [initial_v, stepsizes0, n_steps, n_sample,2]
        objective = training_objective.training_objective(energy, stats_dict)
        best_samples_list = scipy.optimize.fmin_l_bfgs_b(objective.f_df_wrapper, 
                                    initial_params_flat,
                                    args = args_hyper,
                                    maxfun=200)
        best_samples = best_samples_list[0].reshape(n_sample, n_dim)
        
        train_objective.append(best_samples_list[1])
        """
        we only draw scatter plot for last run
        TODO this is a super hacky way to do this. one better option would be to store outputs for all numbers of samples in an array, and then call this as a function with the last element of that array.
        """
        if n_sample==n_sample_list[-1]:
            nps = 100
            initial_draw = initial_params
            plt.scatter(samples_true[:nps,0], samples_true[:nps,1], s=4, marker='+', color='blue', alpha=0.6, label='Independent')
            plt.scatter(best_samples[:nps,0], best_samples[:nps,1],s=4, marker='*', color='red', alpha=0.6, label='Characteristic' )
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.legend(loc='upper right')
            plt.title('Samples')

        
        for stat_name in stats_dict.keys():
            xx = T.fmatrix()
            yy = stats_dict[stat_name](xx)
            stat_func = theano.function([xx], yy, allow_input_downcast=True)

            estimated_samples[stat_name].append(np.mean(stat_func(best_samples)))
            independent_samples[stat_name].append(np.mean(stat_func(samples_true)))

    for stat_name in stats_dict.keys():
        estimated_samples[stat_name] = np.asarray(estimated_samples[stat_name])
        independent_samples[stat_name] = np.asarray(independent_samples[stat_name])
    train_objective = np.asarray(train_objective)

    plt.subplot(1,3,1)
    plt.xscale('log')
    for stat_name in stats_dict.keys():
        color_current = next(color_map)
        plt.plot(n_sample_list, independent_samples[stat_name], '.', markersize=4, marker='+', alpha=0.6, label='Independent ' + stat_name, color = color_current)
        plt.plot(n_sample_list, estimated_samples[stat_name], '.', markersize=4, marker='*', alpha=0.6, label='Characteristic ' + stat_name, color = color_current)
    plt.xlabel('# samples')
    plt.ylabel('Value')
    plt.title('Target statistic')
    plt.legend(loc='upper left')
    #plt.legend(bbox_to_anchor=(0.,1.02, 1.,.102),loc=3,ncol=1,mode='expand', borderaxespad=0.)
           
    true_values = estimated_samples[-1]

    plt.subplot(1,3,2)
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
    plt_name = energy.name + '-' + '_'.join(str(elem) for elem in stats_dict.keys()) 
    if true_init:
       plt_name += "_true-init"
    plt_name += '.pdf'
    plt.savefig(plt_name)
    plt.close()
        

energy = energies.gauss_2d()
        
# stats can be multidimensional, in which case the different stats should lie along the second dimension
# ie stats input is [# samples]x[# data dimensions] and stats output is [# samples]x[# stats]

stats = {
    'mean':lambda x: x,
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'sqr':lambda x: x**2,
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'sqr':lambda x: x**2,
    'E':lambda x: energy.E(x).reshape((-1,1)),
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'sqr':lambda x: x**2,
    'E':lambda x: energy.E(x).reshape((-1,1)),
    'E2':lambda x: energy.E(x).reshape((-1,1))**2,
    }
generate_plot(energy, stats)
generate_plot(energy, stats, true_init=True)

stats = {
    'margcube':lambda x: T.mean(x**3, axis=1).reshape((-1,1)),
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'margcube':lambda x: T.mean(x**3, axis=1).reshape((-1,1)),
    'E':lambda x: energy.E(x).reshape((-1,1)),
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'cube':lambda x: x**3,
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'cube':lambda x: x**3,
    'E':lambda x: energy.E(x).reshape((-1,1)),
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'sin':lambda x: T.sin(x),
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)

stats = {
    'sin':lambda x: T.sin(x),
    'E':lambda x: energy.E(x).reshape((-1,1)),
    'E2':lambda x: energy.E(x).reshape((-1,1))**2,
    }
generate_plot(energy, stats)
#generate_plot(energy, stats, true_init=True)



