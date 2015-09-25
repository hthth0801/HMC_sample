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
import itertools

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



def generate_plot(energy, stats_dict, ndim=2, true_init=False,
    num_samplecounts=25, max_samplecount=20000,
    # num_samplecounts=10, max_samplecount=50,
    n_steps=100,
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
        stepsizes0 = stepsize_baseline*noise_level**random_interval 
       
        initial_v = rng.randn(n_sample, n_dim)
        samples_true = energy.generate_samples(n_sample)

        initial_params = rng.randn(n_sample, n_dim)
        # initial_params = rng3.uniform(size=(n_sample, n_dim))*10. - 5.
        if true_init:
           initial_params = samples_true.copy()
        initial_params_flat = initial_params.flatten()
        """
        args_hyper is the set of hyperparameters: initial momentum, stepsizes, number of samplers, n_sample and n_dim
        """
        args_hyper = [initial_v, stepsizes0, n_steps, n_sample,2]
        best_samples_list = scipy.optimize.fmin_l_bfgs_b(objective.f_df_wrapper, 
                                    initial_params_flat,
                                    args = args_hyper,
                                    maxfun=200,
                                    # disp=1,
                                    )


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


        best_samples = best_samples_list[0].reshape(n_sample, n_dim)
        
        train_objective.append(best_samples_list[1])
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

            initial_pos_vec, final_pos_vec, final_pos = objective.f_samples(best_samples, *args_hyper[:3])
            plt.subplot(2,2,4)
            for traj_i in range(4):
                color_current = next(color_map)
                plt.plot(final_pos[:,traj_i,0], final_pos[:,traj_i,1], markersize=10, marker='x', label='trajectory %d'%traj_i, color = color_current)
                # add the starting point
                plt.plot(final_pos[0,traj_i,0], final_pos[0,traj_i,1], markersize=10, marker='o', label='trajectory %d'%traj_i, color = 'black')
            plt.title('Example HMC trajectories')
     
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
        plt.plot(n_sample_list, independent_samples[stat_name], '.', markersize=12, marker='+', label='Independent ' + stat_name, color = color_current)
        plt.plot(n_sample_list, estimated_samples[stat_name], '.', markersize=12, marker='o', label='Characteristic ' + stat_name, color = color_current)
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
        

energy = energies.gauss_2d()
        
# each stats function takes an input with [# samples]x[# data dimensions], and produces an output which is 
# [# stats]

# stats can be multidimensional, in which case the different stats should lie along the second dimension
# ie stats input is [# samples]x[# data dimensions] and stats output is [# samples]x[# stats]



# ## this is the simple way to set a couple stats and run with them:
# stats = {
#     'mean':lambda x, w: T.sum(w*x, axis=0),
#     'E':lambda x, w: T.sum(w*energy.E(x).reshape((-1,1))),
#     }
# generate_plot(energy, stats)


base_stats = {
    'mean': lambda x: x,
    'sqr': lambda x: x**2,
    'exp': lambda x: T.exp(x),
    'sin': lambda x: T.sin(x),
    'cube':lambda x: x**3,
    'margcube':lambda x: T.mean(x**3, axis=1).reshape((-1,1)),
}






for base_stat_name in base_stats:
    # basic dictionary for this stat
    stat_dict = {}
    stat_dict[base_stat_name] = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)
    # single stat plot
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)

    ## TODO try regularizing with gradient of stat
    # stat_dict = {}
    # stat_dict[base_stat_name] = lambda x, w: T.sum(
    #     w*base_stats[base_stat_name](x),
    #     axis=0)
    # stat_grad = lambda x: theano.gradient.jacobian()
    # stat_dict[base_stat_name+'_grad'] = lambda x, w: T.sum(
    #     w*base_stats[base_stat_name](x),
    #     axis=0)


    ## try regularizing in terms of energy moments around the mean
    stat_dict = {}
    stat_dict[base_stat_name] = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)
    E_mn = lambda x, w: T.sum(
            w*energy.E(x).reshape((-1,1))
            )
    stat_dict['zEmn'] = E_mn
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)
    # second order
    stat_dict['zEsd'] = lambda x, w: T.sum(
            w*(energy.E(x) - E_mn(x, w)).reshape((-1,1))**2
            )**(1./2.)
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)


    ## try regularizing in terms of target statistic moments around the mean
    stat_dict = {}
    stat_mn = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)
    stat_dict[base_stat_name] = stat_mn
    stat_dict[base_stat_name + '_2nd'] = lambda x, w: T.sum(
        w*(base_stats[base_stat_name](x) - stat_mn(x, w))**2,
        axis=0)**(1./2.)
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)


    ## try regularizing with both at once
    stat_dict = {}
    stat_mn = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)
    stat_dict[base_stat_name] = stat_mn
    stat_dict[base_stat_name + '_2nd'] = lambda x, w: T.sum(
        w*(base_stats[base_stat_name](x) - stat_mn(x, w))**2,
        axis=0)**(1./2.)
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)
    E_mn = lambda x, w: T.sum(
            w*energy.E(x).reshape((-1,1))
            )
    stat_dict['zEmn'] = E_mn
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)
    # second order
    stat_dict['zEsd'] = lambda x, w: T.sum(
            w*(energy.E(x) - E_mn(x, w)).reshape((-1,1))**2
            )**(1./2.)
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)


if False:

    ## try regularizing by adding stats to match norms of the energy function
    for E_order in range(1,5):
        stat_dict['zE%d'%E_order] = lambda x, w: T.sum(
            w*abs(energy.E(x)).reshape((-1,1))**E_order
            )**(1./E_order)
        generate_plot(energy, stat_dict)
        generate_plot(energy, stat_dict, true_init=True)

    ## try regularizing by comparing against perturbations of the same stat
    # basic dictionary for this stat
    stat_dict = {}
    stat_dict[base_stat_name] = lambda x, w: T.sum(w*base_stats[base_stat_name](x), axis=0)
    p_mag = 1.2 # magnitude of perturbations to stat
    # add stat perturbed slightly larger
    stat_dict['zper up'] = lambda x, w: T.sum(
        w*abs(base_stats[base_stat_name](x))**p_mag,
        axis=0
        )**(1./p_mag)
    # add stat perturbed slightly smaller
    stat_dict['zper down'] = lambda x, w: T.sum(
        w*abs(base_stats[base_stat_name](x))**(1./p_mag),
        axis=0
        )**p_mag
    generate_plot(energy, stat_dict)
    generate_plot(energy, stat_dict, true_init=True)
