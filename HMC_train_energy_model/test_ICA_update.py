# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:05:21 2015
test and visualize ICA using different algorithms.(LBFGS, SGD, SGD_MOMENTUM)
For visualization, using http://deeplearning.net/tutorial/utilities.html#how-to-plot
For SGD and SGD_MOMENTUM , using https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer/blob/master/generate_figures/optimization_wrapper.py

@author: tian
"""

import numpy as np
from energies import ICA_soft_laplace
import training_objective
import scipy.optimize
import utils
from PIL import Image
import scipy.linalg as linalg
rng = np.random.RandomState(123)

ica_soft = ICA_soft_laplace()
objective = training_objective.training_objective(ica_soft, 'van_hateren', 100)

"""
preparing the initial parameters for the algorithms
"""
#train_x = objective.training_X
#mini_batches_x = objective.mini_batches

n_steps = 50 
n_dim = 10 * 10
n_sample = 100
random_stepsizes = rng.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.2
        # stepsize_baseline = 0.1
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 
       
initial_v = rng.randn(n_sample, n_dim)

initial_params = rng.randn(n_sample+n_dim, n_dim)
initial_params[n_sample:] = initial_params[n_sample:] / np.sqrt(n_dim)
initial_params_flat = initial_params.flatten()
args_hyper = [initial_v, stepsizes0, n_steps, n_sample, n_dim]

def visualize(best_params, best_samples, alg_name, tile_shape_sample, filter_shape=(10,10), tile_shape_J = (10,10), spacing = (1,1)):
    J = best_params
    J_inv = linalg.inv(J) 
    n_sample = best_samples.shape[0]
    save_name = '-' + 'nsamples' + str(n_sample) + '-'+ alg_name + '.png'
    receptive_field = utils.tile_raster_images(J, filter_shape,tile_shape_J, spacing)
    image_rf = Image.fromarray(receptive_field)
    rf_name = 'J' + save_name
    image_rf.save(rf_name)     
    receptive_field_inv = utils.tile_raster_images(J_inv, filter_shape,tile_shape_J, spacing)
    image1 = Image.fromarray(receptive_field_inv)  
    rf_inv_name = 'J_inv' + save_name
    image1.save(rf_inv_name)     
    samples_vis = utils.tile_raster_images(best_samples, filter_shape,tile_shape_sample, spacing)  
    samples_vis_image = Image.fromarray(samples_vis)
    samples_vis_image.save('representative samples.png')

def LBFGS(objective, initial_params, args_hyper):
    train_x = [objective.training_X]
    n_sample = args_hyper[-2]
    n_dim = args_hyper[-1]
    args_hypers = train_x + args_hyper
    best_samples_list = scipy.optimize.fmin_l_bfgs_b(objective.f_df_wrapper, 
                                    initial_params.copy(),
                                    args = args_hypers,
                                    maxfun=1000,
                                    # disp=1,
                                    )
    optimal_params = best_samples_list[0]
    best_samples = optimal_params[:(n_sample*n_dim)].reshape(n_sample, n_dim)
    J = optimal_params[(n_sample*n_dim):].reshape(n_dim, n_dim)
    visualize(J, best_samples, 'lbfgs', (50, int(n_sample/50)))
    
def SGD(objective, initial_params, args_hyper):
    mini_batches_x = objective.mini_batches
    n_sample = args_hyper[-2]
    n_dim = args_hyper[-1]
    N = len(mini_batches_x)
    eta = 0.1
    num_passes = 20
    params_i = initial_params.copy()
    for _ in range(num_passes*N):
        batch_idx = np.random.randint(N)
        batch_i = [mini_batches_x[batch_idx]]
        args_hypers_i = batch_i + args_hyper
        f_i, df_i = objective.f_df_wrapper(params_i, *args_hypers_i)
        params_i -= df_i * eta
        if not np.isfinite(f_i):
            print("Non-finite func")
            break;
    optimal_params = params_i
    best_samples = optimal_params[:(n_sample*n_dim)].reshape(n_sample, n_dim)
    J = optimal_params[(n_sample*n_dim):].reshape(n_dim, n_dim)
    visualize(J, best_samples, 'sgd', (50, int(n_sample/50)))

def SGD_MOMENTUM(objective, initial_params, args_hyper):
    mini_batches_x = objective.mini_batches
    n_sample = args_hyper[-2]
    n_dim = args_hyper[-1]
    momentum = 0.5
    eta = 0.1
    N = len(mini_batches_x)
    f_eval = np.ones(N)*np.nan
    params_i = initial_params.copy()
    inc = 0.0
    num_passes = 20
    for ipass in range(num_passes):
        for ibatch in range(N):
            batch_idx = np.random.randint(N)
            batch_i = [mini_batches_x[batch_idx]]
            args_hypers_i = batch_i + args_hyper
            f_i, df_i = objective.f_df_wrapper(params_i, *args_hypers_i)
            inc = momentum * inc - eta * df_i
            params_i += inc
            f_eval[batch_idx] = f_i
            if not np.isfinite(f_i):
                print("Non-finite func")
                break
        if not np.isfinite(f_i):
            print("Non-finite func. Exit running")
            break
    print(np.mean(f_eval[np.isfinite(f_eval)]), "average finite at last")
    optimal_params = params_i
    best_samples = optimal_params[:(n_sample*n_dim)].reshape(n_sample, n_dim)
    J = optimal_params[(n_sample*n_dim):].reshape(n_dim, n_dim)
    visualize(J, best_samples, 'sgd_momentum', (50, int(n_sample/50)))
    
    
    
    
    
LBFGS( objective,initial_params_flat, args_hyper )   
SGD(objective,initial_params_flat, args_hyper )
SGD_MOMENTUM(objective,initial_params_flat, args_hyper)
"""
visualize the some of the training samples
"""
train_x_sub = objective.training_X[:100, ]
train_x_show = utils.tile_raster_images(train_x_sub, (10,10),(10,10), (1,1))
image_train = Image.fromarray(train_x_show)
image_train.save('training_samples.png')