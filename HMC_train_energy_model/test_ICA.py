# -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 12:14:31 2015
test the training algorithm for ICA model. For visulization, use the function provided in 
http://deeplearning.net/tutorial/utilities.html#how-to-plot
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
objective = training_objective.training_objective(ica_soft, 'pixel_sparse', 1)

"""
preparing the initial parameters for the algorithms
"""
train_x = objective.training_X

n_steps = 50 
imwidth = 4
n_dim = imwidth**2
n_sample = 1000
random_stepsizes = rng.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.2
        # stepsize_baseline = 0.1
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 

# DEBUG
train_x = train_x[:n_sample]
       
initial_v = rng.randn(n_sample, n_dim)

initial_params = rng.randn(n_sample+n_dim, n_dim)
initial_params[n_sample:] = initial_params[n_sample:] / np.sqrt(n_dim) * np.sqrt(n_sample)
initial_params_flat = initial_params.flatten()

#params_original = []
#params_original.append(initial_params_flat[:(n_sample*n_dim)].reshape(n_sample, n_dim)) # for representative samples
#params_original.append(initial_params_flat[(n_sample*n_dim):].reshape(n_dim, n_dim)) # f
args_hyper = [train_x, initial_v, stepsizes0, n_steps, n_sample,n_dim]
best_samples_list = scipy.optimize.fmin_l_bfgs_b(objective.f_df_wrapper, 
                                    initial_params_flat,
                                    args = args_hyper,
                                    maxfun=1000,
                                    #disp=1,
                                    )
                                    
optimal_param = best_samples_list[0]
best_samples = optimal_param[:(n_sample*n_dim)].reshape(n_sample, n_dim)
J = optimal_param[(n_sample*n_dim):].reshape(n_dim, n_dim)
J_inv = linalg.inv(J)
receptive_field = utils.tile_raster_images(J, (imwidth,imwidth),(10,10), (1,1))
image = Image.fromarray(receptive_field).resize((imwidth*10*8, imwidth*10*8))
image.save('J_patches100000_step50_nsample1000.png')     
receptive_field_inv = utils.tile_raster_images(J_inv, (imwidth,imwidth),(10,10), (1,1))
image1 = Image.fromarray(receptive_field_inv).resize((imwidth*10*8, imwidth*10*8))
image1.save('J-inv_patches100000_step50_nsample1000.png')     
samples_vis = utils.tile_raster_images(best_samples, (imwidth,imwidth),(10,10), (1,1))  
samples_vis_image = Image.fromarray(samples_vis).resize((imwidth*10*8, imwidth*10*8))
samples_vis_image.save('representative samples.png')
train_x_sub = train_x[:100, ]
train_x_show = utils.tile_raster_images(train_x_sub, (imwidth,imwidth),(10,10), (1,1))
image_train = Image.fromarray(train_x_show).resize((imwidth*10*8, imwidth*10*8))
image_train.save('training_samples.png')
