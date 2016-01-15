# -*- coding: utf-8 -*-
"""
Created on Fri Jan 01 14:50:02 2016
test Bayesian Logistic Regression
@author: tian
"""

import theano
import theano.tensor as T

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from minimizer_wt_annealing import RMSprop
import energies
#import training_objective
#import training_objective_hmc
#import training_objective_hmc1
#import training_objective_updateLMC
import training_objective_BayLogReg
from load_data import load_german_credit, load_pima_indian, load_heart, load_australian_credit

#energy_2d = energies.gauss_2d()
#energy_nd = energies.gauss_nd(50)
#energy_laplacePixel = energies.laplace_pixel()
"""
n_dim is # of predictors + 1, for australian data, its 15, for german credit, its 25
"""
n_dim = 15
energy_BayLogReg = energies.BayeLogReg()
"""
load the data and the label 
"""
#germanData, germanLabel = load_german_credit();
#germanData, germanLabel = load_pima_indian();
#germanData, germanLabel = load_heart();
germanData, germanLabel = load_australian_credit();
# normalize to let each dimension have mean 1 and std 0
g_mean = np.mean(germanData,axis=0)
g_std = np.std(germanData, axis=0)
germanData = (germanData - g_mean) / g_std
#g_mean = np.mean(germanData, axis=1)
#g_std = np.std(germanData, axis=1)
#germanData = (germanData - g_mean[:, np.newaxis]) / g_std[:, np.newaxis]
"""
set up necessary params
"""
rng = np.random.RandomState(12)
n_sample = 50

"""
for n_steps is large, grad is quite large, even inf
"""
n_steps = 300
#n_steps_hmc = 100
true_init = False

random_stepsizes = rng.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.2
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 

#stepsize_baseline_hmc = 0.2
#stepsizes0_hmc = stepsize_baseline_hmc*noise_level**random_interval
# can also set up the fixed stepsizes, uncomment the following line
#stepsizes0 = 0.2*np.ones(n_sample)
decay_rates0 = 0.9*np.ones(n_sample)
initial_v = rng.randn(n_sample, n_dim) # 1 for alpha
#initial_v = np.zeros((n_sample, n_dim))
args_hyper_lmc = [germanData, germanLabel, initial_v, decay_rates0, stepsizes0, n_steps, n_sample,n_dim] 
#args_hyper_hmc = [initial_v, stepsizes0_hmc, n_steps_hmc, n_sample, n_dim]

num_passes = 1000
decay_alg = 0.9
learning_rate_alg = 5
alg_params = [decay_alg, learning_rate_alg, num_passes]    
#alg_params = [num_passes]

#samples_true = energy_nd.generate_samples(50000)
#samples_true = energy_nd.generate_samples(n_sample)
initial_params = rng.randn(n_sample, n_dim)
if true_init:
   initial_params = samples_true.copy()
initial_params_flat = initial_params.flatten()

base_stats = {
    'mean': lambda x:x,
    #'sqr': lambda x: x**2
    #'sqrt': lambda x: T.sqrt(x**2 + 1e-5),
    #'sqrt_inv': lambda x: x/T.sqrt(x**2 + 1e-5)
    #'log2': lambda x: T.log(1. + x**2)
    #'abs': lambda x: T.abs_(x)
    #'third': lambda x: x**3
    #'sin': lambda x:T.sin(x)
    #'exp': lambda x: T.exp(x)
    #'inv_sqr': lambda x: 1./(x**2),
    #'inv_abs': lambda x: 1./T.abs_(x)
    #'gradient': lambda x: energy_laplacePixel.dE_dtheta(x)
}
stat_dict = {}
for base_stat_name in base_stats:
  
    stat_dict[base_stat_name] = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)
#energy_grad = lambda x: ener.dE_dtheta(x)
#stat_dict['gradient'] = energy_grad
"""
do Langevin dynamic
"""
objective_lmc = training_objective_BayLogReg.training_objective(energy_BayLogReg, stat_dict)
best_samples, f_cost, sample_GE = RMSprop(objective_lmc, alg_params, initial_params_flat.copy(), args_hyper_lmc)
#objective_lmc = training_objective.training_objective(energy_2d, stat_dict)
#best_samples, f_cost = RMSprop(objective_lmc, alg_params, initial_params_flat.copy(), args_hyper_lmc)
#best_samples, f_cost = LBFGS(objective_lmc,alg_params, initial_params_flat.copy(), args_hyper_lmc )
"""
do Hamiltonian dynamic using the new version
"""
#objective_hmc = training_objective_hmc.training_objective(energy_2d, stat_dict)
#best_samples_hmc = RMSprop(objective_hmc, alg_params, initial_params_flat.copy(), args_hyper_hmc)

"""
do Hamiltonian dynamic using the old version
"""
#objective_hmc_old = training_objective_hmc1.training_objective(energy_2d, stat_dict)
#best_samples_hmc_old = RMSprop(objective_hmc_old, alg_params, initial_params_flat.copy(), args_hyper_hmc)
#best_samples = LBFGS(objective, alg_params, initial_params_flat.copy(), args_hyper)
"""
draw 2D contour and the sample positions in different stages
"""
"""
plt.figure()
delta = 0.025
gaussian_x = np.arange(-5.0, 5.0, delta)
gaussian_y = np.arange(-5.0, 5.0, delta)
mesh_X, mesh_Y = np.meshgrid(gaussian_x, gaussian_y)
mesh_xy = np.concatenate((mesh_X.reshape((-1,1)), mesh_Y.reshape((-1,1))), axis=1)
x = T.matrix()
E_func = theano.function([x], energy_2d.E(x), allow_input_downcast=True)
mesh_Z = E_func(mesh_xy).reshape(mesh_X.shape)
gaussian_Contour =plt.contour(mesh_X,mesh_Y, mesh_Z, 14, alpha=0.3)
"""
"""
nps = 100
plt.scatter(samples_true[:nps,0], samples_true[:nps,1], s=10, marker='+', color='blue', alpha=0.6, label='Independent')
plt.scatter(best_samples[:nps,0], best_samples[:nps,1],s=10, marker='*', color='red', alpha=0.6, label='Characteristic Langevin' )
#plt.scatter(best_samples_hmc[:nps,0], best_samples_hmc[:nps,1], s=10, marker = '*', color='yellow', alpha=0.6, label = 'Characteristic HMC')
#plt.scatter(best_samples_hmc_old[:nps,0], best_samples_hmc_old[:nps,1], s=10, marker = '*', color='green', alpha=0.6, label = 'Characteristic HMCd')
plt.scatter(initial_params[:nps,0], initial_params[:nps,1], s=10, marker='o', color='black', alpha=0.6, label='Initial')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='upper right')
plt.title('Samples')
"""
"""
get numerical measure
"""
estimated_samples = defaultdict(list)
independent_samples=defaultdict(list)
estimated_samples_hmc = defaultdict(list)
estimated_samples_hmc_old = defaultdict(list)
for stat_name in sorted(stat_dict.keys()):
    xx = T.fmatrix()
    yy = stat_dict[stat_name](xx, T.ones_like(xx)/xx.shape[0].astype(theano.config.floatX))
    stat_func = theano.function([xx], yy, allow_input_downcast=True)

    # mean here is over dimensions of stats output, NOT over samples
    # mean over samples is taken internally in the stat
    estimated_samples[stat_name].append(np.mean(stat_func(best_samples)))
    #estimated_samples_hmc[stat_name].append(np.mean(stat_func(best_samples_hmc)))
   # estimated_samples_hmc_old[stat_name].append(np.mean(stat_func(best_samples_hmc_old)))
    #independent_samples[stat_name].append(np.mean(stat_func(samples_true)))

for stat_name in sorted(stat_dict.keys()):
    estimated_samples[stat_name] = np.asarray(estimated_samples[stat_name])
    #estimated_samples_hmc[stat_name] = np.asarray(estimated_samples_hmc[stat_name])
   # estimated_samples_hmc_old[stat_name] = np.asarray(estimated_samples_hmc_old[stat_name])
    #independent_samples[stat_name] = np.asarray(independent_samples[stat_name])
print "lmc estimated = ", estimated_samples['mean']
print "mean lmc = ", np.mean(best_samples  ,axis=0)
#print "mean true = ", np.mean(samples_true**2  , axis=0)
#print "hmc estimated = ", estimated_samples_hmc['mean']
#print "mean hmc = ", np.mean(best_samples_hmc, axis=0)
#print "hmc old estimated = ", estimated_samples_hmc_old['mean'] 
#print independent_samples['sqr']