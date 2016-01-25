# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:54:42 2016
Test Log-Gaussian Cox Point Process
@author: user
"""

import theano
import theano.tensor as T

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from minimizer_wt_annealing import RMSprop
import energies

import training_objective_LGCPP


N = 64
n_dim = N**2
energy_LGCPP = energies.LGCPP(N)
true_X, true_Y = energy_LGCPP.generate_samples(N)

"""
set up necessary params
"""
rng = np.random.RandomState(12)
n_sample = 5

"""
for n_steps is large, grad is quite large, even inf
"""
n_steps = 400
#n_steps_hmc = 100
true_init = False

random_stepsizes = rng.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.1
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 

#stepsize_baseline_hmc = 0.2
#stepsizes0_hmc = stepsize_baseline_hmc*noise_level**random_interval
# can also set up the fixed stepsizes, uncomment the following line
#stepsizes0 = 0.2*np.ones(n_sample)
decay_rates0 = 0.99*np.ones(n_sample)
initial_v = rng.randn(n_sample, n_dim) # 1 for alpha
#initial_v = np.zeros((n_sample, n_dim))
args_hyper_lmc = [true_Y,  initial_v, decay_rates0, stepsizes0, n_steps, n_sample,n_dim] 
#args_hyper_hmc = [initial_v, stepsizes0_hmc, n_steps_hmc, n_sample, n_dim]

num_passes = 50
decay_alg = 0.9
learning_rate_alg = 1.
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
objective_lmc = training_objective_LGCPP.training_objective(energy_LGCPP, stat_dict)
best_samples, f_cost, sample_GE = RMSprop(objective_lmc, alg_params, initial_params_flat.copy(), args_hyper_lmc)


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
    estimated_samples[stat_name].append(np.mean(stat_func(best_samples)))
   
estimated_X = np.mean(best_samples, axis=0)
for stat_name in sorted(stat_dict.keys()):
    estimated_samples[stat_name] = np.asarray(estimated_samples[stat_name])
   
print "lmc estimated = ", estimated_samples['mean']
print "mean lmc = ", estimated_X
print "lmc true = ", true_X
print "mean true = ", np.mean(true_X)
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(true_X.reshape(N, N))
plt.subplot(2, 2, 2)
plt.imshow(estimated_X.reshape(N, N))
plt.subplot(2,2,3)
plt.imshow((np.exp(true_X)/(N**2)).reshape(N, N))
plt.subplot(2,2,4)
plt.imshow((np.exp(estimated_X)/(N**2)).reshape(N, N))
