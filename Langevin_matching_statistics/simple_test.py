# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:49:25 2015
Simple test for the specific number of samples
@author: tian
"""
import theano
import theano.tensor as T

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools

from minimizer import RMSprop, LBFGS
import energies
import training_objective

"""
The optimization actually didn't work, possibly because we have a very small
epsilon in front of the parameter, readering the gradient be relatively small. 
Actually, the whole energy landscape is quite flat, maybe try some monotonic transformation 
e.g., exp() or log() to make the landscape more discernable? 
"""
energy_2d = energies.gauss_2d()

"""
set up necessary params
"""
rng = np.random.RandomState(12)
n_sample = 1000
n_dim = 2
n_steps = 300
true_init = False

random_stepsizes = rng.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.1   
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 
# can also set up the fixed stepsizes, uncomment the following line
#stepsizes0 = 0.01*np.ones(n_sample)
decay_rates0 = 0.5*np.ones(n_sample)
initial_v = rng.randn(n_sample, n_dim)
#initial_v = np.zeros((n_sample, n_dim))
args_hyper = [initial_v, decay_rates0, stepsizes0, n_steps, n_sample,2]

num_passes = 100
decay_alg = 0.9
learning_rate_alg = 0.05
alg_params = [decay_alg, learning_rate_alg, num_passes]    


samples_true = energy_2d.generate_samples(n_sample)

initial_params = rng.randn(n_sample, n_dim)
if true_init:
   initial_params = samples_true.copy()
initial_params_flat = initial_params.flatten()

base_stats = {
    'mean': lambda x:x,
   # 'sqr': lambda x: x**2
}
stat_dict = {}
for base_stat_name in base_stats:
  
    stat_dict[base_stat_name] = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)

objective = training_objective.training_objective(energy_2d, stat_dict)
best_samples = RMSprop(objective, alg_params, initial_params_flat.copy(), args_hyper)

"""
draw 2D contour and the sample positions in different stages
"""
delta = 0.025
gaussian_x = np.arange(-5.0, 5.0, delta)
gaussian_y = np.arange(-5.0, 5.0, delta)
mesh_X, mesh_Y = np.meshgrid(gaussian_x, gaussian_y)
mesh_xy = np.concatenate((mesh_X.reshape((-1,1)), mesh_Y.reshape((-1,1))), axis=1)
x = T.matrix()
E_func = theano.function([x], energy_2d.E(x), allow_input_downcast=True)
mesh_Z = E_func(mesh_xy).reshape(mesh_X.shape)
gaussian_Contour =plt.contour(mesh_X,mesh_Y, mesh_Z, 14, alpha=0.3)

nps = 100
plt.scatter(samples_true[:nps,0], samples_true[:nps,1], s=10, marker='+', color='blue', alpha=0.6, label='Independent')
plt.scatter(best_samples[:nps,0], best_samples[:nps,1],s=10, marker='*', color='red', alpha=0.6, label='Characteristic' )
plt.scatter(initial_params[:nps,0], initial_params[:nps,1], s=10, marker='o', color='black', alpha=0.6, label='Initial')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='upper right')
plt.title('Samples')

"""
get numerical measure
"""
estimated_samples = defaultdict(list)
independent_samples=defaultdict(list)
for stat_name in sorted(stat_dict.keys()):
    xx = T.fmatrix()
    yy = stat_dict[stat_name](xx, T.ones_like(xx)/xx.shape[0].astype(theano.config.floatX))
    stat_func = theano.function([xx], yy, allow_input_downcast=True)

    # mean here is over dimensions of stats output, NOT over samples
    # mean over samples is taken internally in the stat
    estimated_samples[stat_name].append(np.mean(stat_func(best_samples)))
    independent_samples[stat_name].append(np.mean(stat_func(samples_true)))

for stat_name in sorted(stat_dict.keys()):
    estimated_samples[stat_name] = np.asarray(estimated_samples[stat_name])
    independent_samples[stat_name] = np.asarray(independent_samples[stat_name])
print estimated_samples['mean']
print independent_samples['mean']