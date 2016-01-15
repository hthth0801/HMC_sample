# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:19:51 2016
get the gradient evaluation plot using the optimation sampling and pure LMC sampling
@author: tian
"""

import theano 
import theano.tensor as T

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from minimizer_wt_annealing import RMSprop

import energies
import training_objective # for nd gaussian
import training_objective_BayLogReg # for bayesian logistic regression
from load_data import load_german_credit, load_pima_indian, load_heart, load_australian_credit

"""
load the data (for bayesian logistic regression)
Normalize the data to ensure each dimension have mean 0 and variance 1
"""
trainData, trainLabel = load_german_credit()
d_mean = np.mean(trainData,axis=0)
d_std = np.std(trainData, axis=0)
trainData = (trainData - d_mean) / d_std

"""
initialize several hyperparamters for the LMC sampler and the RMSprop
"""
rng = np.random.RandomState(12)

n_dim = trainData.shape[1] + 1 
#n_dim = 50

n_sample = 5
n_steps = 100

random_stepsizes = rng.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.001
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 
decay_rates0 = 0.9*np.ones(n_sample)
initial_v = rng.randn(n_sample, n_dim)

args_hyper_lmc = [trainData, trainLabel, initial_v, decay_rates0, stepsizes0, n_steps, n_sample,n_dim] 


num_passes = 50
decay_alg = 0.9
learning_rate_alg = 1
alg_params = [decay_alg, learning_rate_alg, num_passes]    

initial_params = rng.randn(n_sample, n_dim) + 1.
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

energy_BayLogReg = energies.BayeLogReg()

objective_lmc = training_objective_BayLogReg.training_objective(energy_BayLogReg, stat_dict)
"""
run the optimization
"""
best_samples, f_cost, sample_GE = RMSprop(objective_lmc, alg_params, initial_params_flat.copy(), args_hyper_lmc)


"""
Next, we wanna compute the gradient evaluation by using LMC (we only consider one sample to reduce the burn-in period)
First, we need to prepare some initializations, we use the all the samples that after the burn-in period to estimate the statistics
""" 
n_sample_lmc = 1 
n_steps_lmc = 20000
burnIn_lmc = 100
sample_id_lmc = rng.randint(n_sample)
initial_params_lmc = initial_params[sample_id_lmc, :].reshape(n_sample_lmc, n_dim)
#stepsizes0_lmc = stepsizes0[sample_id_lmc].reshape(n_sample_lmc)
stepsizes0_lmc = 0.001*np.ones(n_sample_lmc)
initial_v_lmc = initial_v[sample_id_lmc,:].reshape(n_sample_lmc, n_dim)
decay_rates0_lmc = decay_rates0[sample_id_lmc].reshape(n_sample_lmc)
args_lmc = [initial_params_lmc, trainData, trainLabel, initial_v_lmc, decay_rates0_lmc, stepsizes0_lmc, n_steps_lmc]
 
"""
run the pure langevin monte carlo sampling without accept/reject adjustment
"""
all_p, all_v, all_n, final_pos = objective_lmc.f_samples(*args_lmc)


"""
construct the theano function to evaluate the testing statistic value
"""

for stat_name in sorted(stat_dict.keys()):
    xx = T.fmatrix()
    yy = stat_dict[stat_name](xx, T.ones_like(xx)/xx.shape[0].astype(theano.config.floatX))
    stat_func = theano.function([xx], yy, allow_input_downcast=True)
    sample_GE_list = sample_GE.items()
    sample_GE_order = OrderedDict(sorted(sample_GE_list))
    GradEval = []
    TestStat = []
    GradEval_lmc = []
    TestStat_lmc = []
    for i_ge in sample_GE_order.keys():
        GradEval.append(i_ge)
        TestStat.append(np.mean(stat_func(sample_GE_order[i_ge]))) 
        
    all_p_squeeze = all_p.squeeze()
    n_GE_lmc = 100
    for i_ge in range(n_GE_lmc):
    #for i_ge in range(n_steps_lmc - burnIn_lmc):
        gap = (n_steps_lmc - burnIn_lmc) / n_GE_lmc # we jump over #gap samples
        GradEval_lmc.append(burnIn_lmc + gap*(i_ge+1))
        TestStat_lmc.append(np.mean(stat_func(all_p_squeeze[burnIn_lmc:burnIn_lmc+gap*(i_ge+1), :])))
        
    plt.figure()    
    plt.plot(GradEval, TestStat, marker = 'o', label = 'Sampling by Optimizaion')
    plt.plot(GradEval_lmc, TestStat_lmc, marker = '*', color = 'red', label = 'Langevin Sampling')
    plt.xlabel('number of gradient evaluation')
    plt.ylabel('testing statistics')
    plt.title('GE vs testing statistics')
    plt.legend(loc='upper right')
    









