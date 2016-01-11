# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 09:55:33 2016
test_pure langevin dynamics
@author: user
"""

import theano
import theano.tensor as T
import numpy as np

import training_objective_BayLogReg
import energies

from load_data import load_german_credit, load_pima_indian, load_heart, load_australian_credit


"""
prepare the numpy stuff
"""
australianData, australianLabel = load_australian_credit(); # energy.label
g_mean = np.mean(australianData,axis=0)
g_std = np.std(australianData, axis=0)
australianData = (australianData - g_mean) / g_std # energy.data

rng = np.random.RandomState(12)
n_sample = 1
n_dim = 15 # need to change this when use different dataset
n_steps = 5000. # n_step
initial_params = rng.randn(n_sample, n_dim) # initial_position
#random_stepsizes = rng.rand(n_sample)
#random_interval = 1.5*random_stepsizes-1
#stepsize_baseline = 0.2
#noise_level = 2
#stepsizes0 = stepsize_baseline*noise_level**random_interval # stepsizes
stepsizes0 = 0.08*np.ones(n_sample)
initial_v = rng.randn(n_sample, n_dim) #initial_v
decay_rates0 = 0.99*np.ones(n_sample) # decay_rates
args_lmc = [initial_params, australianData, australianLabel, initial_v, decay_rates0, stepsizes0, n_steps]


"""
prepare the theano stuff
"""
energy_BayLogReg = energies.BayeLogReg()
"""
may not need the following stat_dict. just wanna call f_samples inside the theano_funcs
"""
base_stats = {
    'mean': lambda x:x
    }
stat_dict = {}
for base_stat_name in base_stats:
  
    stat_dict[base_stat_name] = lambda x, w: T.sum(
        w*base_stats[base_stat_name](x),
        axis=0)
        
objective_lmc = training_objective_BayLogReg.training_objective(energy_BayLogReg, stat_dict)

all_p, all_v, all_n, final_pos = objective_lmc.f_samples(*args_lmc)
final_pos
"""
get the samples after the burn-in period
estimated: the estimated parameters
np.mean(estimated): just get the overall mean
"""
new_p = all_p[1000:]
estimated = np.mean(new_p, axis=0)
print "the estimated parameters are: (alpha is the last one) " ,estimated

print "the mean of all parameters ", np.mean(estimated)

"""
next we compute the hamiltonian in theano, and compute the accept prob. along the trajectory
"""
def kinetic_energy(vel):
    
    return 0.5 * (vel ** 2).sum(axis=1)
    
def hamiltonian(pos, vel):
   
    return energy_BayLogReg.E(pos) + kinetic_energy(vel)
    
th_pos = T.matrix('th_pos')
th_vel = T.matrix('th_vel')
func_h = theano.function([th_pos, th_vel, energy_BayLogReg.data, energy_BayLogReg.label], hamiltonian(th_pos, th_vel),name = 'func_hamiltonian', allow_input_downcast = True)
#res_h = func_h(all_p[0], all_v[0], australianData, australianLabel)

res_ham = [];
for i in range(all_p.shape[0]):
    res_ham.append(func_h(all_p[i], all_v[i], australianData, australianLabel))
 
res_h_np = np.asarray(res_ham)
res_hprev_np = np.zeros((res_h_np.shape[0]+1, n_sample))
res_hprev_np[1:] = res_h_np
init_h_np = func_h(initial_params, initial_v, australianData, australianLabel)
res_hprev_np[0] = init_h_np

deltaH = -res_hprev_np[:-1] + res_h_np # after - before
Ones = np.ones(deltaH.shape)
accept_prob = np.minimum(Ones, np.exp(-deltaH))
accept_np = accept_prob.flatten()
acc_f = (accept_np<=0.5)
acc_f[acc_f==True]=1.
acc_f[acc_f==False]=0.
print "the accept_prob ratio along the trajectory ", acc_f.sum() / n_steps
