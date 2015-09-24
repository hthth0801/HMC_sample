# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 15:15:21 2015

@author: tian
"""

import numpy
import theano
import theano.tensor as T
import hmc_sampling
import scipy.optimize
import scipy.linalg as linalg
import timeit

n_sample=1000
n_dim = 2
"""
Two sets of parameters: 
initial_pos: sample positions parameters (theano matrix: (n_sample, n_dim))
theta      : mean for the 2D gaussian (theano vector: (n_dim,))
"""
params = theano.shared(value=numpy.zeros(n_sample*n_dim+n_dim, dtype=theano.config.floatX), name='params', borrow=True)
initial_pos = params[0:n_sample*n_dim].reshape((n_sample, n_dim))
theta = params[n_sample*n_dim:]
    
"""
set up the ground truth mu and cov for 2D gaussian
"""  
rng = numpy.random.RandomState(123)
mu = numpy.array(rng.rand(n_dim)*5, dtype=theano.config.floatX)
rng1=numpy.random.RandomState(444)
cov = numpy.array(rng1.rand(n_dim,n_dim), dtype=theano.config.floatX)
cov = (cov+cov.T)/2.
cov[numpy.arange(n_dim), numpy.arange(n_dim)]=1.0
cov_inv = linalg.inv(cov)
   
print "begin process..."

"""
gaussian_energy(): returns the gaussian energy of our current model. 
  input:  x: theano matrix (x.shape[0]: different samples, x.shape[1]: different dim)    
  output: theano vector which should has size x.shape[0]
"""
def gaussian_energy(x):
    return 0.5 * (T.dot((x - theta), cov_inv) *
                      (x - theta)).sum(axis=1)
                      

#def dE_dtheta(x):
#    return T.dot((theta-x), cov_inv)

"""
dE_dtheta1(): returns the average or (weighted) dE/dtheta (scalar)
  input: x: theano matrix (x.shape[0]: different samples, x.shape[1]: different dim)
         acpt: theano vector which should have size x.shape[0]. Represents the accept
               prob. for different samples. 
  output: if acpt = None: return 1/|N| \sum dE/dtheta. N is the number of samples. (used for param_cost)
          otherwise:      return \sum acpt_i * dE_i/dtheta. (used for sampler_cost) 
"""   
def dE_dtheta1(x, acpt=None):
    if acpt == None:
       return T.grad(T.mean(gaussian_energy(x)), theta, consider_constant=[x])
    else:
       return T.grad(T.sum(acpt*gaussian_energy(x)), theta, consider_constant=[acpt, x])

observe = T.matrix('observe')       
initial_vel = T.matrix('initial_vel')     
#n_steps=30
n_step = T.iscalar('n_step')
stepsizes = T.vector('stepsizes')

# do HMC sampling
[accept,accept1, final_pos_new, final_pos_new1, ndeltaH] = hmc_sampling.hmc_move(initial_vel, initial_pos, gaussian_energy, stepsizes,n_step)

"""
reshape initial_pos, accept and final_pos_new
initial_pos : (n_sample, n_dim)----> (n_sample*n_steps, n_dim) n_steps is the number of different leapfrog sampler we consider
accept:       (n_steps, n_samples) ----> (n_steps*n_samples, )
final_pos_new: (n_steps, n_sample, n_dim) ---> (n_steps*n_sample, n_dim)
"""
initial_pos_vec = T.tile(initial_pos, [final_pos_new.shape[0],1])
accept_flatten = accept.flatten()
final_pos_new_flatten = T.reshape(final_pos_new, (final_pos_new.shape[0]*final_pos_new.shape[1],final_pos_new.shape[2]))

"""
define the sampler_cost
"""
sampler_cost = dE_dtheta1(initial_pos_vec, accept_flatten) - dE_dtheta1(final_pos_new_flatten, accept_flatten)
sampler_cost = 1./(final_pos_new.shape[0]) * sampler_cost
sampler_cost = T.mean(sampler_cost**2)

"""
define the param_cost
"""
param_cost = dE_dtheta1(observe) - dE_dtheta1(initial_pos)
param_cost = T.mean(param_cost**2)

total_cost = n_sample*param_cost + sampler_cost


"""
define the compilable function for evaluation and gradient computation.
"""    
start_time = timeit.default_timer()
func_eval = theano.function([observe, initial_vel,stepsizes, n_step], [total_cost, param_cost, sampler_cost], name='func_eval', allow_input_downcast=True)
func_grad = theano.function([observe, initial_vel,stepsizes, n_step], T.grad(total_cost, params), name='func_grad', allow_input_downcast=True)
end_time = timeit.default_timer()
print "compiling time= ", end_time-start_time

"""
define the different stepsizes
"""
random_stepsizes = numpy.random.rand(n_sample)
random_interval = 1.5*random_stepsizes-1
stepsize_baseline = 0.2
noise_level = 2
stepsizes0 = stepsize_baseline*noise_level**random_interval 
initial_random_vel = theano.shared(numpy.zeros([n_sample,n_dim], dtype=theano.config.floatX), name='rv', borrow=True)

"""
set up the random momentum used in hmc sampler
"""
initial_v = numpy.random.randn(n_sample, n_dim)
initial_random_vel.set_value(initial_v.astype(theano.config.floatX))

def train_fn(param_new):
        params.set_value(param_new.astype(theano.config.floatX), borrow=True)
        initial_v = initial_random_vel.get_value()      
        res, res_param, res_sampler = func_eval(samples_true,initial_v, stepsizes0,30)       
        print "eval=", [res, res_param, res_sampler]    
        return res
        
def train_fn_grad(param_new):
        params.set_value(param_new.astype(theano.config.floatX), borrow=True)
        initial_v = initial_random_vel.get_value()      
        res = func_grad(samples_true,initial_v,stepsizes0,30)
        return res
        
"""
prepare the training dataset: 
samples_true: (n_sample_train, n_dim)
"""
n_sample_train = 1000
samples_sd_Normal = numpy.array(rng.randn(n_sample_train, n_dim), dtype=theano.config.floatX)
samples_true = (linalg.sqrtm(cov).dot(samples_sd_Normal.T)).T + mu   

n_epoch = 5000    
initial_params = numpy.random.randn(n_sample*n_dim+n_dim) 

best_samples_params = scipy.optimize.fmin_l_bfgs_b(func = train_fn, 
                                        x0= initial_params,                                       
                                        fprime = train_fn_grad,
                                        #pgtol=1e-5,
                                        maxiter = n_epoch)

print "true mu =", mu
print "estimated mu=", best_samples_params[0][-2:]
print "MSE = ", numpy.sum((best_samples_params[0][-2:]-mu)**2)


