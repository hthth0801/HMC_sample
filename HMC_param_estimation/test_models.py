# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:20:57 2015
test models
@author: tian
"""
from Models import Gaussian
import numpy as np
import theano
import theano.tensor as T
from scipy.linalg import inv
import timeit
import optimation_solver
n_dim = 2

rng = np.random.RandomState(123)
mu = np.array(rng.rand(n_dim)*5, dtype=theano.config.floatX)
rng1=np.random.RandomState(444)
cov = np.array(rng1.rand(n_dim,n_dim), dtype=theano.config.floatX)
cov = (cov+cov.T)/2.
cov[np.arange(n_dim), np.arange(n_dim)]=1.0
cov_inv = inv(cov)
start_time = timeit.default_timer()

gaussian_2D = Gaussian(mu, cov)
end_time = timeit.default_timer()

params_init = gaussian_2D.initial_params

trainer = optimation_solver.train(gaussian_2D)
#x_init = trainer.param_init_flatten
#mini_batches = trainer.model.mini_batches
#result_model = trainer.f_df_flat(x_init, mini_batches)

#res = trainer.theta_list_to_flat(params_init)
#res_list = trainer.theta_flat_to_list(res)
final_res = trainer.LBFGS()
#final_res = trainer.SGD(20)
#final_res = trainer.LBFGS_minibatch()
#final_res = trainer.SFO()
print "true mu =", mu
print "estimated mu=", final_res[0][-2:]
#print "estimated mu=", final_res[0][-2:]
#print "MSE = ", np.sum((final_res[0][-2:]-mu)**2)
print "compiling time= ", end_time-start_time

#result = gaussian_2D.f_df(params_init)

#X = load_gaussian(mu, cov,n_dim)