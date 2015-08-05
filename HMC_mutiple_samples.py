
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:12:19 2015

@author: user
"""

import numpy
import theano
import theano.tensor as T
import hmc_sampling
import scipy.optimize
import matplotlib.pyplot as plt
import timeit


def test_hmc(n_sample, n_dim, mu,cov_inv,rng,seed):
    """
    parameters:
    (maybe just input a gaussian energy function instead of mu and cov_inv)
    ------------
    n_samples: number of samples. 
    n_dim    : number of dim
    mu       : the ground truth mean 
    cov_inv  : the inverse of the ground truth SD. 
    rng      : random generator for initial value x0 for the optimization process.
    seed     : seed for the random momentum generator (for initial_vel)
    """
    #make the params to be shared theano variable, each row is a sample.
    params = theano.shared(value=numpy.zeros(n_sample*n_dim, dtype=theano.config.floatX), name='params', borrow=True)
    initial_pos = params[0:n_sample*n_dim].reshape((n_sample, n_dim))
    
    #return the gaussian energy. 
    def gaussian_energy(x):
        return 0.5 * (T.dot((x - mu), cov_inv) *
                      (x - mu)).sum(axis=1)
    
    """
    dE_dtheta is used for parameter estimation which returns a jacobian matrix. For now, we ignore it, and only 
    consider the second part.
    """
    #def dE_dtheta(x):
     #   result= T.jacobian(gaussian_energy(x),theta)
     #   return result
        
  
    """
    initial_vel: random initial momentum
    n_step:      # of leapfrog steps
    stepsizes:   stepsizes for different particles.
    """
    initial_vel = T.matrix('initial_vel')    
    n_step = T.iscalar('n_step')
    stepsizes = T.vector('stepsizes')
    
    """
    return value of hmc_move:
    1st:  return accept prob. associated with all trajectories.
    2nd:  return accept prob. associated with only end trajectory.
    3rd:  return final positions associated with all trajectories.
    4th:  return final positions associated with only end trajectory.
    """
    #_,accept,_, final_pos= hmc_sampling.hmc_move(initial_vel,initial_pos, gaussian_energy, 0.1,50)
    accept,_,final_pos,_ = hmc_sampling.hmc_move(initial_vel, initial_pos, gaussian_energy, stepsizes, n_step)
    accept_matrix = accept.dimshuffle(0,1, 'x')
    
    """
    define the objective function:
    uncomment the proper one and then update the total_cost
    """
    sampler_cost_first = T.mean(accept_matrix*(initial_pos-final_pos), axis=1)
    #sampler_cost_second = T.mean(accept_matrix*(initial_pos**2-final_pos**2), axis=1)
    #sampler_cost_third = T.mean(accept_matrix*(initial_pos**3-final_pos**3), axis=1)
    #sampler_cost_sin = T.mean(accept_matrix*(T.sin(T.abs_(initial_pos))-T.sin(T.abs_(final_pos))), axis=1)
    #sampler_cost_exp = T.mean(accept_matrix*(T.exp(initial_pos)-T.exp(final_pos)), axis=1)
    total_cost = sampler_cost_first
    
    """
    1): if we average samples along all the trajectories, use the first two lines.
    2): if we penalize each sampler along the trajectories, uncomment and use the last two lines
    
    """
    total_cost = T.mean(total_cost, axis=0)
    total_cost = (total_cost**2).sum()
    #total_cost = (total_cost**2).sum(axis=1)
    #total_cost = T.mean(total_cost)
    
    func_eval = theano.function([initial_vel,stepsizes,n_step], total_cost, name='func_eval', allow_input_downcast=True)
    func_grad = theano.function([initial_vel,stepsizes,n_step], T.grad(total_cost, params), name='func_grad', allow_input_downcast=True)
  
    """
    set up stepsizes for different particles. 
    """
    rng_stepsize = numpy.random.RandomState(353)
    random_stepsizes = numpy.array(rng_stepsize.rand(n_sample), dtype=theano.config.floatX)
    random_interval = 1.5*random_stepsizes-1
    stepsize_baseline = 0.3
    noise_level = 2
    stepsizes0 = stepsize_baseline*noise_level**random_interval
    
    """
    build the evaluation function and gradient function for scipy optimize
    """
    def train_fn(param_new):
        params.set_value(param_new, borrow=True)
        rng_temp = numpy.random.RandomState(seed)
        initial_v=numpy.array(rng_temp.randn(n_sample,n_dim), dtype=theano.config.floatX)
        res = func_eval(initial_v,stepsizes0,30)
        return res
        
    def train_fn_grad(param_new):
        params.set_value(param_new, borrow=True)
        rng_temp = numpy.random.RandomState(seed)
        initial_v=numpy.array(rng_temp.randn(n_sample,n_dim), dtype=theano.config.floatX)
        res = func_grad(initial_v,stepsizes0,30)
        return res
    n_epoch = 5000 
    best_samples_params = scipy.optimize.fmin_l_bfgs_b(func = train_fn, 
                                        x0= numpy.array(rng.randn(n_sample*n_dim), dtype=theano.config.floatX),
                                        #x0 = numpy.array((1.0/cov_inv)*rng.randn(n_sample*n_dim)+mu, dtype=theano.config.floatX),
                                        fprime = train_fn_grad,
                                        maxiter = n_epoch)
    #res = best_samples_params.reshape((n_sample,n_dim))
    res = (best_samples_params[0]).reshape((n_sample, n_dim))
    
    
    """
    uncomment the proper one to print estimated value for different number of samples.
    """    
    print "estimated mean from representative sample= ", res.mean(axis=0) 
   # print "true mu= ", mu   
   # print "estimated second moment from representative samples= ", (res**2).mean(axis=0)
   # print "true second moment= ", mu**2 + 1./cov_inv
   # print "estimated third moment from representative samples= ", (res**3).mean(axis=0)
   # print "estimated sin(x) from representative samples= ", (numpy.sin(numpy.abs(res))).mean(axis=0)
   # print "estimated exp(x) from representative samples= ", (numpy.exp(res)).mean(axis=0)    
  
    """
    uncomment the proper one to return estimated value for different number of samples
    """
    return res.mean(axis=0)
    #return (res**2).mean(axis=0)
    #return (res**3).mean(axis=0)
    #return (numpy.exp(res)).mean(axis=0)
    #return (numpy.sin(numpy.abs(res))).mean(axis=0)
    
def multiple_hmc():
    nIter = 5
    bases = [numpy.power(10,i) for i in numpy.arange(nIter)]
    
    """
    set up x_axis, i.e., number of samples. [1,2,3,...9,10,20,30,...90,100,200,...]
    """
    x_axis=[]
    for i in bases:
        for j in numpy.arange(9):
            x_axis.append(i+j*i)
    n_samples = numpy.array(x_axis)
    
    """
    set up the ground truth mean and std for gaussian. 
    """
    n_dim=1
    rng = numpy.random.RandomState(123)
    mu = numpy.array(rng.rand(n_dim)*1, dtype=theano.config.floatX)
    rng1=numpy.random.RandomState(444)
    cov = numpy.eye(n_dim, dtype=theano.config.floatX)*rng1.rand(1)
    #cov = (rng1.rand(1)).astype(theano.config.floatX)
    cov_inv = 1./(cov)
    
    """
    get the estimated value for each sample size
    """
    start_time = timeit.default_timer()
    y_estimated_temp = [test_hmc(n_sample, 1, mu,cov_inv,rng,n_sample) for n_sample in n_samples]
    end_time = timeit.default_timer()
    print "computing time= ",end_time-start_time
    y_estimated = numpy.array(y_estimated_temp)
    
    """
    get the value from independent samples
    """
    y_independent = []
    for n_sample in n_samples:
        independent_samples = cov*numpy.array(rng.randn(n_sample,1), dtype=theano.config.floatX)+mu
        """
        uncomment the proper one to get the result for different matching function
        """
        y_independent.append(independent_samples.mean(axis=0))
        #y_independent.append((independent_samples**2).mean(axis=0))
        #y_independent.append((independent_samples**3).mean(axis=0))
        #y_independent.append((numpy.exp(independent_samples).mean(axis=0)))
        #y_independent.append(numpy.sin(numpy.abs(independent_samples)).mean(axis=0))
        
    y_independent = numpy.array(y_independent)
    
    """
    plot using log scale
    """
    plt.subplot(2,1,1)
    plt.xscale('log')
    plt.plot(n_samples, y_estimated, color='blue', lw=2, label='rep. samples with all traj.')
    plt.plot(n_samples, y_independent, color='green', lw=2, label='independent')
    plt.xlabel('number of samples')
    plt.ylabel('value of the matching statistics')
    plt.legend(bbox_to_anchor=(0.,1.02, 1.,.102),loc=3,ncol=2,mode='expand', borderaxespad=0.)
    
    plt.subplot(2,1,2)
    plt.xscale('log')
    """
    uncomment the proper one to get the error curve
    """
    plt.plot(n_samples, numpy.abs(y_estimated-y_independent), color='red')
    #plt.plot(n_samples, numpy.abs(y_estimated**2-y_independent**2), color='red')
    #plt.plot(n_samples, numpy.abs(y_estimated**3-y_independent**3), color='red')
    #plt.plot(n_samples, numpy.abs(numpy.exp(y_estimated)-numpy.exp(y_independent)), color='red')
    #plt.plot(n_samples, numpy.abs(numpy.sin(numpy.abs(y_estimated))-numpy.sin(numpy.abs(y_independent))), color='red')
    
    plt.xlabel('number of samples')
    plt.ylabel('error')
    plt.show()
        
      
if __name__=="__main__":
    multiple_hmc()