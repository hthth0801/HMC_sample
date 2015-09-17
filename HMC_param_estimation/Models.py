# -*- coding: utf-8 -*-
"""
Model calsses. adapts the framework and structures of SFO (Jascha Sohl-Dickstein)
Each model class contains 3 basic components: 
  1. f_df(): returns the numerical function evaluation and gradient
  2. params_init: the initialization of the parameters 
  3. mini_batches: a sequence of the mini_batch of the training samples 
following functions are used for theano builder
  * theano_f_df(): returns the Theano symbolic expression for f and df. 
  * theano_energy(): returns the energy evaluation to be used for HMC   
  * dE_dtheta(): returns the derivative of the energy function
@author: tian
"""
import numpy as np
import theano 
import theano.tensor as T
#import hmc_sampling as HMC
from build_f_df import theano_f_df

class Gaussian:
    def __init__(self, mu, cov, n_batches = 1, n_dim = 2, n_sample=1000, n_step=30):
       
        #following attributes are for theano builder
       # self.n_step = T.iscalar('n_step')
       # self.stepsizes = T.vector('stepsizes')
       # self.initial_pos = T.matrix('initial_pos') #parameter for the representative samples
        self.theta = T.vector('theta') # parameter for the energy based model        
       # self.initial_vel = T.matrix('initial_vel')
       # self.training = T.matrix('training')
        self.n_sample = n_sample
        self.n_dim = n_dim
        self.true_mu = mu
        self.n_step = n_step
        self.cov = cov
        
        
        from scipy.linalg import inv
        cov_inv = inv(cov)
        self.cov_inv = cov_inv
        X = load_gaussian(self.true_mu, self.cov, self.n_dim)
        self.training_samples = X
        self.f_df_theano = theano_f_df(self.theta, self.theano_energy, self.dE_dtheta)
        #self.f_df_theano = theano_f_df(cov_inv)
         # estimate the mu
        #self.initial_params = np.random.randn((n_sample+1), n_dim).astype(np.float64)
        rng3 = np.random.RandomState(125)
        #self.initial_params = [np.random.randn(self.n_sample, self.n_dim), np.random.randn(n_dim,)]
        self.initial_params = [rng3.randn(self.n_sample, self.n_dim), rng3.randn(n_dim,)]
        self.mini_batches=[]
        for ibatch in range(n_batches):
            self.mini_batches.append(X[ibatch::n_batches,:])
            
    def theano_energy(self, x):
        """
        x: theano matrix. [n_samples]*[num_dim] to be used by theano_f_df
        """
        return 0.5 * (T.dot((x - self.theta), self.cov_inv) *
                      (x - self.theta)).sum(axis=1)
                      
    def dE_dtheta(self,x, acpt=None):
        """
        x: theano matrix. [n_samples]*[n_dim]
        acpt: theano vector. [n_samples]
        """
        if acpt == None:
            return T.grad(T.mean(self.theano_energy(x)), self.theta, consider_constant=[x])
        else:
            return T.grad(T.sum(acpt*self.theano_energy(x)), self.theta, consider_constant=[x, acpt])
   
    def f_df(self, params, args):
        """
        returns the numerical f and df, to be used by the optimization solver
        params is the list
        """
        #random_stepsizes = np.random.rand(self.n_sample)
        rng1 = np.random.RandomState(444)
        random_stepsizes = rng1.rand(self.n_sample)
        random_interval = 1.5*random_stepsizes-1
        stepsize_baseline = 0.2
        noise_level = 2
        stepsizes0 = stepsize_baseline*noise_level**random_interval 
        
        rng2 = np.random.RandomState(1)
        initial_v = rng2.randn(self.n_sample, self.n_dim)
        #initial_v = np.random.randn(self.n_sample, self.n_dim)
        samples_batch = args
        #print "samples sizes = ",samples_batch.shape
        #print "params size = ", params[0].shape, params[1].shape
        theano_args = params + [samples_batch, initial_v, stepsizes0, self.n_step]
        #theano_args = params + [initial_v, stepsizes0, self.n_step]
        results = self.f_df_theano(*theano_args)
        #return results[:2], results[2:]
        return results[:2], results[2:]
        
        
def load_gaussian(mu, cov, n_dim, n_train=10000):
    rng4 = np.random.RandomState(5)
    #samples_standard = np.array(np.random.randn(n_train, n_dim)).astype(theano.config.floatX)
    samples_standard = np.array(rng4.randn(n_train, n_dim)).astype(theano.config.floatX)
    from scipy.linalg import sqrtm
    samples_gaussian = (sqrtm(cov).dot(samples_standard.T)).T + mu
    return samples_gaussian

    