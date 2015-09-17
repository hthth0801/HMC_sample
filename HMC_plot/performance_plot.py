# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:29:32 2015

@author: tian
"""
import numpy as np
import timeit
import scipy.optimize
import matplotlib.pyplot as plt
import theano
import scipy.linalg as linalg
def generate_plot(energy_name, mu, cov, args_stats):
     """
     define the gaussian contour
     """    
     from scipy.linalg import inv
     cov_inv = inv(cov)
     def gaussian_2d(x,y,mu, cov_inv):
        var_x = cov_inv[0,0]
        var_y = cov_inv[1,1]
        cov_xy = cov_inv[0,1]
        log_density = 0.5* (var_x*(x-mu[0])**2+var_y*(y-mu[1])**2+ 2.0 * cov_xy *(x-mu[0])*(y-mu[1]))
        return np.exp(-log_density)
     """
     draw the 2D gaussian contour
     """
     delta = 0.025
     plt.clf()
     plt.subplot(3,1,3)
     gaussian_x = np.arange(-5.0, 5.0, delta)
     gaussian_y = np.arange(-5.0, 5.0, delta)
     mesh_X, mesh_Y = np.meshgrid(gaussian_x, gaussian_y)
     mesh_Z = gaussian_2d(mesh_X, mesh_Y, mu, cov_inv)
     gaussian_Contour =plt.contour(mesh_X,mesh_Y, mesh_Z, 20)
     color_map = iter(['b','r', 'k', 'g', 'c', 'm', 'y'])
     
     
     """
     change nIter if you want to get more samples. by default it is 5
     """
     nIter = 3
     bases = [np.power(10,i) for i in np.arange(nIter)]
    
     """
     set up x_axis, i.e., number of samples. [1,2,3,...9,10,20,30,...90,100,200,...]
     """
     x_axis=[]
     for i in bases:
         for j in np.arange(9):
             x_axis.append(i+j*i)
     n_samples = np.array(x_axis)
     
     if energy_name == 'gaussian':
         from Model_match import Gaussian
         gaussian_2D = Gaussian(mu, cov, *args_stats)
         
     estimated_samples = []
     independent_samples=[]
     train_objective = []
     for n_sample in n_samples:
         print "processing sample = ", n_sample
         n_dim=2
         rng1 = np.random.RandomState(444)
         random_stepsizes = rng1.rand(n_sample)
         random_interval = 1.5*random_stepsizes-1
         stepsize_baseline = 0.2
         noise_level = 2
         stepsizes0 = stepsize_baseline*noise_level**random_interval 
        
         rng2 = np.random.RandomState(1)
         initial_v = rng2.randn(n_sample, n_dim)

         rng3 = np.random.RandomState(125)
         initial_params = rng3.randn(n_sample, n_dim)
         initial_params_flat = initial_params.flatten()
         """
         args_hyper is the set of hyperparameters: initial momentum, stepsizes, number of samplers, n_sample and n_dim
         """
         args_hyper = [initial_v, stepsizes0, 30,n_sample,2]
         best_samples_list = scipy.optimize.fmin_l_bfgs_b(gaussian_2D.f_df_wrapper, 
                                            initial_params_flat,
                                            args = args_hyper,
                                            maxfun=200)
         best_samples = best_samples_list[0].reshape(n_sample, n_dim)
         
         train_objective.append(best_samples_list[1])
         rng = np.random.RandomState(100)
         samples_sd_Normal = np.array(rng.randn(n_sample, n_dim), dtype=theano.config.floatX)
         samples_true = (linalg.sqrtm(cov).dot(samples_sd_Normal.T)).T + mu
         """
         we only draw scatter plot for 500 samples
         """
         if n_sample==500:
             initial_draw = initial_params
             color_current = next(color_map)
             plt.scatter(initial_draw[:,0], initial_draw[:,1], s=2, color = color_current)
             color_current = next(color_map)
             plt.scatter(best_samples[:,0], best_samples[:,1],s=2, color=color_current )
             color_current = next(color_map)
             plt.scatter(samples_true[:,0], samples_true[:,1], s=2, color=color_current)
         
         if args_stats[0] == 'first':
             estimated_samples.append(np.sqrt(np.sum((best_samples.mean(axis=0))**2)))
             independent_samples.append(np.sqrt(np.sum((samples_true.mean(axis=0))**2)))
         if args_stats[0] == 'second':
             estimated_samples.append(np.sqrt(np.sum(((best_samples**2).mean(axis=0))**2)))
             independent_samples.append(np.sqrt(np.sum(((samples_true**2).mean(axis=0))**2)))
         if args_stats[0] == 'third':
             estimated_samples.append(np.sqrt(np.sum(((best_samples**3).mean(axis=0))**2)))
             independent_samples.append(np.sqrt(np.sum(((samples_true**3).mean(axis=0))**2)))
         if args_stats[0] == 'exp':
             estimated_samples.append(np.sqrt(np.sum((np.exp(best_samples).mean(axis=0))**2)))
             independent_samples.append(np.sqrt(np.sum((np.exp(samples_true).mean(axis=0))**2)))
         if args_stats[0] == 'sin':
             estimated_samples.append(np.sqrt(np.sum((np.sin(best_samples).mean(axis=0))**2)))
             independent_samples.append(np.sqrt(np.sum((np.sin(samples_true).mean(axis=0))**2)))
     estimated_samples = np.asarray(estimated_samples)
     independent_samples = np.asarray(independent_samples)
     train_objective = np.asarray(train_objective)
         
     plt.subplot(3,1,1)
     plt.xscale('log')
     plt.plot(n_samples, estimated_samples, color='blue', lw=2, label='rep. samples with all traj.')
     plt.plot(n_samples, independent_samples, color='green', lw=2, label='independent')
     plt.xlabel('number of samples')
     plt.ylabel('value of the matching statistics')
     plt.legend(bbox_to_anchor=(0.,1.02, 1.,.102),loc=3,ncol=2,mode='expand', borderaxespad=0.)
             
     plt.subplot(3,1,2)
     plt.xscale('log') 
     plt.yscale('log')
     plt.plot(n_samples, np.abs((estimated_samples-independent_samples)), color='red', label='True error')  
     plt.plot(n_samples, train_objective, color='black', label="training objective")
     plt.xlabel('number of samples')
     plt.ylabel('error')
     plt.legend(bbox_to_anchor=(0.,1.02, 1.,.102),loc=3,ncol=2,mode='expand', borderaxespad=0.)
     
     #plt.subplot(3,1,3)
     
     plt.show() 
         
         
         
n_dim=2         
rng_mu = np.random.RandomState(123)
mu = np.array(rng_mu.rand(n_dim)*1, dtype=theano.config.floatX)
cov = np.array([[0.8, 0.], [0., 0.6]], dtype=theano.config.floatX)
"""
if wanna add Hamiltonian energy matching, just put 'H' after. e.g., args_stats=['first', 'H']
Hamiltonian (if have any) must be put on the second position
"""
args_stats = ['first']     
start_time = timeit.default_timer()
generate_plot('gaussian',mu, cov, args_stats)
end_time = timeit.default_timer()
print "running time= ", end_time-start_time

         
         
         