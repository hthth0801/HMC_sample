# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:17:52 2015
mainly used to draw 2D gaussian and the correspongding initial and optimizaed points
@author: tian
"""

import numpy
import theano
import theano.tensor as T
import hmc_sampling
import scipy.optimize
import scipy.linalg as linalg
import timeit
import matplotlib.pyplot as plt


def test_hmc_all(n_sample=1000, n_dim=2):
    #make the params to be shared theano variable, each row is a sample.
    params = theano.shared(value=numpy.zeros(n_sample*n_dim, dtype=theano.config.floatX), name='params', borrow=True)
    initial_pos = params[0:n_sample*n_dim].reshape((n_sample, n_dim))
    
    rng = numpy.random.RandomState(123)
    mu = numpy.array(rng.rand(n_dim)*5, dtype=theano.config.floatX)
   # rng1=numpy.random.RandomState(444)
   # cov = numpy.eye(n_dim, dtype=theano.config.floatX)*rng1.rand(1)
    #cov=numpy.array([[1.,0.95],[0.95, 1.]], dtype=theano.config.floatX)
   # cov = numpy.array(rng1.rand(n_dim,n_dim), dtype=theano.config.floatX)
   # cov = (cov+cov.T)/2.
   # cov[numpy.arange(n_dim), numpy.arange(n_dim)]=1.0
    cov = numpy.array([[0.8, 0.], [0., 0.6]], dtype=theano.config.floatX)
    cov_inv = linalg.inv(cov)
    #cov_inv = 1./(cov)
    #cov_inv = numpy.eye(n_dim, dtype=theano.config.floatX)
    print "begin process..."
    #return the gaussian energy. 
    def gaussian_energy(x):
        return 0.5 * (T.dot((x - mu), cov_inv) *
                      (x - mu)).sum(axis=1)
    
    """
    next, we draw the 2D gaussian contour also define the color map for different set of points along the optimization process
    """
               
    def gaussian_2d(x,y,mu, cov_inv):
        var_x = cov_inv[0,0]
        var_y = cov_inv[1,1]
        cov_xy = cov_inv[0,1]
        log_density = 0.5* (var_x*(x-mu[0])**2+var_y*(y-mu[1])**2+ 2.0 * cov_xy *(x-mu[0])*(y-mu[1]))
        return numpy.exp(-log_density)
    # draw arrows connecting the A and B, mainly used to show the trajectory of each point
    def drawArrow(A, B):
        plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=0.05, width = 0.001, length_includes_head=True)
              
    delta = 0.025
    plt.clf()
    gaussian_x = numpy.arange(-5.0, 10.0, delta)
    gaussian_y = numpy.arange(-5.0, 10.0, delta)
    mesh_X, mesh_Y = numpy.meshgrid(gaussian_x, gaussian_y)
    mesh_Z = gaussian_2d(mesh_X, mesh_Y, mu, cov_inv)
    gaussian_Contour =plt.contour(mesh_X,mesh_Y, mesh_Z, 20)
    #plt.clabel(gaussian_Contour, inline=1)
    #color_map = iter(cm.rainbow(numpy.linspace(0, 1, 20)))
    color_map = iter(['b','r', 'k', 'g', 'c', 'm', 'y'])
    
    initial_vel = T.matrix('initial_vel')     
    n_step = T.iscalar('n_step')
    stepsizes = T.vector('stepsizes')
    
    [accept, accept1,final_pos_new, final_pos_new1, ndeltaH] = hmc_sampling.hmc_move(initial_vel, initial_pos, gaussian_energy, stepsizes,n_step)
   
    accept_matrix = accept.dimshuffle(0,1, 'x')
    # we get the average along the second dimension, i.e., get the mean along the different samples, so after the first operation
    # we get the 2D matrix: number of steps * number of dim
    
    sampler_cost_first = accept_matrix*(initial_pos-final_pos_new)
    #sampler_cost_second = accept_matrix*(initial_pos**2-final_pos**2)
    #sampler_cost_third = accept_matrix*(initial_pos**3-final_pos**3)
    #sampler_cost_sin = accept_matrix*(T.sin(initial_pos)-T.sin(final_pos))
    #sampler_cost_exp = accept_matrix*(T.exp(initial_pos)-T.exp(final_pos))
    total_cost1 = sampler_cost_first
    
    total_cost1 = T.mean(total_cost1, axis=0)
    total_cost1 = T.sum(total_cost1, axis=0)
    
    total_cost2 = T.mean(accept*ndeltaH, axis=0)
    total_cost2 = T.sum(total_cost2)
    
    total_cost = T.mean(total_cost1**2)+total_cost2**2
   
    start_time = timeit.default_timer()
    func_eval = theano.function([initial_vel,stepsizes,n_step], total_cost, name='func_eval', allow_input_downcast=True)
    func_grad = theano.function([initial_vel,stepsizes,n_step], T.grad(total_cost, params), name='func_grad', allow_input_downcast=True)
    end_time = timeit.default_timer()
    print "compiling time= ", end_time-start_time
   
    """
    define the vared stepsize, if want to use the fixed one, use the following lines instead    
    stepsizes_center=numpy.ones((n_sample,), dtype=theano.config.floatX)*0.25
    stepsizes0  = stepsizes_center
    """
    
    rng_stepsize = numpy.random.RandomState(353)
    random_stepsizes = numpy.array(rng_stepsize.rand(n_sample), dtype=theano.config.floatX)
    random_interval = 1.5*random_stepsizes-1
    stepsize_baseline = 0.2
    noise_level = 2
    stepsizes0 = stepsize_baseline*noise_level**random_interval 
    
    
    def train_fn(param_new):
        """
        draw the obtained points on the contour of the previous 2D gaussian
        """     
        params.set_value(param_new.astype(theano.config.floatX), borrow=True)
        rng_temp = numpy.random.RandomState(1234)
        initial_v=numpy.array(rng_temp.randn(n_sample,n_dim), dtype=theano.config.floatX)
        res = func_eval(initial_v, stepsizes0,30)
        print "cost= ", res
        return res
        
    def train_fn_grad(param_new):
        params.set_value(param_new.astype(theano.config.floatX), borrow=True)
        rng_temp = numpy.random.RandomState(1234)
        initial_v=numpy.array(rng_temp.randn(n_sample,n_dim), dtype=theano.config.floatX)
        res = func_grad(initial_v,stepsizes0,30)
        return res
        
    n_epoch = 5000
    
    samples_sd_Normal = numpy.array(rng.randn(n_sample, n_dim), dtype=theano.config.floatX)
    samples_true = (linalg.sqrtm(cov).dot(samples_sd_Normal.T)).T + mu    #samples_true : nsamples*ndim
    
    #initial_draw = samples_true
    initial_points = numpy.array(rng.randn(n_sample*n_dim), dtype=theano.config.floatX)
    """
    draw the initial points onto the 2D gaussian contour
    """
    initial_draw = initial_points.reshape(n_sample, n_dim)
    color_current = next(color_map)
    plt.scatter(initial_draw[:,0], initial_draw[:,1], s=2, color = color_current)
    
   
    best_samples_params = scipy.optimize.fmin_l_bfgs_b(func = train_fn, 
                                        x0= initial_points,
                                        #x0 = samples_true.flatten(),
                                        fprime = train_fn_grad,
                                        maxiter = n_epoch)
   
    res = (best_samples_params[0]).reshape((n_sample, n_dim))
    """
    draw the optimized points onto the 2D gaussian coutour
    """
    color_final = next(color_map)
    plt.scatter(res[:,0], res[:,1],s=2, color=color_final )
    
    """
    we can also plot the set of points which runs one step HMC further than the optimized points, 
    in this case, the initial positions is the optimized points we just found. (comment the following lines if you dont want to plot the intermidate steps)
    func_final_pos: compile function to get the new positions based on the optimized points, return is a 3D tensor [n_steps]*[n_samples]*[n_dim]
    pos_final_step: just extract new points based on specified LF steps ([n_samples]*[n_dim])
    
    """
    """
    func_final_pos =theano.function([initial_vel, stepsizes, n_step], final_pos_new, name='final_pos', allow_input_downcast=True)
    rng_temp = numpy.random.RandomState(1234)
    initial_v_final = numpy.array(rng_temp.randn(n_sample,n_dim), dtype=theano.config.floatX)
    params.set_value(best_samples_params[0].astype(theano.config.floatX))
    
    pos_final = func_final_pos(initial_v_final, stepsizes0, 30)
    #next get the positions after runing sampler which has 2 steps of LF. if you set pos_final[k,:], that reprents sampler which has k+1 LF steps
    # 0<=k<29 cause here we consider LF steps up to 30. i.e, 2, 3, 4, ....30
    pos_final_step = pos_final[0,:]  
    plt.scatter(pos_final_step[:,0], pos_final_step[:,1], s=2, color='k')
    """
   # for initialP, finalP in zip(initial_draw, res):
   #    drawArrow(initialP, finalP)
    """
    draw the ground truth points which sample from the true underlying distribution
    """
    plt.scatter(samples_true[:,0], samples_true[:,1], s=2, color='k')
    
    """
    get the results for representative samples and for indenpent samples. 
    """
    print "estimated mean from representative sample= ", res.mean(axis=0)
    print "true mean= ", mu
    #print "estimated second moment from representive sample= ", (res**2).mean(axis=0)
    #print "estimated third moment from representative samples= ", (res**3).mean(axis=0)
    #print "estimated sin(x) from representative samples= ", (numpy.sin(res)).mean(axis=0)
    #print "estimated exp(x) from representative samples= ", (numpy.exp(res)).mean(axis=0)
    #print "true second moment= ", numpy.outer(mu, mu)+cov
    independent_samples = samples_true
    #independent_samples = (cov)*(numpy.array(rng.randn(n_sample,n_dim), dtype=theano.config.floatX)+mu
    print "estimated mean from independent samples= ", independent_samples.mean(axis=0)
    #print "estimated second moment from independent samples= ", (independent_samples**2).mean(axis=0)
    #print "estimated third moment from independent samples= ", (independent_samples**3).mean(axis=0)
    #print "estimated sin(x) from independent samples= ", (numpy.sin(independent_samples)).mean(axis=0)
    #print "estimated exp(x) from independent samples= ", (numpy.exp(independent_samples)).mean(axis=0)
    
if __name__=="__main__":
    test_hmc_all()
