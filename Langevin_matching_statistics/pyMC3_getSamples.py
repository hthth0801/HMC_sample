# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 13:56:25 2016
pyMC3 to generate samples for bayesian Logistic regression
@author: user
"""

import theano.tensor as T
from load_data import load_australian_credit, load_german_credit, load_heart, load_pima_indian
import pymc3 as pm
import numpy as np
from pymc3 import summary
from pymc3 import traceplot

germanData, germanLabel = load_australian_credit();
#germanData, germanLabel = load_pima_indian()
# normalize to let each dimension have mean 1 and std 0
g_mean = np.mean(germanData,axis=0)
g_std = np.std(germanData, axis=0)
germanData = (germanData - g_mean) / g_std


with pm.Model() as model:
    alpha = pm.Normal('alpha_pymc3', mu = 0., tau = 1e-2)
    beta = pm.Normal('beta_pymc3', mu=0., tau = 1e-2, shape=14) # for australian data, it has 14 predictors
    y_hat_prob = 1./(1.+T.exp(-(T.sum(beta*germanData, axis=1)+alpha)))
    yhat = pm.Bernoulli('yhat', y_hat_prob, observed = germanLabel)
    trace = pm.sample(10000, pm.NUTS())
    
trace1 = trace[5000:] # get rid of the burn-in samples
summary(trace1)
traceplot(trace1)

alpha_mean = np.mean(trace1['alpha_pymc3'])
beta_mean = np.mean(trace1['beta_pymc3'], axis=0)
param_mean = (np.sum(alpha_mean) + np.sum(beta_mean)) / 15. 
print " the overall mean of the parameters: ", param_mean