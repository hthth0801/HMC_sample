import numpy as np
import scipy.linalg as linalg

import theano
import theano.tensor as T

rng = np.random.RandomState(4321)

class gauss_2d:
	def __init__(self):
		self.mu_np = np.array([1,0.25], dtype=theano.config.floatX).reshape((1,-1))
		self.mu = theano.shared(self.mu_np.ravel()).reshape((1,-1))
		self.cov_np = np.array([[0.8, 0.5], [0.5, 0.6]], dtype=theano.config.floatX)
		self.cov_inv = theano.shared(linalg.inv(self.cov_np))
		self.name = 'gaussian_2d'

	def E(self, X):
		return T.sum( T.dot((X-self.mu), self.cov_inv) * (X-self.mu), axis=1)/2.

	def generate_samples(self, n_sample):
		samples_sd_normal = rng.normal(size=(n_sample, 2)).astype(theano.config.floatX)
		samples_true = (linalg.sqrtm(self.cov_np).dot(samples_sd_normal.T)).T + self.mu_np
		return samples_true
