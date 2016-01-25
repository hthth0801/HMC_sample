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

class gauss_nd:
    def __init__(self, n_dim = 10):
        self.dim = n_dim
        self.mu_np = (rng.rand(1,n_dim).astype(theano.config.floatX).reshape((1,-1)))+2.
        self.mu = theano.shared(self.mu_np.ravel()).reshape((1,-1))
        #self.cov_np = np.eye(n_dim, dtype = theano.config.floatX)
        """
        random set the covariance matrix
        when we set the off-diagonal elements, the final-position of LMC will explode
        """
        cov_rand = rng.randn(n_dim, n_dim).astype(theano.config.floatX)
        #cov_rand = rng.rand(n_dim, n_dim).astype(theano.config.floatX)
        U,_ = linalg.qr(cov_rand)
        V = np.diag(rng.rand(n_dim).astype(theano.config.floatX)+0.1)
        cov = np.dot(U, np.dot(V, U.T))
        #cov = (cov + cov.T)/2.
        #cov[np.arange(n_dim), np.arange(n_dim)]=1.0
        
        #cov = np.dot(cov, cov.T)
        #self.cov_np = rng.rand(n_dim, n_dim).astype(theano.config.floatX)
        #self.cov_np = np.diag(rng.rand(n_dim).astype(theano.config.floatX))
        self.cov_np = cov
        self.cov_inv = theano.shared(linalg.inv(self.cov_np))
        self.name = 'gaussian_nd'
        
    def E(self, X):
        return T.sum(T.dot((X-self.mu), self.cov_inv) * (X-self.mu), axis=1)/2.
        
    def generate_samples(self, n_sample):
        samples_sd_normal = rng.normal(size = (n_sample, self.dim)).astype(theano.config.floatX)
        samples_true = (linalg.sqrtm(self.cov_np).dot(samples_sd_normal.T)).T + self.mu_np
        return samples_true
                

class laplace_pixel:
    def __init__(self, n_dim = 16):
        self.dim = n_dim
        self.theta = T.matrix('theta')
        self.epsilon_np = np.array(0.1, dtype = theano.config.floatX)
        self.epsilon = theano.shared(self.epsilon_np)
        self.name = 'laplace_pixel'
    def E(self, X):
        XJ = T.dot(X, self.theta)
        XJ2_ep = self.epsilon + XJ**2
        return T.sum(T.sqrt(XJ2_ep), axis=1)
    def dE_dtheta(self, X):
        return T.grad(T.mean(self.E(X)), self.theta, consider_constant = [X])
    def generate_samples(self, n_sample):
        return rng.laplace(size = (n_sample, self.dim)).astype(theano.config.floatX)
        
class BayeLogReg:
    def __init__(self):
        """
        theta is the combined parameters. 
        Theano matrix: [n_sample *(n_dim_beta + n_dim_alpha)] 
        """
        #self.theta = T.matrix('theta')
        """
        data: theano matrix [n_data * dim]
        label: theano vector [n_data]
        """
        self.data = T.matrix('data_BayLog')
        self.label = T.vector('label_BayLog')
        self.sigma2_np = np.array(100, dtype = theano.config.floatX)
        self.sigma2 = theano.shared(self.sigma2_np)
        # dim is the dimension of the data, so theta is actually n_sample*(n_dim+1)
        #self.dim = n_dim
        self.name = 'Bayesian Logistic Regression'
    def E(self, Theta):
        """
        Theta is the combined parameters we wanna sample
        Theta: theano matrix. [n_sample * (n_dim+1)], i.e., Theta = [beta, alpha]
        """
        beta = Theta[:, :-1]
        alpha = Theta[:, -1:]
        Prior = 0.5 * (1./self.sigma2) * T.sum(alpha**2, axis=1) + 0.5 * (1./self.sigma2) * T.sum(beta**2, axis=1)
        """
        Do some broadcasting operations
        alpha_ext: first flatten the alpha (n_sample, 1) into a vector, then broadcast it to axis =1
        label_ext: broadcast the vector self.label to axis = 0
        """
        #alpha_flat = alpha.flatten()
        #alpha_ext = alpha_flat.dimshuffle(0,'x')
        alpha_ext = alpha.flatten().dimshuffle(0, 'x') # version1 also
        label_ext = self.label.dimshuffle('x',0) # version1 also
        #Likelihood = -label_ext * (T.dot(beta, self.data.T) + alpha_ext) # version 1
        """
        sigmoid_prob: [n_sample * n_data] n_sample is the # of representative samples, n_data is the number of training data
        """
        sigmoid_prob = T.nnet.sigmoid(T.dot(beta, self.data.T) + alpha_ext)
        # now the likelihood is [n_sample * n_data], now we make summation over the training data
        #Likelihood = T.sum(T.log(1. + T.exp(Likelihood)), axis = 1) # version 1
        Likelihood = - T.sum(label_ext * T.log(sigmoid_prob) + (1.-label_ext) * T.log(1.-sigmoid_prob), axis=1)
        # return the energy which is theano vector [n_sample]
        return Prior + Likelihood
        
class LGCPP:
    """
    Log-Gaussian Cox Point Process
    """
    def __init__(self, N = 64):
        """
        N represents the grid size, default, we consider 64*64 grid
        Set up the hyperparameters below. see Christensen et al. 2005
        """
        d = N**2
        beta = 1/33.
        sigma2 = 1.91
        mu = np.log(126) - sigma2/2
        """
        discretize the [0, 1]^2 into 64 * 64 regular grid
        """
        x_r = np.linspace(0, 1., num=N)
        y_r = np.linspace(0, 1., num = N)
        [xs, ys] = np.meshgrid(x_r, y_r)

        coord_flat = np.asarray([xs.flatten(), ys.flatten()]) # 2 * 4096
        coord_list = coord_flat.T # 4096*2 first column is the x-axis, the second is the y-axis, based on row order

        Sigma = np.zeros((d, d))
        for i in range(coord_list.shape[0]):
            diff = coord_list[i] - coord_list
            Sigma[i] = np.sqrt(np.sum(diff**2, axis = 1))
        """
        set up the cov and mu for the Gaussian 
        """
        self.cov_np = (sigma2 * np.exp(-Sigma / (N*beta))).astype(theano.config.floatX)
        self.cov_np_inv = linalg.inv(self.cov_np)
        self.cov_inv = theano.shared(self.cov_np_inv)
        self.mu_np = ((np.ones(d) * mu).astype(theano.config.floatX)).reshape((1,-1))
        self.mu = theano.shared(self.mu_np.ravel()).reshape((1, -1))
        
        self.Y = T.vector('posion_count_LGCPP')    
        self.area_np = 1./d
        self.area = theano.shared(1./d)
        self.N = N
        self.name = 'LGCPP'
        
    def E(self, X):
        """
        X is the latent field we want to estimate
        """
        """
        Prior for latent field X
        """
        Prior = T.sum( T.dot((X-self.mu), self.cov_inv) * (X-self.mu), axis=1)/2.
        Likelihood = -T.dot(X, self.Y) +  T.sum(self.area * T.exp(X), axis=1)
        #return Prior,  Likelihood, T.dot(X, self.Y)
        return Prior + Likelihood

    def generate_samples(self, N=64):
        """
        TODO: we need to really generate samples based on the whole generative process
        Here, just use the one provided in Ziyu wang's page. 
        MyFile.txt contains two columns: (column first, cause previously saved using matlab)
               first column is the vectorized latent field (to be used to compare the latend field we estimated)
               second column is the vectorized posion count (to be used to initiating the theano vector: self.Y)
               
        Return two things: (1) the generated "true" latent field X. (2) the posion count Y. 
        """
        data_lgcpp = np.loadtxt('MyFile.txt')
        X_python = data_lgcpp[:,0].reshape(N, N)
        Y_python = data_lgcpp[:,1].reshape(N, N)
             
        return X_python.T.ravel().astype(theano.config.floatX), Y_python.T.ravel().astype(theano.config.floatX)
    def generate_samples2(self, n_sample):
        """
        generate one gaussian latent field, then based on this field, generate the poisson count
        """
        #samples_sd_normal = rng.normal(size=(1, self.N**2)).astype(theano.config.floatX)
        #samples_X = (linalg.sqrtm(self.cov_np).dot(samples_sd_normal.T)).T + self.mu_np
        samples_X = rng.multivariate_normal(self.mu_np.ravel(), self.cov_np)
        """
        it seems that the obtained samples_X (by decomposing cov matrix) contains the complex number, 
        so cannot be used to generate the poisson count.  Instead try multivariate_normal function      
        """        
        samples_Y = rng.poisson(self.area_np * np.exp(samples_X), size = (n_sample, self.N**2))
        return samples_X.astype(theano.config.floatX), samples_Y.ravel().astype(theano.config.floatX)
        
        
        
        
        
        
        
        
        
        

     
        

        
        
        
        