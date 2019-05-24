import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GaussianMixtureModel():
    """Density estimation with Gaussian Mixture Models (GMM).

    You can add new functions if you find it useful, but **do not** change
    the names or argument lists of the functions provided.
    """
    def __init__(self, X, K):
        """Initialise GMM class.

        Arguments:
          X -- data, N x D array
          K -- number of mixture components, int
        """
        self.X = X
        self.n = X.shape[0]
        self.D = X.shape[1]
        self.K = K


    def E_step(self, mu, S, pi):
        """Compute the E step of the EM algorithm.

        Arguments:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array

        Returns:
          r_new -- updated component responsabilities, N x K array
        """
        # Assert that all arguments have the right shape
        assert(mu.shape == (self.K, self.D) and               S.shape  == (self.K, self.D, self.D) and               pi.shape == (self.K, 1))
        r_new = np.zeros((self.n, self.K))

        # Task 1: implement the E step and return updated responsabilities
        # Write your code from here...
        for i in range(self.n):
            for k in range(self.K):
                r_new[i,k] = pi[k] * stats.multivariate_normal.pdf(self.X[i], mu[k], S[k],allow_singular=True)
        r_new = r_new / (r_new.sum())
        # ... to here.
        assert(r_new.shape == (self.n, self.K))
        return r_new


    def M_step(self, mu, r):
        """Compute the M step of the EM algorithm.

        Arguments:
          mu -- previous component means, K x D array
          r -- previous component responsabilities,  N x K array

        Returns:
          mu_new -- updated component means, K x D array
          S_new -- updated component covariances, K x D x D array
          pi_new -- updated component weights, K x 1 array
        """
        assert(mu.shape == (self.K, self.D) and               r.shape  == (self.n, self.K))
        mu_new = np.zeros((self.K, self.D))
        S_new  = np.zeros((self.K, self.D, self.D))
        pi_new = np.zeros((self.K, 1))

        # Task 2: implement the M step and return updated mixture parameters
        # Write your code from here...
        
        for k in range(self.K):
            for i in range(self.n):
                pi_new[k] +=r[i,k]
        pi_new = pi_new / self.n
        
        for k in range (self.K):
            for i in range(self.D):
                mu_new[k] += r[k,i] * self.X[i]
            mu_new[k] = mu_new[k] / r[k,:].sum()
            
            
        for j in range(self.K):
            for i in range(self.n):
                y = self.X[i]- mu_new[j]
            S_new[j] = (r[j,:] * np.dot(y, y.T)).sum(axis=0)
            S_new[j] = S_new[j] /r[j,:].sum()
            
        # ... to here.
        assert(mu_new.shape == (self.K, self.D) and               S_new.shape  == (self.K, self.D, self.D) and               pi_new.shape == (self.K, 1))
        
        return mu_new, S_new, pi_new

    def train(self, initial_params):
        """Fit a Gaussian Mixture Model (GMM) to the data in matrix X.

        Arguments:
          initial_params -- dictionary with fields 'mu', 'S', 'pi' and 'K'

        Returns:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array
          r -- component responsabilities, N x K array
        """
        # Assert that initial_params has all the necessary fields
        assert(all([k in initial_params for k in ['mu', 'S', 'pi']]))
        
        mu = np.zeros((self.K, self.D))
        S  = np.zeros((self.K, self.D, self.D))
        pi = np.zeros((self.K, 1))
        r  = np.zeros((self.n, self.K))
        
        mu = initial_params["mu"]
        S = initial_params["S"]
        pi = initial_params["pi"]
        
        # Compute the exact negative log likelihood. Compute the negative log likelihood
        # at each iteration and stop the iteration if it's equal to the exact and does not change anymore

        # Task 3: implement the EM loop to train the GMM
        # Write your code from here..            
        ll_old = 0
        max_iter = 100
        eps = 0.00001
        ll_new_mat= []
        for i in range(max_iter):
            #E_step
            r = self.E_step(mu, S, pi)
            #M_step
            mu, S, pi = self.M_step(mu, r)
           
            
            # update complete log likelihoood
            ll_new = np.sum(np.log(np.sum(r, axis = 1)))
            ll_new_mat.append(-ll_new)
            if np.abs(ll_new - ll_old) < eps:
                break
            ll_old = ll_new

        
        # ... to here.
        assert(mu.shape == (self.K, self.D) and               S.shape  == (self.K, self.D, self.D) andpi.shape == (self.K, 1) and               r.shape  == (self.n, self.K))
        return mu, S, pi, r,ll_new_mat
    
    def initparams(self,X):
        K= self.K
        D= self.D
        mu= np.random.randint(min(X[:,0]),max(X[:,1]),size=(K,D))
        mu=mu.reshape(K,D)
        pi= np.ones(K)/K
        cov= np.zeros((K,D,D))
        for dim in range(len(cov)):
            np.fill_diagonal(cov[dim],np.std(X,axis=0))   
        params={"mu":mu, "S": S, "pi":pi}
 
        return params

    

if __name__ == '__main__':
    np.random.seed(0)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    # Do not write code outside the class definition or
    # this if-block.
    ##########################

    # import geolocations
    with open ('geolocations.csv',newline='') as csvfile:
    data= list(csv.reader(csvfile))
    data=np.asarray(data)
    data=data.astype(float)
    




