import sys,copy,os
import numpy as np
import torch
from torch import nn, distributions
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal as mvn

def is_not_singular(mat):
    if np.linalg.det(mat) == 0:
        return False
    else:
        return True

def make_cov(batch_size,lengthscale,add_diagonal=False):
    
    cov = np.zeros((batch_size,batch_size))
    
    for i in range(batch_size):
        for j in range(batch_size):
            cov[i,j] = kern(i,j,lengthscale)
            if add_diagonal:
                if i == j:
                    cov[i,j] += 1 #make matrix non-singular
    return cov

def kern(a,b,l):
    return np.exp(-(a-b)**2/(2*l**2))

class MVNCurve:
    
    def __init__(self,mean,cov):
        
        self.mean = mean
        self.cov = cov
        
    def sample(self,batch_size):
        self.samps = mvn.rvs(self.mean,self.cov,batch_size)     
        return self.samps
        
    def likelihood(self):
        self.probs = mvn.pdf(self.samps,self.mean,self.cov,allow_singular=True)
        return self.probs

    def density(self,grid):
        print('would need a grid that is 30 x 30! havent tried this yet!')
        sys.exit() 

class Gauss8:
    
    def __init__(self,scale,sigma):
        
        self.scale = scale
        self.sigma = sigma
        
        centers = [(1, 0), 
                   (-1, 0), 
                   (0, 1), 
                   (0, -1), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), 
                   (-1. / np.sqrt(2),1. / np.sqrt(2)), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]

        #spread them out
        self.centers = [(self.scale * x, self.scale * y) for x, y in centers]
        
    def sample_orig(self,batch_size):

        dataset = []

        for i in range(batch_size):

            #choose a point randomly from a standard normal (changing the standard deviation)
            # N(mu,sigma) ~ sigma * randn(...) + mu
            point = np.random.randn(2) * self.sigma #changes the standard deviation of each Gaussian
            idx = np.random.randint(8) #randomly choose one of the 8 gaussians
            center = self.centers[idx]
            point = point + center
            dataset.append(point)

        self.samps = np.array(dataset, dtype="float32")

        return self.samps

    def sample(self,batch_size):

        cs = np.random.randint(0,8,batch_size)
        self.samps = np.zeros((batch_size,2))
        for itera,c in enumerate(cs):
            gauss = mvn(mean=self.centers[c],cov=[[self.sigma**2,0],[0,self.sigma**2]])
            self.samps[itera] = gauss.rvs(1)
            
        return self.samps 
        
    def likelihood(self):
        
        fac = 1/8
        probs = np.zeros(len(self.samps))
        for c in self.centers:
            gauss = mvn(mean=c,cov=[[self.sigma**2,0],[0,self.sigma**2]])
            probs += gauss.pdf(self.samps)

        self.probs = fac * probs
        return self.probs

    def density(self,grid):

        fac = 1/8
        probs = np.zeros(len(grid))
        for c in self.centers:
            gauss = mvn(mean=c,cov=[[self.sigma**2,0],[0,self.sigma**2]])
            probs += gauss.pdf(grid)
        self.pdf = fac * probs
        
        return self.pdf

class GMM_data:
    
    def __init__(self,mean1,mean2,cov1,cov2,prob1,prob2):
        
        self.mean1 = mean1
        self.mean2 = mean2
        self.cov1 = cov1
        self.cov2 = cov2
        self.prob1 = prob1
        self.prob2 = prob2
        self.gauss1 = mvn(mean=self.mean1,cov=self.cov1)
        self.gauss2 = mvn(mean=self.mean2,cov=self.cov2)
        
    def sample(self,batch_size,torch_tensor=False):

        thresh = np.random.uniform(0,1,batch_size)
        ind1 = thresh <= self.prob1
        ind2 = ~ind1

        assert np.sum(ind1) + np.sum(ind2) == batch_size

        samp1 = self.gauss1.rvs(np.sum(ind1))
        samp2 = self.gauss2.rvs(np.sum(ind2))
        self.samps = np.concatenate((samp1,samp2))
            
        if torch_tensor:
            return torch.tensor(self.samps.astype(np.float32))
        else:
            return self.samps
        
    def likelihood(self,plot=False):
        
        self.probs = self.prob1 * self.gauss1.pdf(self.samps) + self.prob2 * self.gauss2.pdf(self.samps)
            
        return self.probs
    
    def density(self,grid):
        npts = int(np.sqrt(len(grid)))
        self.pdf = (self.prob1 * self.gauss1.pdf(grid) + self.prob2 * self.gauss2.pdf(grid)).reshape(npts,npts)
        return self.pdf
    
#def is_pos_semidef(x):
    #return np.all(np.linalg.eigvals(x) >= 0)