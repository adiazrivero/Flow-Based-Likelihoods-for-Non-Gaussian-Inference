#!/bin/env python
import sys,time,os,copy,pickle
import argparse
from multiprocessing import Pool,Manager

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm,skew,kurtosis,cauchy,chi2

from sklearn.decomposition import FastICA
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KernelDensity as KD

# -- nonGaussLike -- 
rootdir = '/n/home04/adiazrivero/likelihood_nongaussianity/MultiDark_powerspectra/'
sys.path.insert(1, rootdir + 'chang_code/nonGaussLike/nongausslike/')
import knn as kNN
import pyflann
from skl_groups.divergences import KNNDivergenceEstimator
from skl_groups.features import Features


rootdir = '/n/home04/adiazrivero/likelihood_nongaussianity/ffjord/GITHUB/'
sys.path.insert(1, rootdir)

from functions import * 

#####################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--data_type',type=str,default='WL')
parser.add_argument('--num_iters',type=int,default=100)
parser.add_argument('--N_mocks',type=int,default=2048)
parser.add_argument('--ell_max',type=float,default=1.25*1e4)
parser.add_argument('--ell_min',type=float,default=100)
args = parser.parse_args()    

if args.data_type == 'BOSS':
    #X_pk, X_w, _ = load_BOSS()
    print('Only have 2048 mocks - no need to iterate!')
    sys.exit()
    
elif args.data_type == 'WL':

    X_pk, X_w, _ = load_WL(ell_min=args.ell_min,ell_max=args.ell_max)

print('Have %s %s mocks, but using only %s mocks' % (len(X_pk),args.data_type,args.N_mocks))
    
sks,kts,epss = [],[],[]

for itr in range(args.num_iters):
    
    print(itr)

    inds = np.random.choice(len(X_pk),args.N_mocks,replace=False)
    X_pk2 = X_pk[inds] 
    X_w2 = X_w[inds]

    init = SkewKurtosis(X_w2, verbose=False)
    arr_sk, arr_kt = init.t_stats(fit_gauss=False)
    
    sks.append(arr_sk)
    kts.append(arr_kt)

    init = PairwiseCovariance(X_pk2)
    Splus_kde = init.transcovariance(Splus=True,plot=False)
    eps_plus = np.sum(Splus_kde,axis=0)
    
    epss.append(eps_plus)

np.save('results/ddl_arrays/DATA_%s_full_lmax%s_sks' % (args.data_type,args.ell_max), sks)
np.save('results/ddl_arrays/DATA_%s_full_lmax%s_kts' % (args.data_type,args.ell_max), kts)
np.save('results/ddl_arrays/DATA_%s_full_lmax%s_epss' % (args.data_type,args.ell_max), epss)


