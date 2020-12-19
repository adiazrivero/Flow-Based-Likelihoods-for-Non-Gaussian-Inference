#!/bin/env python
import sys,time,os,copy,pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/n/home04/adiazrivero/likelihood_nongaussianity/ffjord/')
from train_misc import * 

sys.path.append('/n/home04/adiazrivero/likelihood_nongaussianity/ffjord/GITHUB/')
from NG_functions import *
from DDL_functions import *
        
#####################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--data_type',type=str)
parser.add_argument('--likelihood',type=str)
parser.add_argument('--num_iters',type=int,default=100)

parser.add_argument('--GMM_Ncomp',type=int,default=None)

parser.add_argument('--NF_dir',type=str,default=None)
parser.add_argument('--NF_epoch',type=int,default=None)
parser.add_argument('--NF_restart',action='store_true')

parser.add_argument('--ell_min',type=float,default=100)
parser.add_argument('--ell_max',type=float,default=1.25*1e4)

args = parser.parse_args()    

os.makedirs('results/ddl_arrays/',exist_ok=True)

#####################################################################

assert args.data_type in ['WL','BOSS']

if args.data_type == 'BOSS':
    X_pk, X_w, W, _ = load_BOSS()
    
elif args.data_type == 'WL':
    X_pk, X_w, W, _ = load_WL(ell_min=args.ell_min,ell_max=args.ell_max)  


dim = np.shape(X_pk)[1]

assert args.likelihood in ['ICA','GMM','NF','MVN']

if args.likelihood == 'ICA':
    
    S,A = ICA_loglikelihood(X_w)
    samps = ICA_loglikes_samples
    rgs = [S,A]
    
elif args.likelihood == 'GMM':
    
    gmm = GMM_loglikelihood(X_w,args.GMM_Ncomp)
    samps = GMM_loglikes_samples
    rgs = gmm
    
elif args.likelihood == 'NF':

    model, sample_fn = NF_loglikelihood(X_w,[args.NF_dir,args.NF_epoch,dim,args.NF_restart])
    samps = NF_loglikes_samples
    rgs = [model,sample_fn]
    
elif args.likelihood == 'MVN':
    
    samps = MVN_samples
    rgs = None
    
print('Finished fit!')

sks,kts,epss = [],[],[]

for itr in range(args.num_iters):

    print(itr)

    _, X_ref = samps(X_w,rgs)
    print(np.shape(X_ref))

    #t-stats --> white data
    SK = SkewKurtosis(X_ref, verbose=False)
    arr_sk, arr_kt = SK.t_stats(fit_gauss=False)

    sks.append(arr_sk)
    kts.append(arr_kt)

    #transcovariance --> unwhitened data
    X_ref_unW = np.dot(X_ref, np.linalg.inv(W))
    X_ref_unW += np.mean(X_pk,axis=0)

    PC = PairwiseCovariance(X_ref_unW)
    Splus_kde = PC.transcovariance(Splus=True,plot=False)
    eps_plus = np.sum(Splus_kde,axis=1)    

    epss.append(eps_plus)

    if args.likelihood == 'GMM':

        np.save('results/ddl_arrays/%s_%s%s_lmax%s_sks_full' % (args.data_type,args.likelihood,args.GMM_Ncomp,args.ell_max), sks)
        np.save('results/ddl_arrays/%s_%s%s_lmax%s_kts_full' % (args.data_type,args.likelihood,args.GMM_Ncomp,args.ell_max), kts)
        np.save('results/ddl_arrays/%s_%s%s_lmax%s_epss_full' % (args.data_type,args.likelihood,args.GMM_Ncomp,args.ell_max), epss)

    elif args.likelihood == 'NF':

        np.save('results/ddl_arrays/%s_%s_lmax%s_sks_%s_epoch%s_full' % (args.data_type,args.likelihood,args.NF_dir,args.NF_epoch,args.ell_max), sks)
        np.save('results/ddl_arrays/%s_%s_lmax%s_kts_%s_epoch%s_full' % (args.data_type,args.likelihood,args.NF_dir,args.NF_epoch,args.ell_max), kts)
        np.save('results/ddl_arrays/%s_%s_lmax%s_epss_%s_epoch%s_full' % (args.data_type,args.likelihood,args.NF_dir,args.NF_epoch,args.ell_max), epss)

    else:

        np.save('results/ddl_arrays/%s_%s_lmax%s_sks_full' % (args.data_type,args.likelihood,args.ell_max), sks)
        np.save('results/ddl_arrays/%s_%s_lmax%s_kts_full' % (args.data_type,args.likelihood,args.ell_max), kts)
        np.save('results/ddl_arrays/%s_%s_lmax%s_epss_full' % (args.data_type,args.likelihood,args.ell_max), epss)
