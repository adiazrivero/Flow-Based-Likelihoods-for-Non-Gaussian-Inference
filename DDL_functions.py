import numpy as np

from scipy.stats import multivariate_normal as mvn

from sklearn.decomposition import FastICA
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KernelDensity as KD

import torch

from ffjord.train_misc import build_model_tabular, create_regularization_fns

def MVN_samples(arr,dummy):
    
    samps = mvn.rvs(np.mean(arr,axis=0),np.cov(arr.T),len(arr))
    
    return dummy, samps

def ICA_loglikelihood(arr):

    ica = FastICA(whiten=False,max_iter=1000,tol=1e-3)
    S_ = ica.fit_transform(arr)
    A_ = ica.mixing_
    
    return S_, A_
    
    
def ICA_loglikes_samples_fitSA(arr):

    N_samp = np.shape(arr)[0]
    N_bins = np.shape(arr)[1]

    d = 1
    scotts_b = N_samp**(-1./(d+4))
    print('scotts_b = %.2f' % scotts_b)

    ica = FastICA(whiten=False,max_iter=1000,tol=1e-3)
    S_ = ica.fit_transform(arr)
    A_ = ica.mixing_
    #W_ica = ica.components_
    
    X_ref_ica_unmixed = np.zeros((N_samp,N_bins))
    X_ref_ica = np.zeros((N_samp,N_bins))

    loglike_ica = np.zeros(np.shape(arr)[0])

    for j in range(N_bins):

        X_ica_ind = S_[:,j].reshape(-1, 1)
        kde = KernelDensity(bandwidth=scotts_b,kernel='gaussian').fit(X_ica_ind)

        samps = kde.sample(N_samp) #samples for the jth bin
        X_ref_ica_unmixed[:,j] = np.ndarray.flatten(samps)

        log_dens = kde.score_samples(X_ica_ind)#.reshape(len(X_ica_ind),1))
        loglike_ica += log_dens

    X_ref_ica = np.dot(X_ref_ica_unmixed,A_.T) #should be akin to samples X^mock.
    
    return loglike_ica, X_ref_ica

def ICA_loglikes_samples(arr, lst):

    S_ = lst[0]
    A_ = lst[1]
    
    N_samp = np.shape(arr)[0]
    N_bins = np.shape(arr)[1]

    d = 1
    scotts_b = N_samp**(-1./(d+4))
    
    X_ref_ica_unmixed = np.zeros((2048,N_bins))
    X_ref_ica = np.zeros((2048,N_bins))

    loglike_ica = np.zeros(np.shape(arr)[0])

    for j in range(N_bins):

        X_ica_ind = S_[:,j].reshape(-1, 1)
        kde = KernelDensity(bandwidth=scotts_b,kernel='gaussian').fit(X_ica_ind)
        
        samps = kde.sample(2048)
        X_ref_ica_unmixed[:,j] = np.ndarray.flatten(samps)

    X_ref_ica = np.dot(X_ref_ica_unmixed,A_.T)
    
    return loglike_ica, X_ref_ica

def GMM_loglikelihood(arr,N_comp=None):
    
    if N_comp == None:
        N_comp = np.shape(arr)[1]   

    gmm = GMM(n_components=N_comp, covariance_type='full', init_params='kmeans')
    gmm.fit(arr)    
    
    return gmm
    
def GMM_loglikes_samples_fitGMM(arr,N_comp=None):
    
    if N_comp == None:
        N_comp = np.shape(arr)[1]   
        
    N_samp = np.shape(arr)[0]
    gmm = GMM(n_components=N_comp, covariance_type='full', init_params='kmeans')
    gmm.fit(arr)
    X_ref_gmm, cs = gmm.sample(N_samp)
    loglike_gmm = gmm.score_samples(arr)
    
    return loglike_gmm, X_ref_gmm

def GMM_loglikes_samples(arr, gmm):
    
    N_samp = 2048
    
    X_ref_gmm, cs = gmm.sample(N_samp)
    loglike_gmm = gmm.score_samples(arr)

    return loglike_gmm, X_ref_gmm

def NF_loglikelihood(arr,arglist):
    
    load_dir = arglist[0]
    epoch = arglist[1]
    dim = arglist[2]
    restart = arglist[3]
    
    model, sample_fn, _ = restart_ffjord(load_dir,epoch=epoch,dim=dim,restart=restart)
    
    return model, sample_fn
    
def NF_loglikes_samples(arr,lst,loglike=False): 
    
    model = lst[0]
    sample_fn = lst[1]

    if loglike:
        
        with torch.no_grad():

            arr = arr.type(torch.float32).to(device)
            zero = torch.zeros(arr.shape[0], 1).to(arr)
            z, delta_logp = model(arr, zero)
            logpz = standard_normal_logprob(z).sum(1, keepdim=True)
            logpx = logpz - delta_logp

        loglike_nf = logpx.cpu().numpy()

    else:
        loglike_nf = []

    X_ref_nf = flow_samples(2048, np.shape(arr)[1], prior_sample=torch.randn, transform=sample_fn).astype(np.float64)
    
    return loglike_nf, X_ref_nf


def NF_loglikes_samples_byepoch(arr,load_dir,epoch,dim,restart=False):
    
    model, sample_fn, density_fn = restart_ffjord(load_dir,epoch=epoch,dim=dim,restart=restart)
    
    with torch.no_grad():

        arr = arr.type(torch.float32).to(device)
        zero = torch.zeros(arr.shape[0], 1).to(arr)
        z, delta_logp = model(arr, zero)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)
        logpx = logpz - delta_logp
    
    loglike_nf = logpx.cpu().numpy()

    X_ref_nf = flow_samples(np.shape(arr)[0], np.shape(arr)[1], prior_sample=torch.randn, transform=sample_fn).astype(np.float64)
    
    return loglike_nf, X_ref_nf, sample_fn


def restart_ffjord(load_dir, epoch, dim, restart=False):

    if not restart:
        checkpoint = torch.load('/n/holyscratch01/dvorkin_lab/adiazrivero/ffjord/' + load_dir + '_arrays/epoch%s_checkpt.pth' % epoch, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load('/n/holyscratch01/dvorkin_lab/adiazrivero/ffjord/' + load_dir + '_arrays/restart/epoch%s_checkpt.pth' % epoch, map_location=torch.device('cpu'))
    args = checkpoint['args']
    device = torch.device('cuda:0')

    #re-initialize model
    if args.glow or args.nf:
        print('CONSTRUCTING MODEL WITH GLOW OR NF')
        model = construct_model(args).to(device)
    else:
        print('CONSTRUCTING REGULAR MODEL')
        model = build_ffjord_model(args, device)

    #model = build_ffjord_model(args, device)
    
    """
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, dim, regularization_fns).to(device)
    if args.spectral_norm: 
        add_spectral_norm(model)
    set_cnf_options(args, model)
    """
    
    model.load_state_dict(checkpoint['state_dict'])
    sample_fn, density_fn = get_transforms(model)

    return model, sample_fn, density_fn

def build_ffjord_model(args, device):
    
        regularization_fns, regularization_coeffs = create_regularization_fns(args)
        """
        if args.dropout:
            
            print('Network with dropout (p=%s)!' % args.dropout_p)
            
            model = build_model_tabular_dropout(args, args.data_dim, regularization_fns).to(device)
        
        else:
        """
        model = build_model_tabular(args, args.data_dim, regularization_fns).to(device)
        
        if args.spectral_norm: 
            add_spectral_norm(model)
        
        set_cnf_options(args, model)
        
        return model
