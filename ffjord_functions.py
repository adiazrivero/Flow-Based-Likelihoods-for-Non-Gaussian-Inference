import numpy as np
import torch
import sys,time,os
import pickle

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular,build_model_tabular_dropout,construct_model

"""
Many of these functions are adapted from the original FFJORD code. 
"""

def restart_ffjord(load_dir,epoch,dim,restart=False,device=torch.device('cuda:0')):

    checkpoints = torch.load(load_dir + 'epoch%s_checkpt.pth' % args.epoch)
    args = checkpoint['args']

    model = build_ffjord_model(args, device)
    model.load_state_dict(checkpoint['state_dict'])
    
    sample_fn, density_fn = get_transforms(model)

    return model, sample_fn, density_fn

def flow_samples(num_samples, dim, prior_sample, transform, device='cuda:0'):
    """
    taken from ffjord
    """
    memory = num_samples
    z = prior_sample(num_samples, dim).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)

    for ii in torch.split(inds, int(memory**2)):
        zk.append(transform(z[ii]))
    zk = torch.cat(zk, 0).cpu().detach().numpy()

    return zk 

def build_ffjord_model(args,device):
    
    regularization_fns, regularization_coeffs = create_regularization_fns(args)

    model = build_model_tabular(args, args.data_dim, regularization_fns).to(device)

    if args.spectral_norm: 
        add_spectral_norm(model)

    set_cnf_options(args, model)

    return model
    
def get_transforms(model):
    """
    taken from ffjord
    """
    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn

def compute_loss(x, args, model, device):
    """
    taken from ffjord
    """
    x = x.type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = model(x, zero)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)
    logpx = logpz - delta_logp

    if args.loss_dimred == 'mean':
        loss = -torch.mean(logpx)
    elif args.loss_dimred == 'sum':
        loss = -torch.sum(logpx)    
    else:
        print('Dimensionality reduction not understood!')

    return loss,logpx

def compute_loss_test(test_data, args, model, device):
    """
    taken from ffjord
    """    
    with torch.no_grad():
        
        if type(test_data) != torch.Tensor:
            test_data = torch.from_numpy(test_data).type(torch.float32).to(device)
        else:
            test_data = test_data.to(device)
        zero = torch.zeros(test_data.shape[0], 1).to(test_data)

        z, delta_logp = model(test_data, zero)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)
        logpx = logpz - delta_logp

    if args.loss_dimred == 'mean':
        loss = -torch.mean(logpx)
    elif args.loss_dimred == 'sum':
        loss = -torch.sum(logpx)    
    else:
        print('Dimensionality reduction not understood!')
    
    return loss,logpx
   

def gridpoints(npts,mini,maxi):
    side = np.linspace(mini, maxi, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    x = torch.from_numpy(x).type(torch.float32)#.to(device)
    return x,side  
