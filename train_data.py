import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os,sys
import time

import numpy as np
from scipy.stats import multivariate_normal as mvn

import torch
import torch.optim as optim
import torch.utils.data as torchdata
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau


sys.path.append("/n/home04/adiazrivero/likelihood_nongaussianity/ffjord/")
import lib.utils as utils
import lib.layers.odefunc as odefunc
from my_toy_data import GMM_data,Gauss8,MVNCurve,kern,make_cov
from ffjord_functions import *
from mock_datasets import load_WL

##################################################

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']

parser = argparse.ArgumentParser('Continuous Normalizing Flow')

parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','GMM2'],
    type=str, default='pinwheel'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--nepochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--label', type=str, default='experiment')
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--loss_dimred',type=str,default='mean')

parser.add_argument('--epoch',type=int,default=0)
parser.add_argument('--restart',action='store_true')

parser.add_argument('--ell_min',type=float,default=100)
parser.add_argument('--ell_max',type=float,default=12500)

args = parser.parse_args()

##################################################

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.device = device 

dim0 = args.dims.split('-')[0]

if args.WL:
    data_type = 'WL'
    
elif args.BOSS:
    data_type = 'BOSS'

if not args.restart:
    
    save_dir = rootdir + '%s_lr%s_bs%s_blocks%s_dim%s/' % (args.label,args.lr,args.batch_size,args.num_blocks,dim0) 
    os.makedirs(save_dir,exist_ok=True)

    save_figures = save_dir + 'plots/'
    os.makedirs(save_figures, exist_ok=True)

    save_arrays = save_dir + 'arrays/'
    os.makedirs(save_arrays,exist_ok=True)
    
    ini,fin = 1, args.nepochs + 1

else:

    print('Creating new directories!')

    load_arrays = args.load_dir + 'arrays/'
    
    restart_dir = args.load_dir + 'restart/'
    os.makedirs(restart_dir, exist_ok=True)
    
    save_digs = restart_dir + 'plots/'
    os.makedirs(save_figures, exist_ok=True)

    save_arrays = restart_dir + 'arrays/'
    os.makedirs(save_arrays,exist_ok=True)
    
    ini,fin = args.epoch + 1, args.epoch + args.nepochs + 1

if args.restart:
    
    print('Retrieving loss curves')
    
    train_losses = np.load(load_arrays + 'train_losses.npy').tolist()
    test_losses = np.load(load_arrays + 'test_losses.npy').tolist()

    train_losses = train_losses[:args.epoch]
    test_losses = test_losses[:args.epoch]
    
else:
    train_losses, test_losses = [],[]

################################################################################
########################## LOAD DATA AND PRE-PROCESS ###########################
################################################################################
    
print('LOADING WL DATA')

X_pk, X_w, W, _ = load_WL(ell_min=args.ell_min, ell_max=args.ell_max)

#use mean-subtracted + whitened data for training
data = X_w

dim = np.shape(data)[1]
args.dim = dim

mean = np.mean(data,axis=0)
C_x = np.cov(data.T)

mvn_loglike = mvn.logpdf(data, mean=mean, cov=C_x)

train_data = data[:int(0.8*len(data))]
train_mvn_loglike = mvn_loglike[:int(0.8*len(data))]

val_data = data[int(0.8*len(data)):int(0.9*len(data))]
val_mvn_loglike = mvn_loglike[int(0.8*len(data)):int(0.9*len(data))]

test_data = data[int(0.9*len(data)):]
test_mvn_loglike = mvn_loglike[int(0.9*len(data)):]

print('Number of training samples: ', len(train_data))
print('Number of validation samples: ', len(val_data))
print('Number of testing samples: ', len(val_data))

np.save(save_arrays + 'mvn_train_loglike', train_mvn_loglike)
np.save(save_arrays + 'mvn_val_loglike', val_mvn_loglike)
np.save(save_arrays + 'mvn_test_loglike', val_mvn_loglike)

np.save(save_arrays + 'test_set', test_data)

trainloader = torchdata.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

#############################################################

model = build_ffjord_model(args, device)

if args.restart:
    
    print('Loading checkpoint!')
    
    checkpoints = torch.load(load_arrays + 'epoch%s_checkpt.pth' % args.epoch)
    args = checkpoint['args']
    model.load_state_dict(checkpoint['state_dict'])

print(model)
print(count_parameters(model))

#############################################################

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model.train()

for epoch in range(ini,fin):
    
    print('\n Epoch %s' % epoch)

    for count,x in enumerate(trainloader):

        optimizer.zero_grad()
        loss,train_nf_loglike = compute_loss(x, args, model, device)

        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
    
        val_loss,val_nf_loglike = compute_loss_test(val_data, args, model, device)
        val_losses.append(val_loss.item())
    
    fig,ax = plt.subplots(1,3,figsize=(15,5))

    ax[0].plot(val_mvn_loglike,val_mvn_loglike,color='lightgreen',linestyle='dashed')
    ax[0].scatter(val_mvn_loglike,val_nf_loglike.cpu().detach().numpy(),s=1,color='darkgreen')
    ax[0].set_xlabel('true loglikelihood')
    ax[0].set_ylabel('reconstructed loglikelihood')

    ax[1].set_title('residual')
    ax[1].scatter(val_mvn_loglike,val_mvn_loglike - val_nf_loglike.cpu().detach().numpy().reshape(-1),s=1,color='darkgreen')
    ax[1].axhline(0,color='k')

    ax[2].plot(range(len(train_losses)),train_losses,label='train loss')
    ax[2].plot(range(len(val_losses)),val_losses,label='val loss')
    ax[2].legend()

    plt.savefig(save_figures + '/epoch%s.png' % epoch)
    plt.close()

    np.save(save_arrays + 'epoch%s_predvalloglike' % (epoch), val_nf_loglike.cpu().detach().numpy()) 
    np.save(save_arrays + 'train_losses', train_losses)   
    np.save(save_arrays + 'val_losses', val_losses)
    
    torch.save({'args': args,'state_dict': model.state_dict(),}, 
               os.path.join(save_arrays, 'epoch%s_checkpt.pth' % epoch))

