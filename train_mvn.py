import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os,sys
import time
import pickle
import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.integrate import simps

import torch
import torch.optim as optim

sys.path.append("/n/home04/adiazrivero/likelihood_nongaussianity/ffjord/GITHUB/ffjord")
import lib.utils as utils
import lib.layers.odefunc as odefunc

sys.path.append("/n/home04/adiazrivero/likelihood_nongaussianity/ffjord/GITHUB/")
from my_toy_data import GMM_data,Gauss8
from ffjord_functions import *

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
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--loss_dimred',type=str,default='mean')

parser.add_argument('--toy',action='store_true')
parser.add_argument('--diag_term',action='store_true')

parser.add_argument('--epoch',type=int,default=0)
parser.add_argument('--restart',action='store_true')

parser.add_argument('--whiten',action='store_true')

parser.add_argument('--ell_min',type=float,default=100)
parser.add_argument('--ell_max',type=float,default=12500)

args = parser.parse_args()

##################################################

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.device = device 

dim0 = args.dims.split('-')[0]

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


#############################################################

if args.restart:
    
    print('Retrieving loss curves')
    
    train_losses = np.load(load_arrays + 'train_losses.npy').tolist()
    test_losses = np.load(load_arrays + 'test_losses.npy').tolist()
    
    train_losses = train_losses[:args.epoch]
    test_losses = test_losses[:args.epoch]
    
else:
    train_losses, test_losses = [],[]


####### define mu and cov for MVN likelihood #######   

thetas = np.linspace(0, 360, args.data_dim)
mu = 10 * np.sin(thetas)
cov = make_cov(args.data_dim,lengthscale=np.sqrt(8),add_diagonal=args.diag_term)
    
print('COV DETERMINANT:', np.linalg.det(cov))

toy = MVNCurve(mean=mu,cov=cov)
toy_test = MVNCurve(mean=mu,cov=cov)

#since we produce new samples every iteration, we just use a test set instead of validation/test split
test_data = toy_test.sample(args.batch_size)
true_test_loglike = np.log(toy_test.likelihood())

np.save(save_arrays + 'truetestloglike',true_test_loglike)

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

for itr in range(ini,fin):

    optimizer.zero_grad()
    
    x = toy.sample(args.batch_size) 
    true_loglike = np.log(toy.likelihood())
    x = torch.Tensor(x)

    loss, train_nf_loglike = compute_loss(x, args, model, device)
    loss_test, test_nf_loglike = compute_loss_test(test_data, args, model, device)

    train_losses.append(loss.item())
    test_losses.append(loss_test.item())
    loss.backward()
    optimizer.step()

    print('\n Saving to', save_dir)

    fig,ax = plt.subplots(1,3,figsize=(10,4))

    ax[0].scatter(true_loglike,train_nf_loglike.cpu().detach().numpy(),s=1,color='blue')
    ax[0].plot(true_loglike,true_loglike,color='red')
    ax[0].plot(true_test_loglike,true_test_loglike,color='lightgreen',linestyle='dashed')
    ax[0].scatter(true_test_loglike,test_nf_loglike.cpu().detach().numpy(),s=1,color='darkgreen')
    ax[0].set_xlabel('true loglikelihood')
    ax[0].set_ylabel('reconstructed loglikelihood')

    ax[1].set_title('residual')
    ax[1].scatter(true_loglike,true_loglike-train_nf_loglike.cpu().detach().numpy().reshape(-1),s=1,color='blue')
    ax[1].scatter(true_test_loglike,true_test_loglike-test_nf_loglike.cpu().detach().numpy().reshape(-1),s=1,color='darkgreen')
    ax[1].axhline(0,color='k')

    ax[2].plot(range(len(train_losses)),train_losses,label='train')
    ax[2].plot(range(len(test_losses)),test_losses,label='test')
    ax[2].legend()

    plt.savefig(save_figures + 'epoch%s.png' % itr)
    plt.close()

    np.save(save_arrays + 'epoch%s_truetrainloglike' % (itr), true_loglike)             
    np.save(save_arrays + 'epoch%s_predtrainloglike' % (itr), train_nf_loglike.cpu().detach().numpy())                   
    np.save(save_arrays + 'epoch%s_predtestloglike' % (itr), test_nf_loglike.cpu().detach().numpy()) 
    np.save(save_arrays + 'train_losses',train_losses)
    np.save(save_arrays + 'test_losses',test_losses)

    torch.save({'args': args,'state_dict': model.state_dict(),}, 
               os.path.join(save_arrays, 'epoch%s_checkpt.pth' % itr))

