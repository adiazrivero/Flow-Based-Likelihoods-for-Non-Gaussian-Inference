import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os,sys
import time

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import multivariate_normal as mvn
from scipy.integrate import simps

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


parser.add_argument('--label', type=str)
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--data_dim',type=int,default=2)
parser.add_argument('--loss_dimred',type=str,default='mean')

parser.add_argument('--restart',action='store_true')
parser.add_argument('--load_dir', type=str)
parser.add_argument('--epoch',type=int,default=0)

args = parser.parse_args()

##################################################

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.device = device 

dim0 = args.dims.split('-')[0]

rootdir = '/n/home04/adiazrivero/likelihood_nongaussianity/ffjord/GITHUB/EXPERIMENTS/'
os.makedirs(rootdir,exist_ok=True)

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
    
    train_losses = np.load(load_arrays + 'epoch%s_trainloss.npy' % args.epoch).tolist()
    test_losses = np.load(load_arrays + 'epoch%s_testloss.npy' % args.epoch).tolist()
    
    train_losses = train_losses[:args.epoch]
    test_losses = test_losses[:args.epoch]
    
else:
    train_losses, test_losses = [],[]
    
#############################################################

model = build_ffjord_model(args,device)

if args.restart:
    
    print('Loading checkpoint!')
    
    checkpoints = torch.load(load_arrays + 'epoch%s_checkpt.pth' % args.epoch)
    args = checkpoint['args']
    model.load_state_dict(checkpoint['state_dict'])

print(model)
print(count_parameters(model))

##################################################

if args.data == '8gaussians':
    
    toy = Gauss8(scale=4./np.sqrt(2),sigma=0.25)
    toy_test = Gauss8(scale=4./np.sqrt(2),sigma=0.25)
    mini,maxi = -4,4
    
    assert args.data_dim == 2
    #dim = 2
    
elif args.data == 'GMM2':
    
    mean1 = [0.5,0.00]
    cov1 = [[0.050,0.033],
            [0.033,0.050]]

    mean2 = [-0.5,-0.5]
    cov2 = [[0.060,-0.040],
            [-0.040,0.040]]

    prob1 = 0.5
    prob2 = 1 - prob1

    toy = GMM_data(mean1,mean2,cov1,cov2,prob1,prob2)
    toy_test = GMM_data(mean1,mean2,cov1,cov2,prob1,prob2)
    mini,maxi = -1.5,2.5
    
    #dim = 2
    assert args.data_dim == 2
    
else:
    print('Havent implement others yet, have fun experimenting!')
    sys.exit()

#############################################################

npts = 100
p,side = gridpoints(npts,mini,maxi)

true_pdf = toy.density(p).reshape(npts,npts)
np.savez(save_arrays + 'truepdf',arr=true_pdf,mini=mini,maxi=maxi,npts=npts)
true_prob = simps(simps(true_pdf,side),side)

#since we produce new samples every iteration, we just use a test set instead of validation/test split
test_data = toy_test.sample(args.batch_size)
true_test_loglike = np.log(toy_test.likelihood())
np.save(save_arrays + 'truetestloglike',true_test_loglike)

#############################################################

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model.train()

for itr in range(ini,fin):

    optimizer.zero_grad()

    # generate data
    x = toy.sample(args.batch_size) 
    true_loglike = np.log(toy.likelihood())
    
    x = torch.Tensor(x)
    loss,train_nf_loglike = compute_loss(x, args, model, device)
    loss.backward()
    optimizer.step()
    
    loss_test,test_nf_loglike = compute_loss_test(test_data, args, model, device)
    train_losses.append(loss.item())
    test_losses.append(loss_test.item())
    
    if itr % 5 == 0:

        #reconstructed pdf
        _,final_pdf = compute_loss_test(p, args, model, device)
        final_pdf = np.exp(final_pdf.cpu().numpy()).reshape((npts,npts))
        final_prob = simps(simps(final_pdf,side), side)
        diff = true_prob - final_prob

        fig,ax = plt.subplots(3,3,figsize=(20,15))

        ax[0,0].scatter(true_loglike,train_nf_loglike.cpu().detach().numpy(),s=1,color='blue')
        ax[0,0].plot(true_loglike,true_loglike,color='red')
        ax[0,0].plot(true_test_loglike,true_test_loglike,color='lightgreen',linestyle='dashed')
        ax[0,0].scatter(true_test_loglike,test_nf_loglike.cpu().detach().numpy(),s=1,color='darkgreen')
        ax[0,0].set_xlabel('true loglikelihood')
        ax[0,0].set_ylabel('reconstructure loglikelihood')

        ax[0,1].set_title('residual')
        ax[0,1].scatter(true_loglike,true_loglike-train_nf_loglike.cpu().detach().numpy().reshape(-1),s=1,color='blue')
        ax[0,1].scatter(true_test_loglike,true_test_loglike-test_nf_loglike.cpu().detach().numpy().reshape(-1),s=1,color='darkgreen')
        ax[0,1].axhline(0,color='k')

        ax[0,2].plot(range(len(train_losses)),train_losses,label='train')
        ax[0,2].plot(range(len(test_losses)),test_losses,label='test')

        ax[1,0].imshow(final_pdf,origin='lower')
        ax[1,0].set_axis_off()
        ax[1,0].set_title('model')

        ax[1,1].imshow(true_pdf,origin='lower')
        ax[1,1].set_axis_off()
        ax[1,1].set_title('true')

        ax[1,2].imshow(true_pdf - final_pdf,origin='lower')
        ax[1,2].set_axis_off()
        ax[1,2].set_title('residual (%.2e)' % diff)

        ax[2,0].set_ylabel('p(y)')
        ax[2,0].set_xlabel('y')
        ax[2,0].plot(side,simps(final_pdf,side),label='model')
        ax[2,0].plot(side,simps(true_pdf,side),label='true')
        ax[2,0].legend()

        ax[2,1].set_xlabel('x')
        ax[2,1].set_ylabel('p(x)')
        ax[2,1].plot(side,simps(final_pdf.T,side))
        ax[2,1].plot(side,simps(true_pdf.T,side)) 

        ax[2,2].set_xlabel('x or y')
        ax[2,2].plot(side,simps(final_pdf,side)-simps(true_pdf,side),label='p(y) residual')
        ax[2,2].plot(side,simps(final_pdf.T,side)-simps(true_pdf.T,side),label='p(x) residual')
        ax[2,2].axhline(0,color='k')
        ax[2,2].legend()      

        plt.savefig(save_figures + 'epoch%s.png' % itr)
        plt.close()

        np.save(save_arrays + 'epoch%s_trainloss' % (itr), train_losses)
        np.save(save_arrays + 'epoch%s_testloss' % (itr), test_losses)

        np.save(save_arrays + 'epoch%s_truetrainloglike' % (itr),true_loglike)             
        np.save(save_arrays + 'epoch%s_predtrainloglike' % (itr),train_nf_loglike.cpu().detach().numpy())                   
        np.save(save_arrays + 'epoch%s_predtestloglike' % (itr),test_nf_loglike.cpu().detach().numpy()) 
        np.savez(save_arrays + 'epoch%s_predpdf' % (itr),arr=final_pdf,mini=mini,maxi=maxi,npts=npts)    

        torch.save({'args': args,'state_dict': model.state_dict(),}, 
                   os.path.join(save_arrays, 'epoch%s_checkpt.pth' % itr))

