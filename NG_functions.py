import numpy as np
import matplotlib.pyplot as plt
import sys,time,os
import pickle

from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm,skew,kurtosis

from sklearn.decomposition import FastICA
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture as GMM

rootdir = '/n/home04/adiazrivero/likelihood_nongaussianity/MultiDark_powerspectra/'
sys.path.insert(1, rootdir + 'chang_code/nonGaussLike/nongausslike/')
import knn as kNN
import pyflann
from skl_groups.divergences import KNNDivergenceEstimator
from skl_groups.features import Features

from matplotlib.lines import Line2D

def variance_skewness(n):
    return (6 * n * (n-1)) / ((n-2) * (n+1) * (n+3))

def variance_kurtosis(n):
    return (24 * n * (n-1)**2) / ((n-3) * (n-2) * (n+3) * (n+5))

def rebin(bins):
    newbins = [(bins[count] + bins[count+1]) / 2 for count in range(len(bins[:-1]))]
    return np.array(newbins)

def whiten(X): 
    
    C_x = np.cov(X.T) 
    W = np.linalg.cholesky(np.linalg.inv(C_x) ) 
    
    return np.dot(X, W), W

class SkewKurtosis:
    
    def __init__(self,data,verbose=False):
        
        self.data = data
        self.num_samples = np.shape(data)[0]
        self.num_bins = np.shape(data)[1]
        
        #since the KDE is done one bin at a time d = 1
        d = 1
        self.scotts_b = self.num_samples**(-1./(d+4))
                
        if verbose == True:
            print('Number of samples: ', self.num_samples,'Number of bins: ',self.num_bins)
        
    def t_stats(self,fit_gauss=False):
        
        listt_sk, listt_kt = [], []

        if fit_gauss == True:
            numrows = int(np.ceil(self.num_bins / 5))
            fig,ax = plt.subplots(numrows,5,figsize=(20,20),sharex=True,sharey=True)
        
        count1 = 0
        count2 = 0
        tot = 0
                
        for i in range(self.num_bins):

            data_bin = self.data[:,i]  
            sk = skew(data_bin)
            se_sk = np.sqrt(variance_skewness(self.num_samples))
            t_sk = sk/se_sk
            listt_sk.append(np.abs(t_sk))

            kt = kurtosis(data_bin)
            se_kt = np.sqrt(variance_kurtosis(self.num_samples))
            t_kt = kt/se_kt
            listt_kt.append(np.abs(t_kt))

            if fit_gauss == True:

                data_bin = data_bin.reshape(-1,1)

                n, bins, patches = ax[count1,count2].hist(data_bin,bins=np.linspace(np.amin(data_bin),np.amax(data_bin),50),density=True,label='skew = %.2f  kurt = %.2f' % (sk,kt),color='lightblue')

                kde = KernelDensity(kernel='gaussian', bandwidth=self.scotts_b).fit(data_bin)
                xs = np.linspace(np.amin(data_bin),np.amax(data_bin),50).reshape(-1, 1)
                log_dens = kde.score_samples(xs)
                ax[count1,count2].plot(xs[:,0], np.exp(log_dens),label='KDE',color='k',linewidth=2)
                
                (mu, sigma) = norm.fit(data_bin)
                y = norm.pdf(bins, mu, sigma)
                ax[count1,count2].plot(bins, y, 'r--', linewidth=2,label='Fit N(%.1f,%.1f)' % (mu,sigma))
                #ax[count1,count2].set_title('Bin %s' % i, fontsize=15)
                
                #BOSS
                ax[count1,count2].text(-4,0.45,'Bin %s' % i,fontsize=15)
                ax[count1,count2].text(1,0.4,'$t_{\\rm skew}$ = %.2f' % (t_sk),fontsize=15)
                ax[count1,count2].text(1,0.3,'$t_{\\rm kurt}$ = %.2f' % (t_kt),fontsize=15)
                #WL
                #ax[count1,count2].text(-4,0.45,'Bin %s' % i,fontsize=15)
                #ax[count1,count2].text(1.75,0.4,'$t_{\\rm skew}$ = %.2f' % (t_sk),fontsize=15)
                #ax[count1,count2].text(1.75,0.3,'$t_{\\rm kurt}$ = %.2f' % (t_kt),fontsize=15)
                
                count2 += 1
                if count2 == 5:
                    count1 += 1
                    count2 = 0   
                tot += 1
        
        if fit_gauss == True:
            custom_lines = [Line2D([0], [0], color='lightblue', lw=4),
                            Line2D([0], [0], color='red',linestyle='dashed'),
                            Line2D([0], [0], color='k',)]

            ax[count1,count2].legend(custom_lines, ['Data', 'Gaussian Fit', 'KDE'],fontsize=15)
            ax[count1,count2].get_xaxis().set_visible(False)
            ax[count1,count2].get_yaxis().set_visible(False)

            while count2 < 4:
                count2 += 1
                fig.delaxes(ax[count1,count2])
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
        
        self.arr_sk = np.array(listt_sk)
        self.arr_kt = np.array(listt_kt)
        
        return self.arr_sk,self.arr_kt
    
    def union_intersection(self,verbose=False):
        
        twosig_sk = self.arr_sk[self.arr_sk > 2]
        twosig_sk_ind = np.array(range(len(self.arr_sk)))[self.arr_sk > 2]
        twosig_kt = self.arr_kt[self.arr_kt > 2]
        twosig_kt_ind = np.array(range(len(self.arr_kt)))[self.arr_kt > 2]

        union = len(set(twosig_sk_ind) | set(twosig_kt_ind))
        intersection = len(set(twosig_sk_ind) & set(twosig_kt_ind))

        if verbose == True:
            print('Total number of elements that have at least one measure > 2 sigma : ', union)
            print('Total number of elements that have both measures > 2 sigma : ', intersection)
        
        return union,intersection
    
    def figures(self):

        fig, ax = plt.subplots(figsize=(8,5))
        fig.suptitle('Nongaussianity of individual bins (%s samples)' % self.num_samples,fontsize=20)
        ax.yaxis.grid(which="major", color='gray', linestyle='-', linewidth=1)
        ml = MultipleLocator()
        ax.xaxis.set_minor_locator(ml)
        ax.xaxis.grid(which="both", color='gray', linestyle='dashed', linewidth=0.7, alpha=0.5)
        ax.scatter(range(len(self.arr_sk)),self.arr_sk,marker='x',label='$t_{skew}$')
        ax.scatter(range(len(self.arr_kt)),self.arr_kt,marker='v',label='$t_{kurt}$')
        ax.legend(loc='upper right',fontsize=20)
        ax.set_xlabel('Bin number',fontsize=20)
        ax.set_ylabel('$\mid$ $t$-statistic $\mid$',fontsize=20)
        plt.show()
        
    def compare2Gauss(self,n_draws=1000,plot=False):

        mu = np.mean(self.data,axis=0)
        cov = np.cov(self.data.T)

        all_sk, all_kt = [], []
        all_union, all_inter = [], []

        for _ in range(n_draws):

            samps = mvn.rvs(mu,cov,self.num_samples)
            init = SkewKurtosis(samps,verbose=False)
            arr_sk2, arr_kt2 = init.t_stats()
            uni2,inter2 = init.union_intersection(verbose=False)
            all_sk.append(arr_sk2), all_kt.append(arr_kt2)
            all_union.append(uni2), all_inter.append(inter2) 

        all_sk, all_kt = np.array(all_sk), np.array(all_kt)
        
        if plot: 
            fig,ax = plt.subplots(7,5,figsize=(20,20))
            count1 = 0
            count2 = 0
            for i in range(self.num_bins):
                ax[count1,count2].axvline(self.arr_sk[i],label='True $t_{\\rm skew}$',color='k')
                ax[count1,count2].axvline(self.arr_kt[i],label='True $t_{\\rm kurt}$',color='k',linestyle='dotted')
                ax[count1,count2].hist(all_sk[:,i],bins=np.linspace(0,4,25),label='Bin %s $t_{\\rm skew}$' % i,color='b')
                ax[count1,count2].hist(all_kt[:,i],alpha=0.8,bins=np.linspace(0,4,25),label='Bin %s $t_{\\rm kurt}$' % i,color='orange')

                ax[count1,count2].axvline(np.mean(all_sk[:,i],axis=0),label='mean',color='red')
                ax[count1,count2].axvline(np.median(all_kt[:,i],axis=0),label='median',color='red',linestyle='dotted')
                ax[count1,count2].axvline(np.percentile(all_kt[:,i],95,axis=0),label='95th perc.',color='red',linestyle='dashed')

                ax[count1,count2].set_xlim(0,6)
                ax[count1,count2].grid()
                count2 += 1
                if count2 == 5:
                    count1 += 1
                    count2 = 0       
            ax[0,0].legend(fontsize=10)
                    
            plt.show()

        return all_sk,all_kt

class PairwiseCovariance:
    
    def __init__(self,data):
        
        self.data = data
        self.num_samples = np.shape(data)[0]
        self.num_bins = np.shape(data)[1]
        
        d = 1
        self.scotts_b = self.num_samples**(-1./(d+4))  
        
    def transcovariance(self, Splus=False, Sdiv=False, Smult=False, n_bins=50, plot=False, example=False):
            
        X_meansub = self.data - np.mean(self.data,axis=0)
        
        mat_hist = np.zeros((self.num_bins,self.num_bins))
        mat_kde = np.zeros((self.num_bins,self.num_bins))
        rng = range(self.num_bins)

        for i in rng: 

            for j in range(i,rng[-1]+1):

                if i == j:
                    continue

                cols = X_meansub[:,[i,j]]
                X_w, W = whiten(cols)
                
                if Splus == True:
                    metric = np.sum(X_w,axis=1)
                    
                elif Sdiv == True:
                    metric = np.array([row[0] / row[1] for row in X_w])

                elif Smult == True:
                    metric = np.array([row[0] * row[1] for row in X_w])                 
                    
                metric = metric.reshape(-1,1)

                xs0_hist = np.linspace(np.amin(metric),np.amax(metric),n_bins+1)
                
                xs0 = np.linspace(np.amin(metric),np.amax(metric),n_bins)
                dx = xs0[1] - xs0[0]
                xs = xs0.reshape(-1, 1)
                
                if Splus == True:
                    label = 'Norm'
                    func = norm
                    
                    theory1 = func.pdf(xs0,0,np.sqrt(2))
                    hist2, bins, patches = plt.hist(metric, density=True, bins=xs0_hist)
                    xs2 = rebin(bins)       
                    theory2 = func.pdf(xs2,0,np.sqrt(2))
                    
                elif Sdiv == True:
                    label = 'Cauchy'
                    func = cauchy
                    
                    theory1 = func.pdf(xs0)
                    hist2, bins, patches = plt.hist(metric, density=True, bins=xs0_hist)
                    xs2 = rebin(bins)       
                    theory2 = func.pdf(xs2)
                    
                elif Smult == True:
                    label = 'VarGamma'
                    func = VarGamma
                    
                    theory1 = np.array(func.pdf(xs0,sigma=1/2))   
                    hist2, bins, patches = plt.hist(metric, density=True, bins=xs0_hist)
                    xs2 = rebin(bins)       
                    theory2 = np.array(func.pdf(xs2,sigma=1/2))
                    #area_vg = simps(theory.reshape(-1), dx=dx)
                    #area_kde = simps(np.exp(H), dx=dx)
                    #print(area_vg,area_kde) 
                    
                #using KDE
                kde = KernelDensity(kernel='gaussian', bandwidth=self.scotts_b).fit(metric)
                H = kde.score_samples(xs)
                hist1 = np.exp(H)

                diff1 = 1/(len(xs)) * np.sum((hist1 - theory1)**2)
                mat_kde[i][j] = diff1 
                mat_kde[j][i] = diff1
                
                #using histogram
                diff2 = 1/(len(xs2)) * np.sum((hist2 - theory2)**2)
                mat_hist[i][j] = diff2
                mat_hist[j][i] = diff2
                
                if plot == True:
                    
                    print('diff KDE (%s bins): %.5f' % (len(hist1),diff1), 'diff hist (%s bins): %.5f' % (len(hist2),diff2))
                    
                    plt.plot(xs[:,0], hist1, label='KDE')
                    plt.plot(xs, theory1, 'r--', linewidth=2,label='theory')
                    plt.plot(xs2, theory2, 'g', linewidth=2,label='theory2')
                    plt.title('%s %s' % (i,j))
                    plt.legend()
                    plt.show()  
 
                    if example:
                        sys.exit()
            
                else:
                    plt.close()
                    
        return mat_kde, mat_hist   
    
    
def kNNdiv_general(X, Y, Knn=3, div_func='kl',alpha=None, njobs=1,): #renyi:.5
    """
    kNN divergence estimate for samples drawn from any two arbitrary distributions.
    """
    if Y.shape[1] != X.shape[1]:
        raise ValueError('dimension between X_white and Gaussian reference distribution do not match') 
        
    if isinstance(Knn, int): 
        Knns = [Knn]
    elif isinstance(Knn, list): 
        Knns = Knn    
    
    if alpha is not None:
        div_func = div_func +':%s' % alpha
    
    kNN = KNNDivergenceEstimator(div_funcs=[div_func], Ks=Knns, version='slow', clamp=False, n_jobs=njobs)
    feat = Features([X, Y])
    div_knn = kNN.fit_transform(feat)
    
    if len(Knns) == 1: 
        return div_knn[0][0][0][1]
    
    div_knns = np.zeros(len(Knns))
    for i in range(len(Knns)): 
        div_knns[i] = div_knn[0][i][0][1]
    return div_knns
    
def npd_function(X, n_sample=50, k=10, s=2000, verbose=False):
    """
    X : 2D numpy array whose likelihood we want to compare to a Gaussian likelihood
    n_sample: number of independent Gaussian realizations of the data to draw
    k : number of nearest-neighbors
    s : number of samples to draw from the Gaussian 
    """
    C_x = np.cov(X.T)
    mu = np.mean(X,axis=0) #np.zeros(dim)

    X_ref_gauss = mvn.rvs(mu, C_x, size=s)

    kl_ref, kl_data = [], []

    for i in range(n_sample):
        
        if verbose:
            print(i)

        Y_ref_gauss = mvn.rvs(mu, C_x, size=s-1)
        
        kl_ref.append(kNNdiv_general(X_ref_gauss, Y_ref_gauss, Knn=k, alpha=None, div_func='kl'))
        kl_data.append(kNNdiv_general(X, Y_ref_gauss, Knn=k, alpha=None, div_func='kl'))
        
    return kl_ref, kl_data


def npd_function_gmm(X, n_sample=50, k=10, s=2000, n_comp=10, verbose=False):

    gmm = GMM(n_components=n_comp, covariance_type='full',init_params='kmeans')
    gmm.fit(X)
    X_ref,_ = gmm.sample(s)
    
    kl_ref, kl_data = [], []

    for i in range(n_sample):
        
        if verbose:
            print(i)
        
        Y_ref,_ = gmm.sample(s-1)

        kl_ref.append(kNNdiv_general(X_ref, Y_ref, Knn=k, alpha=None, div_func='kl'))
        kl_data.append(kNNdiv_general(X, Y_ref, Knn=k, alpha=None, div_func='kl'))    
        
    return kl_ref, kl_data

def npd_function_ica(X, n_sample=50, k=10, s=2000, verbose=False):
    
    N_samp = np.shape(X)[0]
    N_bins = np.shape(X)[1]
    
    d = 1
    scotts_b = N_samp**(-1./(d+4))

    ica = FastICA(whiten=False,max_iter=1000,tol=1e-2)
    S_ = ica.fit_transform(X)
    A_ = ica.mixing_

    Y_ref_ica_unmixed = np.zeros((n_sample,N_samp-1,N_bins))
    Y_ref_ica = np.zeros((n_sample,N_samp-1,N_bins))

    for i in range(n_sample): 
        for j in range(N_bins):
            X_ica_ind = S_[:,j].reshape(-1, 1) #shape is now (2048,1)
            kde = KernelDensity(bandwidth=scotts_b,kernel='gaussian').fit(X_ica_ind)
            samps = kde.sample(N_samp-1) #2047 samples for the jth k bin
            Y_ref_ica_unmixed[i,:,j] = np.ndarray.flatten(samps)
        Y_ref_ica[i] = np.dot(Y_ref_ica_unmixed[i],A_.T)#applying the mixing matrix to undo the ICA transformation
    
    X_ref_ica_unmixed = np.zeros((N_samp,N_bins))
    X_ref_ica = np.zeros((N_samp,N_bins))

    for j in range(N_bins):
        X_ica_ind = S_[:,j].reshape(-1, 1) #shape is now (2048,1)
        kde = KernelDensity(bandwidth=scotts_b,kernel='gaussian').fit(X_ica_ind)
        samps = kde.sample(N_samp)#2047 samples for the jth k bin
        X_ref_ica_unmixed[:,j] = np.ndarray.flatten(samps)

    X_ref_ica = np.dot(X_ref_ica_unmixed,A_.T) #applying the mixing matrix to undo the ICA transformation

    kl_ref, kl_data = [], []

    for i in range(n_sample):
        
        if verbose:
            print(i)
        Y_ref_ica_samp = Y_ref_ica[i]
        kl_ref.append(kNNdiv_general(X_ref_ica, Y_ref_ica_samp, Knn=k, alpha=None, div_func='kl'))
        kl_data.append(kNNdiv_general(X, Y_ref_ica_samp, Knn=k, alpha=None, div_func='kl'))   
        
    return kl_ref, kl_data     