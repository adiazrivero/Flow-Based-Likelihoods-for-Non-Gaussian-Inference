import numpy as np
import pickle 
from NG_functions import whiten

def load_BOSS():
    
    dirii = '/n/home04/adiazrivero/likelihood_nongaussianity/MultiDark_powerspectra/ps_changhoon/'
    ks02 = np.load(dirii + 'ks_ell_0-2.npy')
    ks4 = np.load(dirii + 'ks_ell_4.npy')

    ks = np.concatenate(( np.concatenate((ks02,ks02)) , ks4))
    filelist = os.listdir(dirii)

    powerspectra = np.zeros((2049,37))
    count2 = 0
    for file in filelist:
        
        if file[0] == 'p':
            arr = np.load(dirii + file)
            arr2 = arr[~np.all(arr == 0, axis=1)]
            for i in arr2:
                powerspectra[count2] = i 
                count2 += 1

    powerspectra = powerspectra[~np.all(powerspectra == 0, axis=1)]
    arr2 = np.around(powerspectra,decimals=1)
    arr3,indices = np.unique(arr2,return_index=True,axis=0)

    X_pk = powerspectra[indices]
    X_res = X_pk - np.mean(X_pk,axis=0)
    X_w, W = whiten(X_res)
    n, dim = X_w.shape
    print("%i dimensional data with %i samples" % (dim, n))
    
    return X_pk, X_w, W, ks

def load_WL(ell_min=100,ell_max=1.25*1e4):
    
    cosmo = 'Om0.300_Ol0.700'
    rootdir = '/n/holyscratch01/dvorkin_lab/adiazrivero/new_WL_mocks09032020/new_WL_mocks_061120/storage/%s/512b240/' % cosmo
    arr = np.zeros((1,127))

    mapdirs = ['Maps/','Maps_2/','Maps_3/']
    indices = [[[1,5000],[5000,10000],[10000,15000],[15000,20000],[20000,26000]],
              [[1,5000],[5000,10000],[10000,15000],[15000,20000],[20000,22000]],
              [[1,5000],[5000,10000],[10000,15000],[15000,20000],[20000,25000],[25000,30000]]]

    for mapdir,ixs in zip(mapdirs,indices):

        for count,nums in enumerate(ixs):

            if count < 2:
                filename = 'powerspectra_l_256_%s_%s.pkl' % (nums[0],nums[1])
            else: 
                filename = 'powerspectra_l256_%s_%s.pkl' % (nums[0],nums[1])

            with open(rootdir + mapdir + filename, 'rb') as file:

                dicti = pickle.load(file,encoding='latin1')
                #print(dicti.keys())
                if count == 0:
                    ells = dicti['ells']

                arr = np.concatenate((arr,dicti[cosmo]))

    powerspectra = arr[1:]
    inds = (ell_min < ells) & (ells < ell_max)

    X_pk = powerspectra[:,inds]
    l = ells[inds]    
    
    if ell_max > 5*1e3:
        X_pk = X_pk[:,::2]
        l = l[::2]
    
    X_res = X_pk - np.mean(X_pk,axis=0)
    X_w, W = whiten(X_res)
    n, dim = X_w.shape
    print("%i dimensional data with %i samples" % (dim, n))

    return X_pk, X_w, W, l


def global_rescale(image, global_mini, global_maxi, interp_mini=0,interp_maxi=255):
    if interp_mini == 0 and interp_maxi == 255:
        return np.interp(image, (global_mini,global_maxi), (interp_mini,interp_maxi)).astype(int)
    else:
        return np.interp(image, (global_mini,global_maxi), (interp_mini,interp_maxi))
    
    