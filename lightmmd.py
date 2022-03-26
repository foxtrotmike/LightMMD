# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:51:11 2022
Private code file. 
For information, contact: fayyaz.minhas@warwick.ac.uk
@author: fayya
"""

from timeit import default_timer as timer

import numpy as np
import math
import torch

def norm_sq(x):
  return (x**2).sum()

def distance(x,y):
    new_size = [x.size()[0]] + list(y.size())
    d = x.unsqueeze(1).expand(*new_size) - y.unsqueeze(0).expand(*new_size)
    return d.pow(2)
  

def mmd_rbf(x, y, sigma):
    
    gamma = 1/(2*(sigma**2))
    
    Kxx = (-gamma * distance(x,x).sum(dim=2)).exp()
    kyy = (-gamma * distance(y,y).sum(dim=2)).exp()
    kxy = (-gamma * distance(x,y).sum(dim=2)).exp()
    
    return torch.mean(Kxx) + torch.mean(kyy) - 2*torch.mean(kxy)
  

def mmd_fourier(x1, x2, sigma, dim_r=1024):
    """
    Approximate RBF kernel by random features
    """
    rnd_a = torch.empty((x1.size()[1], dim_r)).normal_()
    rnd_b = torch.empty(dim_r).uniform_()

    rW_n = 1/sigma * rnd_a
    rb_u = 2 * math.pi * rnd_b
    rf0 = math.sqrt(2/dim_r) * torch.cos(x1.mm(rW_n) + rb_u.expand(x1.size()[0], dim_r))
    rf1 = math.sqrt(2/dim_r) * torch.cos(x2.mm(rW_n) + rb_u.expand(x2.size()[0], dim_r))
    
    k0=1
    nPos = x1.size(0)
    nNeg = x2.size(0)
    # biased, unbiased
    return norm_sq(rf0.mean(0) - rf1.mean(0)), norm_sq(rf0.mean(0) - rf1.mean(0)) +  norm_sq(rf0.mean(0))/(nPos-1) + norm_sq(rf1.mean(0))/(nNeg-1) - k0*(nPos+nNeg-2)/(nPos-1)/(nNeg-1);


def compare_mmd_estimators(p, q):
  print('MMD computation:')
  a, b, sigmas = [], [], []
  for sigma in (-1 + np.arange(0, 21)*0.2):
    sigma = 10**sigma
    a_ = mmd_rbf(p, q, sigma=sigma)
    
    # mmd_fourier gives each time slightly different result because it samples
    # a new random projection each time
    b_, u_ = mmd_fourier(p, q, sigma, dim_r=1024)
    print("sigma: %.5f\tRBF: %.5f\tFourier: %.5f" % (sigma, a_.cpu().data.numpy(), b_.cpu().data.numpy()))
    a.append(a_)
    b.append(u_)
    sigmas.append(sigma)
  return a, b, sigmas

dd = 10
nn = 320
p = torch.randn((nn,dd))
q = 3 + torch.randn((nn,dd))
a,b,sigmas = compare_mmd_estimators(p, q)

import matplotlib.pyplot as plt
plt.plot(sigmas, np.sqrt(a))
plt.plot(sigmas, np.sqrt(b))
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np


from lightonml.encoding.models import EncoderDecoder
from lightonml.encoding.base import MultiThresholdEncoder
from lightonml.encoding.models import train
from lightonml.projections.sklearn import OPUMap
class LightMMD:
    def __init__(self, ):
        pass

    def fit(self,X, bits_per_feature = 16, n_components = 4096):        
        self.n_features = X.shape[1]
        self.bits_per_feature = bits_per_feature
        
        self.n_components = n_components
        self.encoder = MultiThresholdEncoder(thresholds='linspace', n_bins=bits_per_feature, columnwise=False).fit(X)#EncoderDecoder(self.n_features, self.n_features * self.bits_per_feature)

        Xenc = self.encoder.transform(X)
        start = timer()
        self.random_mapping = OPUMap(n_components=self.n_components, linear = False,ndims=1, simulated = False, max_n_features=Xenc.shape[1]).fit(Xenc)        
        end = timer()
        print("OPUMap Fit Time",end - start) 
        return self
    def mmd(self,p,q,gamma = 1.0):
        def preprocess(x):
            xe = self.encoder.transform(x)      
            start = timer()
            xt = self.random_mapping.transform(xe)
            end = timer()
            print("Transform Time",end - start) 
            xt = xt*np.sqrt(gamma)#*x.shape[1]            
            #import pdb;pdb.set_trace()
            f = np.hstack((np.cos(xt),np.sin(xt)))
            return np.sqrt(2/(xt.shape[1]))*f
        pt = preprocess(p)
        qt = preprocess(q)
        phi_p = np.mean(pt,axis=0)
        phi_q = np.mean(qt,axis=0)
        d = np.linalg.norm(phi_p-phi_q)
        return d


#%%
mmd = LightMMD().fit(np.vstack((p,q)))
G = sigmas#[0.00001,0.0001,0.001,0.01,0.1,1.0,2.0,4.0,8.0,16.0]
D = []
start = timer()
for s in sigmas:
    g = 1/(2*((s)**2))
    d = mmd.mmd(p,q,gamma = g)
    D.append(d)
end = timer()
print("Time",end - start) 
plt.figure()
plt.plot(sigmas,D,'o-')
mmd.random_mapping.close()
