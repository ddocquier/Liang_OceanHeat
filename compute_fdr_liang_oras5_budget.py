#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute FDR significance
    ORAS5
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
PROGRAMMER
    D. Docquier
LAST UPDATE
    17/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.stats.multitest import multipletests

# Options
slice_y = 1 # 1 or 2
first_slice = 500
alpha_fdr = 0.05 # alpha of FDR
nvar = 3 # number of variables
n_iter = 500 # number of bootstrap realizations
depth = 50 # 50 or 300 m

# Working directories
dir_grid = '/ec/res4/hpcperm/cvaf/ORAS5/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/ORAS5/'

# Load latitude and longitude
filename = dir_grid + 'mesh_mask.nc'
fh = Dataset(filename, mode='r')
lon = fh.variables['nav_lon'][:]
lat = fh.variables['nav_lat'][:]
fh.close()
ny,nx = lon.shape

# File names
filename_liang = dir_output + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_01_1988-2017.npy'
filename_sig = dir_output + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_sig_fdr_1988-2017_slice_' + str(slice_y) +  '_alpha005.npy'  

# Load Liang index and 1st boostraped value
if slice_y == 1:
    ny_slice= first_slice
    start_slice = int(0)
    end_slice = int(ny_slice)
elif slice_y == 2:
    ny_slice = int(ny - first_slice)
    start_slice = int(first_slice)
    end_slice = int(ny)
boot_tau = np.zeros((ny_slice,nx,n_iter,nvar,nvar),dtype='float32')
tau_init,boot_init = np.load(filename_liang,allow_pickle=True)
#print(np.size(boot_init,0))
#print(np.size(boot_init[start_slice:end_slice,:,:,:],0))
#print(np.size(boot_tau,0))
boot_tau[:,:,0,:,:] = boot_init[start_slice:end_slice,:,:,:]
tau = tau_init[start_slice:end_slice,:,:,:]
del boot_init,tau_init

# Load bootstraped values
for i in np.arange(n_iter-1):
    print(i)
    if i < 8:
        filename = dir_output + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_0' + str(i+2) + '_1988-2017.npy'
    else:
        filename = dir_output + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_' + str(i+2) + '_1988-2017.npy'
    boot_init = np.load(filename,allow_pickle=True)[0]
    boot_tau[:,:,i+1,:,:] = boot_init[start_slice:end_slice,:,:,:]
    del boot_init
    
# Compute p value of tau
pval_tau = np.zeros((ny_slice,nx,nvar,nvar))
for y in np.arange(ny_slice):
    print(y)
    for x in np.arange(nx):
        for i in np.arange(nvar):
            for j in np.arange(nvar):
                pval_tau[y,x,i,j] = (np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) >= np.abs(tau[y,x,i,j]))  \
                    + np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) <= -np.abs(tau[y,x,i,j]))) / n_iter

# Clear variables
del boot_tau

# Compute FDR
sig_tau_fdr = np.zeros((ny_slice,nx,nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        pval_tau_1d = np.ravel(pval_tau[:,:,i,j])
        sig_tau_fdr_init = multipletests(pval_tau_1d,alpha=alpha_fdr,method='fdr_bh')[0]
        sig_tau_fdr_init[sig_tau_fdr_init==True] = 1
        sig_tau_fdr_init[sig_tau_fdr_init==False] = 0
        sig_tau_fdr[:,:,i,j] = np.reshape(sig_tau_fdr_init,(ny_slice,nx))

# Save variables
np.save(filename_sig,[sig_tau_fdr])