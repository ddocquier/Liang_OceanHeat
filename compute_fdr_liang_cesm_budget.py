#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute significance via False Discovery Rate (FDR; Wilks [2016])
    EC-Earth3P; HadGEM3-GC31
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
PROGRAMMER
    D. Docquier
LAST UPDATE
    14/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.stats.multitest import multipletests

# Options
model = 'CESM1-CAM5-SE-LR' # CESM1-CAM5-SE-LR; CESM1-CAM5-SE-HR
member = 'r1i1p1f1'
alpha_fdr = 0.05 # alpha of FDR
nvar = 3 # number of variables
n_iter = 500 # number of bootstrap realizations
depth = 50 # 50 or 300

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'

# Load latitude and longitude
grid = 'gn'
filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_' + member + '_' + grid + '_1988-2017.nc'
fh = Dataset(filename, mode='r')
lat = fh.variables['lat'][:]
lon = fh.variables['lon'][:]
fh.close()
ncol = lon.shape[0]

# File names
filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_01_1988-2017.npy'
filename_sig = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_alpha005.npy'
    
# Load Liang index and 1st boostraped value
boot_tau = np.zeros((ncol,n_iter,nvar,nvar),dtype='float32')
tau,boot_tau[:,0,:,:] = np.load(filename_liang,allow_pickle=True)

# Load bootstraped values
for i in np.arange(n_iter-1):
    print(i)
    if i < 8:
        filename = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(i+2) + '_1988-2017.npy'
    else:
        filename = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(i+2) + '_1988-2017.npy'
    boot_tau[:,i+1,:,:] = np.load(filename,allow_pickle=True)
    
# Compute p value of tau
pval_tau = np.zeros((ncol,nvar,nvar))
for y in np.arange(ncol):
    print(y)
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            pval_tau[y,i,j] = (np.count_nonzero((boot_tau[y,:,i,j]-tau[y,i,j]) >= np.abs(tau[y,i,j]))  \
                + np.count_nonzero((boot_tau[y,:,i,j]-tau[y,i,j]) <= -np.abs(tau[y,i,j]))) / n_iter

# Clear variables
del boot_tau

# Compute FDR
sig_tau_fdr = np.zeros((ncol,nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_fdr[:,i,j] = multipletests(pval_tau[:,i,j],alpha=alpha_fdr,method='fdr_bh')[0]

# Save variables
np.save(filename_sig,[sig_tau_fdr])