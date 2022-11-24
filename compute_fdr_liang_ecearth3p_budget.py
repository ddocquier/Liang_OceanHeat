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
model = 'EC-Earth3P' # EC-Earth3P; EC-Earth3P-HR; HadGEM3-GC31-LL; HadGEM3-GC31-MM; HadGEM3-GC31-HM
if model == 'EC-Earth3P' or model == 'EC-Earth3P-HR':
    member = 'r1i1p2f1'
else:
    member = 'r1i1p1f1'
alpha_fdr = 0.05 # alpha of FDR
nvar = 3 # number of variables
n_iter = 500 # number of bootstrap realizations
depth = 50 # 50 or 300

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'

# Load latitude and longitude
if model == 'EC-Earth3P' or model == 'EC-Earth3P-HR':
    grid = 'gr'
elif model == 'HadGEM3-GC31-LL' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
    grid = 'gn'
filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape

# File names
filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_01_1988-2017.npy'
filename_sig = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_alpha005.npy'
    
# Load Liang index and 1st boostraped value
boot_tau = np.zeros((ny,nx,n_iter,nvar,nvar),dtype='float32')
tau,boot_tau[:,:,0,:,:] = np.load(filename_liang,allow_pickle=True)

# Load bootstraped values
for i in np.arange(n_iter-1):
    print(i)
    if i < 8:
        filename = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(i+2) + '_1988-2017.npy'
    else:
        filename = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(i+2) + '_1988-2017.npy'
    boot_tau[:,:,i+1,:,:] = np.load(filename,allow_pickle=True)
    
# Compute p value of tau
pval_tau = np.zeros((ny,nx,nvar,nvar))
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        for i in np.arange(nvar):
            for j in np.arange(nvar):
                pval_tau[y,x,i,j] = (np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) >= np.abs(tau[y,x,i,j]))  \
                    + np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) <= -np.abs(tau[y,x,i,j]))) / n_iter

# Clear variables
del boot_tau

# Compute FDR
sig_tau_fdr = np.zeros((ny,nx,nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        pval_tau_1d = np.ravel(pval_tau[:,:,i,j])
        sig_tau_fdr_init = multipletests(pval_tau_1d,alpha=alpha_fdr,method='fdr_bh')[0]
        sig_tau_fdr_init[sig_tau_fdr_init==True] = 1
        sig_tau_fdr_init[sig_tau_fdr_init==False] = 0
        sig_tau_fdr[:,:,i,j] = np.reshape(sig_tau_fdr_init,(ny,nx))

# Save variables
np.save(filename_sig,[sig_tau_fdr])