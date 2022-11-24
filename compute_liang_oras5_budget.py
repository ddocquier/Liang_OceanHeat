#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute Liang index OHCt - OHTc - Qnet
    Product: ORAS5 (0.25deg)
    Ocean heat budget based on Roberts et al. (2017)
    OHCt: ocean heat content tendency (dOHC/dt)
    OHTc: ocean heat transport convergence
    Qnet: net downward surface heat flux, which includes both solar (shortwave) and non-solar heat fluxes; it is positive downwards
    Non-solar heat flux = net longwave radiation + sensible heat flux + latent heat flux + heat flux associated with sea-ice freezing/melting + heat flux associated with melting of snow falling over the ocean and of the ice runoff (https://www.cmcc.it/wp-content/uploads/2015/02/rp0248-ans-12-2014.pdf)
PROGRAMMER
    D. Docquier
LAST UPDATE
    17/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import seasonal_decompose
import sys

# Import my functions
sys.path.append('/home/cvaf/Air-Sea/')
from function_liang_nvar2 import compute_liang_nvar

# Options
boot_iter = int(sys.argv[1]) # bootstrap realization (to be used in the job file)
save_qnet = False # True: save Qnet; False: use existing file
nvar = 3 # number of variables
depth = 50 # depth for OHT and OHC integrations (50m or 300m)

# Time parameters
nyears = int(2017-1988+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months
dt = 1 # time step
mon_to_sec = 30.4375 * 24. * 60. * 60. # conversion /month to /sec

# Working directories
dir_oras5 = '/ec/res4/hpcperm/cvaf/ORAS5/'
dir_output = '/perm/cvaf/ROADMAP/Air-Sea/output/'
dir_output2 = '/ec/res4/hpcperm/cvaf/ROADMAP/ORAS5/'

# Load OHT divergence from ORAS5
filename = dir_output + 'div_oht_ORAS5_opa0_' + str(depth) + 'm_1988-2017.npy'
OHTd = np.load(filename,allow_pickle=True)[0]
OHTd[OHTd==0.] = np.nan
OHTd[OHTd<-1.e5] = np.nan
OHTd[OHTd>1.e5] = np.nan
ny = np.size(OHTd,1)
nx = np.size(OHTd,2)

# Compute OHT convergence (inverse of divergence)
OHTc = -OHTd

# Load OHC
filename = dir_output + 'OHC_ORAS5_opa0_' + str(depth) + 'm_1988-2017.npy'
OHC = np.load(filename,allow_pickle=True)[0]
OHC = np.array(OHC,dtype='float32')

# Load latitude and longitude from ORAS5
filename = dir_oras5 + 'mesh_mask.nc'
fh = Dataset(filename, mode='r')
lon = fh.variables['nav_lon'][:]
lat = fh.variables['nav_lat'][:]
fh.close()

# Initialize variables (with zeroes)
if save_qnet == True:
    Qnet = np.zeros((nm,ny,nx),dtype='float32')
OHCt = np.zeros((nm,ny,nx),dtype='float32')
tau = np.zeros((ny,nx,nvar,nvar))
boot_tau = np.zeros((ny,nx,nvar,nvar))

# File names
if boot_iter < 10:
    filename_liang = dir_output2 + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017.npy'
else:
    filename_liang = dir_output2 + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017.npy'
filename_qnet = dir_output + 'Qnet_ORAS5_1988-2017.npy'

# Loop over years
if save_qnet == True:
    for year in np.arange(nyears):
        print(1988+year)
    
        # Load Qnet from ORAS5
        for i in np.arange(nmy):
            if (i+1) < 10:
                filename = dir_oras5 + 'opa0/sohefldo/sohefldo_ORAS5_1m_' + str(1988+year) + '0' + str(i+1) + '_grid_T_02.nc'
            else:
                filename = dir_oras5 + 'opa0/sohefldo/sohefldo_ORAS5_1m_' + str(1988+year) + str(i+1) + '_grid_T_02.nc'       
            fh = Dataset(filename,mode='r')
            Qnet[year*nmy+i,:,:] = fh.variables['sohefldo'][0,:,:]
            Qnet[Qnet>1.e4] = np.nan
            fh.close()

    # Save Qnet in a file
    np.save(filename_qnet,Qnet)

else:
    Qnet = np.load(filename_qnet,allow_pickle=True)
    
# Remove trend and seasonality of OHC, Qnet and OHTc, compute OHC trend and compute Liang index in each grid point
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        if np.count_nonzero(np.isnan(OHTc[:,y,x])) >= 1 or np.count_nonzero(np.isnan(OHC[:,y,x])) >= 1 or np.count_nonzero(np.isnan(Qnet[:,y,x])) >= 1:
            tau[y,x,:,:] = np.nan
        else:
            OHCt[1:nm-1,y,x] = (OHC[2:nm,y,x] - OHC[0:nm-2,y,x]) / (2.*dt*mon_to_sec)
            OHCt_resid = seasonal_decompose(OHCt[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            OHTc_resid = seasonal_decompose(OHTc[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            Qnet_resid = seasonal_decompose(Qnet[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            xx = np.array((OHCt_resid,OHTc_resid,Qnet_resid))
            notused,tau[y,x,:,:],notused,boot_tau[y,x,:,:] = compute_liang_nvar(xx,dt)
    
# Save variables
if boot_iter == 1:
    np.save(filename_liang,[tau,boot_tau])
else:
    np.save(filename_liang,[boot_tau])