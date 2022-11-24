#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute Liang index OHCt-OHTc-Qnet
    CESM1-CAM5-SE (interpolation of ocean fields onto CAM5 grid)
    Ocean heat budget based on Roberts et al. (2017)
    OHCt: ocean heat content tendency (dOHC/dt)
    OHTc: ocean heat transport convergence
    Qnet: net downward surface heat flux, which includes both solar (shortwave) and non-solar heat fluxes; it is positive downwards
    Non-solar heat flux = net longwave radiation + sensible heat flux + latent heat flux
PROGRAMMER
    D. Docquier
LAST UPDATE
    23/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
from scipy.interpolate import griddata

# Import my functions
sys.path.append('/home/cvaf/Air-Sea/')
from function_liang_nvar2 import compute_liang_nvar

# Options
boot_iter = int(sys.argv[1]) # bootstrap realization (to be used in the job file)
model = 'CESM1-CAM5-SE-LR' # CESM1-CAM-SE-LR or CESM1-CAM-SE-HR
interpolation_oht = False # True: interpolate OHTc from ocean onto atmospheric grid; False: use existing file
interpolation_ohc = False # True: interpolate OHC from ocean onto atmospheric grid; False: use existing file
save_qnet = False # True: save Qnet; False: use existing file
member = 'r1i1p1f1'
nvar = 3 # number of variables
depth = 50 # OHT and OHC integrations (50m or 300m)

# Time parameters
nyears = int(2017-1988+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months
dt = 1 # time step
mon_to_sec = 30.4375 * 24. * 60. * 60. # conversion /month to /sec

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'

# Load latitude and longitude from atmospheric grid
filename = dir_input + member + '/rlds/rlds_Amon_' + model + '_' + member + '_gn_1988-2017.nc'
fh = Dataset(filename, mode='r')
lat = fh.variables['lat'][:]
lon = fh.variables['lon'][:]
fh.close()
ncol = lon.shape[0]

# Load latitude and longitude from ocean grid
filename = dir_input + 'grid/gridx_' + model + '.nc'
fh = Dataset(filename, mode='r')
lat_pop = fh.variables['lat'][:]
lon_pop = fh.variables['lon'][:]
fh.close()

# Interpolate OHT convergence from ocean onto atmospheric grid
filename_interp = dir_input + member + '/OHT/divOHT_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp.npy'
if interpolation_oht == True:
    filename = dir_input + member + '/OHT/divOHT_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
    if model == 'CESM1-CAM5-SE-LR':
        oht,notused,notused = np.load(filename,allow_pickle=True)
        oht_interp = np.zeros((nm,ncol),dtype='float32')
    else:
        oht = np.load(filename,allow_pickle=True)[0]
        oht_interp = np.zeros((nm,ncol),dtype='float16')
    for i in np.arange(nm):
        print(i)
        oht_interp[i,:] = griddata((lon_pop.ravel(),lat_pop.ravel()),oht[i,:,:].ravel(),(lon,lat),method='linear')
    np.save(filename_interp,[oht_interp])
    del oht
else:  
    oht_interp = np.load(filename_interp,allow_pickle=True)[0]
oht_interp[oht_interp==0.] = np.nan
oht_interp = -oht_interp

# Interpolate OHC from ocean onto atmospheric grid
filename_interp = dir_input + member + '/OHC/OHC_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp.npy'
if interpolation_ohc == True:
    filename = dir_input + member + '/OHC/OHC_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
    if model == 'CESM1-CAM5-SE-LR':
        ohc,notused,notused = np.load(filename,allow_pickle=True)
        ohc_interp = np.zeros((nm,ncol),dtype='float32')
    else:
        ohc = np.load(filename,allow_pickle=True)[0]
        ohc_interp = np.zeros((nm,ncol),dtype='float32')
    for i in np.arange(nm):
        print(i)
        ohc_interp[i,:] = griddata((lon_pop.ravel(),lat_pop.ravel()),ohc[i,:,:].ravel(),(lon,lat),method='linear')
    np.save(filename_interp,[ohc_interp])
    del ohc
else:
    ohc_interp = np.load(filename_interp,allow_pickle=True)[0]
ohc_interp[ohc_interp==0.] = np.nan

# Initialize variables (with zeroes)
Qnet = np.zeros((nm,ncol),dtype='float32')
OHCt = np.zeros((nm,ncol),dtype='float32')
tau = np.zeros((ncol,nvar,nvar))
boot_tau = np.zeros((ncol,nvar,nvar))

# Compute and save Qnet
filename_qnet = dir_input + member + '/Qnet/Qnet_' + model + '_' + member + '_1988-2017.npy'
if save_qnet == True:
    
    # Load LHF
    filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_' + member + '_gn_1988-2017.nc'
    fh = Dataset(filename, mode='r')
    lhf = fh.variables['hfls'][:]
    fh.close()
    
    # Load SHF
    filename = dir_input + member + '/hfss/hfss_Amon_' + model + '_' + member + '_gn_1988-2017.nc'
    fh = Dataset(filename, mode='r')
    shf = fh.variables['hfss'][:]
    fh.close()
    
    # Load LWFdown
    filename = dir_input + member + '/rlds/rlds_Amon_' + model + '_' + member + '_gn_1988-2017.nc'
    fh = Dataset(filename, mode='r')
    lwfdown = fh.variables['rlds'][:]
    fh.close()
    
    # Load LWFup
    filename = dir_input + member + '/rlus/rlus_Amon_' + model + '_' + member + '_gn_1988-2017.nc'
    fh = Dataset(filename, mode='r')
    lwfup = fh.variables['rlus'][:]
    fh.close()
    
    # Load SWFdown
    filename = dir_input + member + '/rsds/rsds_Amon_' + model + '_' + member + '_gn_1988-2017.nc'
    fh = Dataset(filename, mode='r')
    swfdown = fh.variables['rsds'][:]
    fh.close()
    
     # Load SWFup
    filename = dir_input + member + '/rsus/rsus_Amon_' + model + '_' + member + '_gn_1988-2017.nc'
    fh = Dataset(filename, mode='r')
    swfup = fh.variables['rsus'][:]
    fh.close()
    
    # Compute net surface heat flux (Qnet)
    Qnet = lwfdown - lwfup + swfdown - swfup - lhf - shf
    
    # Delete variables
    del lhf,shf,lwfdown,lwfup,swfdown,swfup
    
    # Save Qnet in a file
    np.save(filename_qnet,[Qnet])
    
else:
    
    # Load Qnet
    Qnet = np.load(filename_qnet,allow_pickle=True)[0]

# Remove trend and seasonality of OHC, Qnet and OHTc, compute OHC tendency and compute Liang index in each grid point
for y in np.arange(ncol):
    print(y)
    if np.count_nonzero(np.isnan(ohc_interp[:,y])) >= 1 or np.count_nonzero(np.isnan(oht_interp[:,y])) >= 1 or np.count_nonzero(np.isnan(Qnet[:,y])) >= 1:
        tau[y,:,:] = np.nan
    else:
        OHCt[1:nm-1,y] = (ohc_interp[2:nm,y] - ohc_interp[0:nm-2,y]) / (2.*dt*mon_to_sec)
        OHCt_resid = seasonal_decompose(OHCt[:,y],model='additive',period=12,extrapolate_trend='freq').resid
        Qnet_resid = seasonal_decompose(Qnet[:,y],model='additive',period=12,extrapolate_trend='freq').resid
        OHTc_resid = seasonal_decompose(oht_interp[:,y],model='additive',period=12,extrapolate_trend='freq').resid
        xx = np.array((OHCt_resid,OHTc_resid,Qnet_resid))
        notused,tau[y,:,:],notused,boot_tau[y,:,:] = compute_liang_nvar(xx,dt)
    
# Save variables
if boot_iter < 10:
    filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017.npy'
else:
    filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017.npy'
if boot_iter == 1:
    np.save(filename_liang,[tau,boot_tau])
else:
    np.save(filename_liang,[boot_tau])