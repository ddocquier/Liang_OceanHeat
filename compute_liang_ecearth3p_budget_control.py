#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute Liang index OHCt - OHTc - Qnet
    EC-Earth3P-HR control-1950 (interpolation of ocean fields onto IFS grid)
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
model = 'EC-Earth3P-HR'
interpolation_oht = False # True: interpolate OHTc from NEMO onto IFS; False: use existing file
interpolation_ohc = False # True: interpolate OHC from NEMO onto IFS; False: use existing file
save_qnet = False # True: save Qnet; False: use existing file
member = 'r1i1p2f1' # member
start_year = 1950 # 1988 (historical); 1950 (control)
end_year = 2049 # 2017 (historical); 2049 (control)
nvar = 3 # number of variables
depth = 50 # depth for OHT and OHC integrations (50m, 300m or 6000m)

# Time parameters
nyears = int(2049-1950+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months
dt = 1 # time step
mon_to_sec = 30.4375 * 24. * 60. * 60. # conversion /month to /sec

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/control-1950/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/control-1950/'

# Load latitude and longitude from IFS grid
grid = 'gr'
filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_control-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape
    
# Load latitude and longitude from NEMO grid
filename = '/ec/res4/hpcperm/cvaf/' + model + '/grid/thkcello_' + model + '.nc'
fh = Dataset(filename, mode='r')
lat_nemo = fh.variables['latitude'][:]
lon_nemo = fh.variables['longitude'][:]
fh.close()

# Interpolate OHC from NEMO onto IFS grid
filename_interp = dir_input + member + '/OHC/OHC_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_' + str(start_year) +'-' + str(end_year) + '_interp.npy'
if interpolation_ohc == True:
    filename = dir_input + member + '/OHC/OHC_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_' + str(start_year) +'-' + str(end_year) + '.npy'
    ohc = np.load(filename,allow_pickle=True)[0]
    ohc_interp = np.zeros((nm,ny,nx))
    for i in np.arange(nm):
        print(i)
        ohc_interp[i,:,:] = griddata((lon_nemo.ravel(),lat_nemo.ravel()),ohc[i,:,:].ravel(),(lon,lat),method='linear')
    np.save(filename_interp,[ohc_interp])
    del ohc
else:
    ohc_interp = np.load(filename_interp,allow_pickle=True)[0]
ohc_interp[ohc_interp==0.] = np.nan

# Interpolate OHT convergence from NEMO onto IFS grid
filename_interp = dir_input + member + '/OHT/divOHT_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_' + str(start_year) +'-' + str(end_year) + '_interp.npy'
if interpolation_oht == True:
    filename = dir_input + member + '/OHT/divOHT_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_' + str(start_year) +'-' + str(end_year) + '.npy'
    oht = np.load(filename,allow_pickle=True)[0]
    oht_interp = np.zeros((nm,ny,nx))
    for i in np.arange(nm):
        print(i)
        oht_interp[i,:,:] = griddata((lon_nemo.ravel(),lat_nemo.ravel()),oht[i,:,:].ravel(),(lon,lat),method='linear')
    np.save(filename_interp,[oht_interp])
    del oht
else:
    oht_interp = np.load(filename_interp,allow_pickle=True)[0]
oht_interp[oht_interp==0.] = np.nan
oht_interp = -oht_interp

# File names
if boot_iter < 10:
    filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_' + str(start_year) +'-' + str(end_year) + '.npy'
else:
    filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_' + str(start_year) +'-' + str(end_year) + '.npy'
    
# Initialize variables (with zeroes)
Qnet = np.zeros((nm,ny,nx),dtype='float32')
OHCt = np.zeros((nm,ny,nx),dtype='float32')
tau = np.zeros((ny,nx,nvar,nvar))
boot_tau = np.zeros((ny,nx,nvar,nvar))

# Save Qnet - loop over years
if save_qnet == True:
    for year in np.arange(nyears):
        print(start_year+year)
        
        # Select scenario
        scenario = 'control-1950'
    
        # Load LHF
        filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        lhf = fh.variables['hfls'][:]
        fh.close()
    
        # Load SHF
        filename = dir_input + member + '/hfss/hfss_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        shf = fh.variables['hfss'][:]
        fh.close()
        
        # Load LWFdown
        filename = dir_input + member + '/rlds/rlds_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        lwfdown = fh.variables['rlds'][:]
        fh.close()
        
        # Load LWFup
        filename = dir_input + member + '/rlus/rlus_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        lwfup = fh.variables['rlus'][:]
        fh.close()
        
        # Load SWFdown
        filename = dir_input + member + '/rsds/rsds_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        swfdown = fh.variables['rsds'][:]
        fh.close()
        
         # Load SWFup
        filename = dir_input + member + '/rsus/rsus_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        swfup = fh.variables['rsus'][:]
        fh.close()
        
        # Compute net surface heat flux (Qnet)
        Qnet[year*nmy:year*nmy+nmy,:,:] = lwfdown - lwfup + swfdown - swfup - lhf - shf
        
        # Delete variables
        del lhf,shf,lwfdown,lwfup,swfdown,swfup
        
    # Save Qnet
    filename = dir_input + member + '/Qnet/Qnet_' + model + '_control-1950_' + member + '_' + str(start_year) +'-' + str(end_year) + '.npy'
    np.save(filename,[Qnet])

else:
    filename = dir_input + member + '/Qnet/Qnet_' + model + '_control-1950_' + member + '_' + str(start_year) +'-' + str(end_year) + '.npy'
    Qnet = np.load(filename,allow_pickle=True)[0]

# Compute OHC tendency, remove trend and seasonality of OHCt, Qnet and OHTc, and compute Liang index in each grid point
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        if np.count_nonzero(np.isnan(ohc_interp[:,y,x])) >= 1 or np.count_nonzero(np.isnan(oht_interp[:,y,x])) >= 1 or np.count_nonzero(np.isnan(Qnet[:,y,x])) >= 1:
            tau[y,x,:,:] = np.nan
        else:
            OHCt[1:nm-1,y,x] = (ohc_interp[2:nm,y,x] - ohc_interp[0:nm-2,y,x]) / (2.*dt*mon_to_sec)
            OHCt_resid = seasonal_decompose(OHCt[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            OHTc_resid = seasonal_decompose(oht_interp[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            Qnet_resid = seasonal_decompose(Qnet[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            xx = np.array((OHCt_resid,OHTc_resid,Qnet_resid))
            notused,tau[y,x,:,:],notused,boot_tau[y,x,:,:] = compute_liang_nvar(xx,dt)
    
# Save variables
if boot_iter == 1:
    np.save(filename_liang,[tau,boot_tau])
else:
    np.save(filename_liang,[boot_tau])