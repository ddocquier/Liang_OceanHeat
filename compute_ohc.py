#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute ocean heat content at every grid point (integrated over the vertical)
    cp * rho * T * dz [J/m^2]
    EC-Earth3P or HadGEM3-GC31
PROGRAMMER
    D. Docquier
LAST UPDATE
    24/10/2022
'''

# Standard libraries
from netCDF4 import Dataset
import numpy as np

# Options
model = 'EC-Earth3P-HR' # EC-Earth3P; EC-Earth3P-HR; HadGEM3-GC31-LL; HadGEM3-GC31-MM; HadGEM3-GC31-HM
use_control = True # True: control-1950; False: hist-1950
member = 'r1i1p2f1' # r1i1p2f1 for EC-Earth3P; r1i1p1f1 for HadGEM3-GC31
start_year = 1950 # 1988 (historical); 1950 (control)
end_year = 2049 # 2017 (historical); 2049 (control)
save_var = True
depth = 50 # 50m, 300m or 6000m (full depth)
if depth == 50:
    nz = 18 # number of vertical levels
elif depth == 300:
    nz = 35 # number of vertical levels
elif depth == 6000:
    nz = 75

# Time parameters
nyears = int(end_year-start_year+1)
nmy = int(12) # number of months in a year
nm = nyears * nmy

# Working directories
if use_control == True:
    dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/control-1950/' + member +  '/'
else:
    dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/' + member +  '/'
dir_grid = '/ec/res4/hpcperm/cvaf/' + model + '/grid/'

# Load grid size in Z (gridz)
filename = dir_grid + 'thkcello_' + model + '.nc'
fh = Dataset(filename, mode='r')
gridz = fh.variables['thkcello'][:]
gridz = gridz[0,0:nz,:,:]
notused,ny,nx = gridz.shape
fh.close()

# Constant parameters
rho = 1027. # seawater density (1027 kg m^{-3} [Lien et al., 2017])
cp = 3985. # specific ocean heat capacity (3985 J kg^{-1} K^{-1} [Lien et al., 2017])

# Initialization of OHC
if model == 'EC-Earth3P-HR' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
    ohc = np.zeros((nm,ny,nx),dtype='float32')
else:
    ohc = np.zeros((nm,ny,nx))

# Loop over years
for year in np.arange(nyears):
    print(start_year+year)

    if use_control == True:
        period = 'control-1950'
    else:
        if (start_year+year) <= 2014:
            period = 'hist-1950'
        else:
            period = 'highres-future'

    # Retrieve potential temperature (degC)
    filename = dir_input + 'thetao/thetao_Omon_' + model + '_' + period + '_' + member + '_gn_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
    fh = Dataset(filename, mode='r')
    temp = fh.variables['thetao'][:,0:nz,:,:]
    if model == 'EC-Earth3P-HR' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
        temp = np.array(temp,dtype='float32') + 273.15 # convert into K
    else:
        temp = temp + 273.15 # convert into K
    fh.close()

    # Compute OHC
    for t in np.arange(nmy):
        for z in np.arange(nz):
            mask_temp = np.zeros((ny,nx))
            mask_temp[(temp[t,z,:,:] >= -3000.) * (temp[t,z,:,:] <= 3000.)] = 1
            mask_gridz = np.zeros((ny,nx))
            mask_gridz[(gridz[z,:,:] >= 0.) * (gridz[z,:,:] <= 1.e4)] = 1
            dwkh = temp[t,z,:,:] * mask_temp * gridz[z,:,:] * mask_gridz
            dwkh[np.isnan(dwkh)] = 0.
            ohc[year*nmy+t,:,:] = ohc[year*nmy+t,:,:] + dwkh * rho * cp
    del temp

# Save variables
if save_var == True:
    if use_control == True:
        filename = dir_input + 'OHC/OHC_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_' + str(start_year) + '-' + str(end_year) + '.npy'
    else:
        filename = dir_input + 'OHC/OHC_' + model + '_' + member + '_' + str(depth) + 'm_' + str(start_year) + '-' + str(end_year) + '.npy'
    np.save(filename,[ohc])