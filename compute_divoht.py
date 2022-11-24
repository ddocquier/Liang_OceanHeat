#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute horizontal OHT divergence at every grid point (integrated over the vertical)
    EC-Earth3P or HadGEM3-GC31
PROGRAMMER
    D. Docquier
LAST UPDATE
    21/10/2022
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

# Function to compute moving average and shift to the left (to get T on u-grid)
def movingaverage(variable,window_size):
    window = np.ones(int(window_size))/float(window_size)
    runningmean_u = np.apply_along_axis(lambda m: np.convolve(m,window,'same'),axis=1,arr=variable)
    runningmean_u[:,:-1] = runningmean_u[:,1:]
    runningmean_v = np.apply_along_axis(lambda m: np.convolve(m,window,'same'),axis=0,arr=variable)
    runningmean_v[:-1,:] = runningmean_v[1:,:]
    return runningmean_u,runningmean_v

# Load grid size in X (gridx)
filename = dir_grid + 'gridx_' + model + '.nc'
fh = Dataset(filename, mode='r')
gridx = fh.variables['dx'][:]
fh.close()

# Load grid size in Y (gridy)
filename = dir_grid + 'gridy_' + model + '.nc'
fh = Dataset(filename, mode='r')
gridy = fh.variables['dy'][:]
fh.close()

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

# Initialization of OHT divergence
if model == 'EC-Earth3P-HR' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
    div_oht = np.zeros((nm,ny,nx),dtype='float32')
else:
    div_oht = np.zeros((nm,ny,nx))

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

    # Retrieve zonal velocity u (m/s)
    filename = dir_input + 'uo/uo_Omon_' + model + '_' + period + '_' + member + '_gn_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
    fh = Dataset(filename, mode='r')
    u = fh.variables['uo'][:,0:nz,:,:]
    if model == 'EC-Earth3P-HR' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
        if depth == 50:
            u = np.array(u,dtype='float32')
        else:
            u = np.array(u,dtype='float16')
    fh.close()

    # Retrieve meridional velocity v (m/s)
    filename = dir_input + 'vo/vo_Omon_' + model + '_' + period + '_' + member + '_gn_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
    fh = Dataset(filename, mode='r')
    v = fh.variables['vo'][:,0:nz,:,:]
    if model == 'EC-Earth3P-HR' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
        if depth == 50:
            v = np.array(v,dtype='float32')
        else:
            v = np.array(v,dtype='float16')
    fh.close()

    # Retrieve potential temperature (degC)
    filename = dir_input + 'thetao/thetao_Omon_' + model + '_' + period + '_' + member + '_gn_' + str(start_year+year) + '01-' + str(start_year+year) + '12.nc'
    fh = Dataset(filename, mode='r')
    temp = fh.variables['thetao'][:,0:nz,:,:]
    if model == 'EC-Earth3P-HR' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
        if depth == 50:
            temp = np.array(temp,dtype='float32')
        else:
            temp = np.array(temp,dtype='float16')
    fh.close()

    # Compute OHT
    if model == 'EC-Earth3P-HR' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
        temp_u = np.zeros((nmy,nz,ny,nx),dtype='float32')
        div_ut = np.zeros((nmy,nz,ny,nx),dtype='float32')
        temp_v = np.zeros((nmy,nz,ny,nx),dtype='float32')
        div_vt = np.zeros((nmy,nz,ny,nx),dtype='float32')
    else:
        temp_u = np.zeros((nmy,nz,ny,nx))
        div_ut = np.zeros((nmy,nz,ny,nx))
        temp_v = np.zeros((nmy,nz,ny,nx))
        div_vt = np.zeros((nmy,nz,ny,nx))
    for t in np.arange(nmy):
        for z in np.arange(nz):
            temp_u[t,z,:,:],temp_v[t,z,:,:] = movingaverage(temp[t,z,:,:],2)
            
            div_ut[t,z,:,1:nx-1] = (u[t,z,:,2:nx] * temp_u[t,z,:,2:nx] - u[t,z,:,0:nx-2] * temp_u[t,z,:,0:nx-2]) / (2.*gridx[:,1:nx-1])
            mask_temp_u = np.zeros((ny,nx))
            mask_temp_u[(temp_u[t,z,:,:] >= -3000.) * (temp_u[t,z,:,:] <= 3000.)] = 1
            div_ut[t,z,:,:] = div_ut[t,z,:,:] * mask_temp_u
            mask_ut = np.zeros((ny,nx))
            mask_ut[(div_ut[t,z,:,:] >= -3000.) * (div_ut[t,z,:,:] <= 3000.)] = 1
            
            div_vt[t,z,1:ny-1,:] = (v[t,z,2:ny,:] * temp_v[t,z,2:ny,:] - v[t,z,0:ny-2,:] * temp_v[t,z,0:ny-2,:]) / (2.*gridy[1:ny-1,:])
            mask_temp_v = np.zeros((ny,nx))
            mask_temp_v[(temp_v[t,z,:,:] >= -3000.) * (temp_v[t,z,:,:] <= 3000.)] = 1
            div_vt[t,z,:,:] = div_vt[t,z,:,:] * mask_temp_v
            mask_vt = np.zeros((ny,nx))
            mask_vt[(div_vt[t,z,:,:] >= -3000.) * (div_vt[t,z,:,:] <= 3000.)] = 1
            
            mask_gridz = np.zeros((ny,nx))
            mask_gridz[(gridz[z,:,:] >= 0.) * (gridz[z,:,:] <= 1.e4)] = 1
            
            dwkh = (div_ut[t,z,:,:] * mask_ut + div_vt[t,z,:,:] * mask_vt) * gridz[z,:,:] * mask_gridz
            dwkh[np.isnan(dwkh)] = 0.
            div_oht[year*nmy+t,:,:] = div_oht[year*nmy+t,:,:] + dwkh * rho * cp

    del u,v,temp,temp_u,div_ut,temp_v,div_vt

# Save variables
if save_var == True:
    if use_control == True:
        filename = dir_input + 'OHT/divOHT_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_' + str(start_year) + '-' + str(end_year) + '.npy'
    else:
        filename = dir_input + 'OHT/divOHT_' + model + '_' + member + '_' + str(depth) + 'm_' + str(start_year) + '-' + str(end_year) + '.npy'
    np.save(filename,[div_oht])
