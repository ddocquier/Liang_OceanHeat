#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
GOAL
    Save ocean heat content: cp * rho * T * dz [J/m^2]
    Product: ORAS5 (0.25deg)
    cp = specific ocean heat capacity = 3985 J kg^{-1} K^{-1} [Lien et al., 2017]
    rho = seawater density = 1027 kg m^{-3} [Lien et al., 2017]
    T = potential temperature [K]
    dz = vertical grid size [m]
PROGRAMMER
    D. Docquier
LAST UPDATE
    25/11/2021
'''

# Options
save_var = True
start_year = 1988
end_year = 2017
member = 'opa0'
depth = 50 # 50m or 300m
if depth == 50:
    nz = 18 # number of vertical levels
elif depth == 300:
    nz = 35 # number of vertical levels

# Standard libraries
from netCDF4 import Dataset
import numpy as np

# Working directories
dir_input = '/scratch/ms/be/cvaf/ORAS5/'
dir_output = '/perm/ms/be/cvaf/ROADMAP/Air-Sea/output/'

# Load grid sizes in X (gridx), Y (gridy) and Z (gridz)
filename = dir_input + 'mesh_mask.nc'
fh = Dataset(filename, mode='r')
gridx = fh.variables['e1u'][0,:,:]
gridz = fh.variables['e3t_0'][0,:]
tmask = fh.variables['tmask'][0,:,:,:]
ny,nx = gridx.shape
fh.close()

# Constant parameters
rho = 1027. # seawater density (1027 kg m^{-3} [Lien et al., 2017])
cp = 3985. # specific ocean heat capacity (3985 J kg^{-1} K^{-1} [Lien et al., 2017])

# Time parameters
nyears = int(end_year-start_year+1)
nmy = int(12)
nm = nyears * nmy

# Loop over years
ohc = np.zeros((nm,ny,nx))
for year in np.arange(nyears):
    print(start_year+year)
    
    # Loop over months
    for mon in np.arange(nmy):
        
        # Set month string
        if (mon+1) < 10:
            month = '0' + str(mon+1)
        else:
            month = str(mon+1)

        # Load potential temperature (degC)
        filename = dir_input + str(member) + '/votemper/thetao_ORAS5_1m_' + str(start_year+year) + month + '_300m.nc'
        fh = Dataset(filename, mode='r')
        temp = fh.variables['votemper'][0,:,:,:]
        temp = temp + 273.15 # converto into K
        fh.close()
    
        # Compute ocean heat content (OHC) in each grid cell, integrated over the upper ocean
        for z in np.arange(nz):
            ohc[year*nmy+mon,:,:] = ohc[year*nmy+mon,:,:] + temp[z,:,:] * tmask[z,:,:] * gridz[z] * rho * cp
    
# Save variables
if save_var == True:
    filename = dir_output + 'OHC_ORAS5_' + str(member) + '_' + str(depth) + 'm_1988-2017.npy'
    np.save(filename,[ohc])