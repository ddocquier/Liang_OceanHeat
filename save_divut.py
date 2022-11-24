#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
GOAL
    Save divergence of horizontal ocean heat flux: cp * rho * div(uT) * dz [W/m2]
    Product: ORAS5 (0.25deg)
    cp = specific ocean heat capacity = 3985 J kg^{-1} K^{-1} [Lien et al., 2017]
    rho = seawater density = 1027 kg m^{-3} [Lien et al., 2017]
    div(uT) = [u(i+1,j) * T(i+1,j) - u(i-1,j) * T(i-1,j)] / (2*dx) + [v(i,j+1) * T(i,j+1) - v(i,j-1) * T(i,j-1)] / (2*dy)
    u and v = ocean velocity on u-grid and v-grid, respectively [m s^{-1}]
    T = potential temperature, computed on the respective u and v grids [degC, as it is in fact the difference between T and Tref, the latter being usually set to 0 degC]
    dx = horizontal grid size in X direction [m]
    dy = horizontal grid size in Y direction [m]
    dz = vertical grid size [m]
PROGRAMMER
    D. Docquier
LAST UPDATE
    30/05/2022
'''

# Options
save_var = True
start_year = 1988
end_year = 2017
member = 'opa0'
nz = 35 # number of vertical levels (18: upper 50m; 31: upper 200m; 35: upper 300m)

# Standard libraries
from netCDF4 import Dataset
import numpy as np

# Function to compute moving average and shift to the left (to get T on u-grid) or bottom (to get T on v-grid)
# Arakawa C-grid (NEMO)
def movingaverage(variable,window_size):
    window = np.ones(int(window_size))/float(window_size)
    runningmean_u = np.apply_along_axis(lambda m: np.convolve(m,window,'same'),axis=1,arr=variable) # compute moving average of T on u-grid
    runningmean_u[:,:-1] = runningmean_u[:,1:] # shift to the left (to get T on u-grid)
    runningmean_v = np.apply_along_axis(lambda m: np.convolve(m,window,'same'),axis=0,arr=variable) # compute moving average of T on v-grid
    runningmean_v[:-1,:] = runningmean_v[1:,:] # shift to the bottom (to get T on v-grid)
    return runningmean_u,runningmean_v

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/ORAS5/'
dir_output = '/perm/cvaf/ROADMAP/Air-Sea/output/'

# Load grid sizes in X (gridx), Y (gridy) and Z (gridz)
filename = dir_input + 'mesh_mask.nc'
fh = Dataset(filename, mode='r')
gridx = fh.variables['e1u'][0,:,:]
gridy = fh.variables['e2u'][0,:,:]
gridz = fh.variables['e3t_0'][0,:]
tmask = fh.variables['tmask'][0,:,:,:]
umask = fh.variables['umask'][0,:,:,:]
vmask = fh.variables['vmask'][0,:,:,:]
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
div_oht = np.zeros((nm,ny,nx))
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
        fh.close()
        
        # Load zonal velocity u (m/s)
        filename = dir_input + str(member) + '/vozocrtx/uo_ORAS5_1m_' + str(start_year+year) + month + '_300m.nc'   
        fh = Dataset(filename, mode='r')
        u = fh.variables['vozocrtx'][0,:,:,:]
        fh.close()
        
        # Load meridional velocity v (m/s)
        filename = dir_input + str(member) + '/vomecrty/vo_ORAS5_1m_' + str(start_year+year) + month + '_300m.nc'   
        fh = Dataset(filename, mode='r')
        v = fh.variables['vomecrty'][0,:,:,:]
        fh.close()
    
        # Compute ocean heat flux divergence in each grid cell, integrated over the upper ocean
        temp_u = np.zeros((nz,ny,nx))
        temp_v = np.zeros((nz,ny,nx))
        div_uT = np.zeros((nz,ny,nx))
        div_vT = np.zeros((nz,ny,nx))
        for z in np.arange(nz):
            temp_u[z,:,:],temp_v[z,:,:] = movingaverage(temp[z,:,:]*tmask[z,:,:],2)
            div_uT[z,:,1:nx-1] = (u[z,:,2:nx] * temp_u[z,:,2:nx] - u[z,:,0:nx-2] * temp_u[z,:,0:nx-2]) / (2.*gridx[:,1:nx-1])
            div_vT[z,1:ny-1,:] = (v[z,2:ny,:] * temp_v[z,2:ny,:] - v[z,0:ny-2,:] * temp_v[z,0:ny-2,:]) / (2.*gridy[1:ny-1,:])
            dwkh = (div_uT[z,:,:] * umask[z,:,:] + div_vT[z,:,:] * vmask[z,:,:]) * gridz[z]
            div_oht[year*nmy+mon,:,:] = div_oht[year*nmy+mon,:,:] + dwkh * rho * cp
    
# Save variables
if save_var == True:
    if nz == 18:
        filename = dir_output + 'div_oht_ORAS5_' + str(member) + '_50m_1988-2017.npy'
    elif nz == 35:
        filename = dir_output + 'div_oht_ORAS5_' + str(member) + '_300m_1988-2017.npy'
    np.save(filename,[div_oht])