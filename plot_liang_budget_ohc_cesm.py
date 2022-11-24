#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Map Liang index Qnet / OHTc --> OHCt
    CESM1-CAM5-SE
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
    Fig. A2: CESM1-LR, upper 50m
    Fig. A6: CESM1-HR, upper 50m
PROGRAMMER
    D. Docquier
LAST UPDATE
    23/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from scipy import interpolate
from scipy.ndimage import median_filter

# Options
save_fig = True
model = 'CESM1-CAM5-SE-LR' # CESM1-CAM5-SE-LR; CESM1-CAM5-SE-HR
member = 'r1i1p1f1'
load_significance = True # True: load significance of Liang index
use_filter = True # True: use median filter for significant contours (not to have isolated significant points)
depth = 50 # 50m or 300m for the integration of ocean heat flux
nvar = 3 # number of variables

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/Models/'

# Load latitude and longitude from CAM5 grid
grid = 'gn'
filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_' + member + '_' + grid + '_1988-2017.nc'
fh = Dataset(filename, mode='r')
lat_cam = fh.variables['lat'][:]
lon_cam = fh.variables['lon'][:]
fh.close()

# Load latitude and longitude from IFS grid
if model == 'CESM1-CAM5-SE-LR':
    model_interp = 'EC-Earth3P'
elif model == 'CESM1-CAM5-SE-HR':
    model_interp = 'EC-Earth3P-HR'
filename = '/ec/res4/hpcperm/cvaf/' + model_interp + '/r1i1p2f1/hfls/hfls_Amon_' + model_interp + '_hist-1950_r1i1p2f1_gr_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape
    
# Load Liang index and interpolate onto EC-Earth3 grid
filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
tau_init = np.load(filename_liang,allow_pickle=True)[0]
tau = np.zeros((ny,nx,nvar,nvar),dtype='float32')
for i in np.arange(nvar):
    print(i)
    for j in np.arange(nvar): 
       tau[:,:,i,j] = interpolate.griddata((lon_cam,lat_cam),tau_init[:,i,j],(lon,lat),method='linear')

# Load significance of Liang index and interpolate onto EC-Earth3 grid
if load_significance == True:
    filename_sig = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_alpha005.npy'
    sig_tau_fdr_init = np.load(filename_sig,allow_pickle=True)[0]
    sig_tau_fdr = np.zeros((ny,nx,nvar,nvar),dtype='float32')
    for i in np.arange(nvar):
        print(i)
        for j in np.arange(nvar): 
           sig_tau_fdr[:,:,i,j] = interpolate.griddata((lon_cam,lat_cam),sig_tau_fdr_init[:,i,j],(lon,lat),method='linear')

    # Image filtering
    if use_filter == True:
        for i in np.arange(nvar):
            print(i)
            for j in np.arange(nvar):
                sig_tau_fdr[:,:,i,j] = median_filter(sig_tau_fdr[:,:,i,j],size=3,mode='nearest')

# Cartopy projection
proj = ccrs.Robinson()

# Palettes
palette_tau = plt.cm.seismic._resample(20)
min_tau = -30.
max_tau = 30.


########
# Maps #
########

# Names of models
if model == 'CESM1-CAM5-SE-LR':
    model_name = 'CESM1-LR'
elif model == 'CESM1-CAM5-SE-HR':
    model_name = 'CESM1-HR'

## Relative transfer of information
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau OHTc-->OHCt
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau[:,:,1,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
if load_significance == True:
    ax1.contour(lon,lat,sig_tau_fdr[:,:,1,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$\tau_{OHTc \longrightarrow OHCt}$ 1988-2017 - ' + model_name,fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{OHTc \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# tau Qnet->OHCt
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau[:,:,2,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
if load_significance == True:
    ax2.contour(lon,lat,sig_tau_fdr[:,:,2,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$\tau_{Qnet \longrightarrow OHCt}$ 1988-2017 - ' + model_name,fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{Qnet \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# Save figure
if save_fig == True:
    if model == 'CESM1-CAM5-SE-LR':
        fig.savefig(dir_fig + 'fig_a2.png')
    elif model == 'CESM1-CAM5-SE-HR':
        fig.savefig(dir_fig + 'fig_a6.png')