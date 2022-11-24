#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Map Liang index Qnet / OHTc --> OHCt
    ORAS5
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
    Fig. 3: upper 50m
    Fig. 6: upper 300m
PROGRAMMER
    D. Docquier
LAST UPDATE
    17/11/2022
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
from scipy.interpolate import griddata
from scipy.ndimage import median_filter

# Options
save_fig = True
save_var = True # save result of interpolation onto EC-Earth3-HR (in order to have a continuous plot with Cartopy)
use_filter = True # True: use median filter for significant contours (not to have isolated significant points)
depth = 50 # 50m or 300m for the integration of ocean heat flux
nvar = 3 # number of variables

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/ROADMAP/ORAS5/'
dir_grid = '/ec/res4/hpcperm/cvaf/ORAS5/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/Models/'

# Load latitude and longitude
filename = dir_grid + 'mesh_mask.nc'
fh = Dataset(filename, mode='r')
lon_nemo = fh.variables['nav_lon'][:]
lon_nemo[lon_nemo<0.] = lon_nemo[lon_nemo<0.] + 360.
lat_nemo = fh.variables['nav_lat'][:]
fh.close()

# Load latitude and longitude from IFS-HR grid
filename = '/ec/res4/hpcperm/cvaf/EC-Earth3P-HR/r1i1p2f1/hfls/hfls_Amon_EC-Earth3P-HR_hist-1950_r1i1p2f1_gr_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape
    
# Load Liang index
filename_liang = dir_input + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_1988-2017.npy'
tau_init = np.load(filename_liang,allow_pickle=True)[0]

# Load significance slice 1
filename_sig1 = dir_input + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_sig_fdr_1988-2017_slice_1_alpha005.npy'
sig_tau_fdr_init1 = np.load(filename_sig1,allow_pickle=True)[0]

# Load significance slice 2
filename_sig2 = dir_input + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_sig_fdr_1988-2017_slice_2_alpha005.npy'
sig_tau_fdr_init2 = np.load(filename_sig2,allow_pickle=True)[0]

# Concatenate datasets
sig_tau_fdr_init = np.concatenate((sig_tau_fdr_init1,sig_tau_fdr_init2),axis=0)

# Interpolate from NEMO to IFS grid and filter
if use_filter == True:
    filename = dir_input + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_sig_fdr_1988-2017_interp_alpha005_filter.npy'
else:
    filename = dir_input + 'OHCbudget_Liang_ORAS5_' + str(depth) + 'm_sig_fdr_1988-2017_interp_alpha005.npy'
if save_var == True:
    tau = np.zeros((ny,nx,nvar,nvar))
    sig_tau_fdr = np.zeros((ny,nx,nvar,nvar))
    for i in np.arange(nvar):
        print(i)
        for j in np.arange(nvar):
            tau[:,:,i,j] = griddata((lon_nemo.ravel(),lat_nemo.ravel()),tau_init[:,:,i,j].ravel(),(lon,lat),method='linear')
            sig_tau_fdr[:,:,i,j] = griddata((lon_nemo.ravel(),lat_nemo.ravel()),sig_tau_fdr_init[:,:,i,j].ravel(),(lon,lat),method='nearest')
            if use_filter == True:
                sig_tau_fdr[:,:,i,j] = median_filter(sig_tau_fdr[:,:,i,j],size=3,mode='nearest') # image filtering
    np.save(filename,[tau,sig_tau_fdr])
else:
    tau,sig_tau_fdr = np.load(filename,allow_pickle=True)

# Cartopy projection
proj = ccrs.Robinson()

# Palettes
palette_tau = plt.cm.seismic._resample(20)
min_tau = -30.
max_tau = 30.
colorbar_ticks = [-30,-15,0,15,30]


########
# Maps #
########

# Relative transfer of information
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau OHTc-->OHCt
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau[:,:,1,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax1.contour(lon,lat,sig_tau_fdr[:,:,1,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$\tau_{OHTc \longrightarrow OHCt}$ 1988-2017 - ORAS5',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=colorbar_ticks,extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{OHTc \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# tau Qnet->OHCt
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau[:,:,2,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax2.contour(lon,lat,sig_tau_fdr[:,:,2,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$\tau_{Qnet \longrightarrow OHCt}$ 1988-2017 - ORAS5',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=colorbar_ticks,extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{Qnet \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# Save figure
if save_fig == True:
    if depth == 50:
        fig.savefig(dir_fig + 'fig3.png')
    elif depth == 300:
        fig.savefig(dir_fig + 'fig6.png')