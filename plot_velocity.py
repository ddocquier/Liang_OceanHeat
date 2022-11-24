#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Map ocean surface velocity January 1988
    Fig. 7: EC-Earth3-LR and EC-Earth3-HR
PROGRAMMER
    D. Docquier
LAST UPDATE
    14/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
import matplotlib.ticker as mticker

# Options
save_fig = True

# Working directories
dir_input = '/home/dadocq/Documents/Papers/My_Papers/Air-Sea/output/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/Air-Sea/Models/OS/LaTeX/'

# Load velocity EC-Earth3-LR
filename = dir_input + 'uo_Omon_EC-Earth3P_hist-1950_r1i1p2f1_gn_sfc_198801.nc'
fh = Dataset(filename, mode='r')
uo_lr = fh.variables['uo'][0,0,:,:]
lon = fh.variables['longitude'][:]
lat = fh.variables['latitude'][:]
fh.close()

# Load velocity EC-Earth3-HR
filename = dir_input + 'uo_Omon_EC-Earth3P-HR_hist-1950_r1i1p2f1_gn_198801_sfc_upscaling.nc'
fh = Dataset(filename, mode='r')
uo_hr = fh.variables['uo'][0,0,:,:]
fh.close()

# Cartopy projection
proj = ccrs.Robinson()

# Palettes
palette_uo = plt.cm.seismic._resample(20)
min_uo = -0.5
max_uo = 0.5


########
# Maps #
########

# Ocean surface velocity - January 1988
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau OHCt-->Qnet
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,uo_lr,cmap=palette_uo,vmin=min_uo,vmax=max_uo,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title('Meridional velocity January 1988 - EC-Earth3-LR',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[min_uo,min_uo/2,0,max_uo/2,max_uo],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Meridional velocity (m s$^{-1}$)',fontsize=18)

# tau Qnet->OHCt
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,uo_hr,cmap=palette_uo,vmin=min_uo,vmax=max_uo,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title('Meridional velocity January 1988 - EC-Earth3-HR',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[min_uo,min_uo/2,0,max_uo/2,max_uo],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Meridional velocity (m s$^{-1}$)',fontsize=18)

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig7.png')
