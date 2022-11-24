#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Map Liang index OHCt / Qnet / residual / OHTc - 2D
    EC-Earth3P-HR
    Fig. 9
PROGRAMMER
    D. Docquier
LAST UPDATE
    21/11/2022
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
from scipy.ndimage import median_filter

# Options
save_fig = True
model = 'EC-Earth3P-HR'
load_significance = True # True: load significance of Liang index
member = 'r1i1p2f1' # r1i1p2f1 for EC-Earth3P; r1i1p1f1 for HadGEM3-GC31
depth = 50 # 50m or 300m for the integration of OHC and OHT convergence
nvar = 2 # number of variables
use_filter = True # True: use median filter for significant contours (not to have isolated significant points)

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/Models/'

# Load latitude and longitude
grid = 'gr'
filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
    
# Load Liang index - OHTc
filename_liang1 = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_2D_OHTc.npy'
tau_ohtc = np.load(filename_liang1,allow_pickle=True)[0]

# Load Liang index - Qnet
filename_liang2 = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_2D_Qnet.npy'
tau_qnet = np.load(filename_liang2,allow_pickle=True)[0]

# Load Liang index - residual
filename_liang3 = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_2D_residual.npy'
tau_residual = np.load(filename_liang3,allow_pickle=True)[0]

# Load significance of Liang index
if load_significance == True:
    filename_sig1 = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_2D_OHTc.npy'
    filename_sig2 = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_2D_Qnet.npy'
    filename_sig3= dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_2D_residual.npy'
    sig_tau_fdr_ohtc = np.load(filename_sig1,allow_pickle=True)[0]
    sig_tau_fdr_qnet = np.load(filename_sig2,allow_pickle=True)[0]
    sig_tau_fdr_residual = np.load(filename_sig3,allow_pickle=True)[0]

    # Image filtering
    if use_filter == True:
        for i in np.arange(nvar):
            print(i)
            for j in np.arange(nvar):
                sig_tau_fdr_ohtc[:,:,i,j] = median_filter(sig_tau_fdr_ohtc[:,:,i,j],size=3,mode='nearest')
                sig_tau_fdr_qnet[:,:,i,j] = median_filter(sig_tau_fdr_qnet[:,:,i,j],size=3,mode='nearest')
                sig_tau_fdr_residual[:,:,i,j] = median_filter(sig_tau_fdr_residual[:,:,i,j],size=3,mode='nearest')

# Cartopy projection
proj = ccrs.Robinson()

# Palettes
palette_tau = plt.cm.seismic._resample(20)
min_tau = -30.
max_tau = 30.


########
# Maps #
########

# Relative transfer of information
fig = plt.figure(figsize=(12,18))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau OHTc-->OHCt
ax1 = fig.add_subplot(3,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau_ohtc[:,:,1,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
if load_significance == True:
    ax1.contour(lon,lat,sig_tau_fdr_ohtc[:,:,1,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$\tau_{OHTc \longrightarrow OHCt}$ 1988-2017 - EC-Earth3-HR',fontsize=24)
cb_ax = fig.add_axes([0.87,0.7,0.02,0.2])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{OHTc \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# tau Qnet->OHCt or Residual->OHCt
ax2 = fig.add_subplot(3,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau_qnet[:,:,1,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
if load_significance == True:
    ax2.contour(lon,lat,sig_tau_fdr_qnet[:,:,1,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$\tau_{Qnet \longrightarrow OHCt}$ 1988-2017 - EC-Earth3-HR',fontsize=24)
cb_ax = fig.add_axes([0.87,0.4,0.02,0.2])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{Qnet \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# tau Residual->OHCt
ax3 = fig.add_subplot(3,1,3,projection=proj)
cs3 = ax3.pcolormesh(lon,lat,tau_residual[:,:,1,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
if load_significance == True:
    ax3.contour(lon,lat,sig_tau_fdr_residual[:,:,1,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax3.coastlines()
ax3.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax3.set_title(r'$\tau_{Residual \longrightarrow OHCt}$ 1988-2017 - EC-Earth3-HR',fontsize=24)
cb_ax = fig.add_axes([0.87,0.05,0.02,0.2])
cbar = fig.colorbar(cs3,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{Residual \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig9.png')