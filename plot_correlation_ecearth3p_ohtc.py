#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Map correlations between OHCt and OHTc
    Fig. 8: EC-Earth3P and EC-Earth3P-HR (interpolation of ocean fields onto atmospheric grid)
PROGRAMMER
    D. Docquier
LAST UPDATE
    14/11/2022
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
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import median_filter

# Options
save_fig = True
model_lr = 'EC-Earth3P'
model_hr = 'EC-Earth3P-HR'
member = 'r1i1p2f1'
depth = 50 # 50m or 300m for the integration of ocean heat flux
alpha_fdr = 0.05 # alpha of FDR
use_filter = True # True: use median filter for significant contours (not to have isolated significant points)

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/Models/'

# Load latitude and longitude from IFS-LR
grid = 'gr'
filename = dir_input + model_lr + '/' + member + '/hfls/hfls_Amon_' + model_lr + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon_lr,lat_lr = np.meshgrid(lon_init,lat_init)
fh.close()
ny_lr,nx_lr = lon_lr.shape

# Load latitude and longitude from IFS-HR
filename = dir_input + model_hr + '/' + member + '/hfls/hfls_Amon_' + model_hr + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon_hr,lat_hr = np.meshgrid(lon_init,lat_init)
fh.close()
ny_hr,nx_hr = lon_hr.shape
    
# Load correlations LR
filename = dir_output + model_lr + '/OHCbugdet_Corr_' + model_lr + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
corrcoef1_lr,corrcoef2_lr,pval_corrcoef1_lr,pval_corrcoef2_lr = np.load(filename,allow_pickle=True)

# Load correlations HR
filename = dir_output + model_hr + '/OHCbugdet_Corr_' + model_hr + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
corrcoef1_hr,corrcoef2_hr,pval_corrcoef1_hr,pval_corrcoef2_hr = np.load(filename,allow_pickle=True)

# Compute FDR OHCt-OHTc LR
pval_corrcoef2_1d = np.ravel(pval_corrcoef2_lr)
sig_corrcoef2_1d = multipletests(pval_corrcoef2_1d,alpha=alpha_fdr,method='fdr_bh')[0]
sig_corrcoef2_1d[sig_corrcoef2_1d==True] = 1
sig_corrcoef2_1d[sig_corrcoef2_1d==False] = 0
sig_corrcoef2_lr = np.reshape(sig_corrcoef2_1d,(ny_lr,nx_lr))

# Compute FDR OHCt-OHTc HR
pval_corrcoef2_1d = np.ravel(pval_corrcoef2_hr)
sig_corrcoef2_1d = multipletests(pval_corrcoef2_1d,alpha=alpha_fdr,method='fdr_bh')[0]
sig_corrcoef2_1d[sig_corrcoef2_1d==True] = 1
sig_corrcoef2_1d[sig_corrcoef2_1d==False] = 0
sig_corrcoef2_hr = np.reshape(sig_corrcoef2_1d,(ny_hr,nx_hr))

# Image filtering
if use_filter == True:
    sig_corrcoef2_lr = median_filter(sig_corrcoef2_lr,size=3,mode='nearest')
    sig_corrcoef2_hr = median_filter(sig_corrcoef2_hr,size=3,mode='nearest')

# Cartopy projection
proj = ccrs.Robinson()

# Palettes
palette_diff = plt.cm.seismic._resample(20)
min_cor = -1.
max_cor = 1.


########
# Maps #
########

# Correlation coefficients
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# OHCt-OHTc LR
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon_lr,lat_lr,corrcoef2_lr,cmap=palette_diff,vmin=min_cor,vmax=max_cor,transform=ccrs.PlateCarree())
ax1.contour(lon_lr,lat_lr,sig_corrcoef2_lr,range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title('$R_{OHCt-OHTc}$ 1988-2017 - EC-Earth3-LR',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-1,-0.5,0,0.5,1],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Correlation coefficient',fontsize=18)

# OHCt-OHTc HR
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon_hr,lat_hr,corrcoef2_hr,cmap=palette_diff,vmin=min_cor,vmax=max_cor,transform=ccrs.PlateCarree())
ax2.contour(lon_hr,lat_hr,sig_corrcoef2_hr,range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title('$R_{OHCt-OHTc}$ 1988-2017 - EC-Earth3-HR',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-1,-0.5,0,0.5,1],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Correlation coefficient',fontsize=18)
    
# Save figure
if save_fig == True:
    if use_filter == True:
#        fig.savefig(dir_fig + 'Correlation_EC-Earth3P_' + str(depth) + 'm_medianfilter.png')
        fig.savefig(dir_fig + 'fig8.png')
    else:
        fig.savefig(dir_fig + 'Correlation_EC-Earth3P_' + str(depth) + 'm.png')