#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Map Liang index Qnet / OHTc / Residual --> OHCt
    EC-Earth3P; HadGEM3-GC31
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
    Fig. 1: EC-Earth3-LR, upper 50m
    Fig. 2: EC-Earth3-HR, upper 50m 
    Fig. 4: EC-Earth3-LR, upper 300m
    Fig. 5: EC-Earth3-HR, upper 300m
    Fig. A1: HadGEM3-LL, upper 50m
    Fig. A3: HadGEM3-MM, upper 50m
    Fig. A4: HadGEM3-HM, upper 50m
    Fig. A5: HadGEM3-HH, upper 50m
    Fig. A7: EC-Earth3-HR with control instead of historical
    Fig. A8: EC-Earth3-HR with upscaling, upper 50m
    Fig. A9: EC-Earth3-HR with Residual instead of Qnet
    Fig. A10: EC-Earth3-HR with Residual instead of OHTc
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
from scipy.ndimage import median_filter

# Options
save_fig = True # True: save figure
model = 'EC-Earth3P' # EC-Earth3P; EC-Earth3P-HR; HadGEM3-GC31-LL; HadGEM3-GC31-MM; HadGEM3-GC31-HM; HadGEM3-GC31-HH
load_significance = True # True: load significance of Liang index
use_residual = 0 # 0: no residual; 1: use residual instead of Qnet; 2: use residual instead of OHTc
use_filter = True # True: use median filter for significant contours (not to have isolated significant points)
upscaling = False # True: interpolate from IFS-HR to IFS-LR; False: no upscaling
use_control = False # True: use control run; False: use historical run
depth = 50 # 50m or 300m for the integration of OHC and OHT convergence
nvar = 3 # number of variables (default: 3)

# Other options
if model == 'EC-Earth3P-HR' and upscaling == True:
    model_interp = 'EC-Earth3P'
else:
    model_interp = model
if model == 'EC-Earth3P' or model == 'EC-Earth3P-HR':
    member = 'r1i1p2f1'
else:
    member = 'r1i1p1f1'

# Working directories
if model == 'HadGEM3-GC31-HH':
    dir_input = '/ec/res4/scratch/cvaf/' + model_interp + '/'
else:
    dir_input = '/ec/res4/hpcperm/cvaf/' + model_interp + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/Models/'

# Load latitude and longitude
if model == 'EC-Earth3P' or model == 'EC-Earth3P-HR':
    grid = 'gr'
else:
    grid = 'gn'
filename = dir_input + member + '/hfls/hfls_Amon_' + model_interp + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
    
# Load Liang index
if model == 'EC-Earth3P-HR' and upscaling == True:
    filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_upscaling.npy' # thetao, uo, vo and Qnet upscaled to LR
else:
    if use_residual == 0:
        if use_control == True:
            filename_liang = dir_output + 'control-1950/OHCbudget_Liang_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_1950-2049.npy'
        else:
            filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
    elif use_residual == 1:
        filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_residual.npy'
    elif use_residual == 2:
        filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_residual2.npy'
tau = np.load(filename_liang,allow_pickle=True)[0]

# Load significance of Liang index
if load_significance == True:
    if model == 'EC-Earth3P-HR' and upscaling == True:
        filename_sig = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_upscaling.npy'
    else:
        if use_residual == 0:
            if use_control == True:
                filename_sig = dir_output + 'control-1950/OHCbudget_Liang_' + model + '_control-1950_' + member + '_' + str(depth) + 'm_sig_fdr_1950-2049.npy'
            else:
                filename_sig = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_alpha005.npy'
        elif use_residual == 1:
            filename_sig = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_residual.npy'
        elif use_residual == 2:
            filename_sig = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_sig_fdr_1988-2017_residual2.npy'
    sig_tau_fdr = np.load(filename_sig,allow_pickle=True)[0]

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
if model == 'EC-Earth3P':
    model_name = 'EC-Earth3-LR'
elif model == 'EC-Earth3P-HR':
    model_name = 'EC-Earth3-HR'
elif model == 'HadGEM3-GC31-LL':
    model_name = 'HadGEM3-LL'
elif model == 'HadGEM3-GC31-MM':
    model_name = 'HadGEM3-MM'
elif model == 'HadGEM3-GC31-HM':
    model_name = 'HadGEM3-HM'
elif model == 'HadGEM3-GC31-HH':
    model_name = 'HadGEM3-HH'

# Relative transfer of information
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau OHTc-->OHCt or Qnet->OHCt
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
if use_residual == 0 or use_residual == 1:
    if use_control == True:
        ax1.set_title(r'$\tau_{OHTc \longrightarrow OHCt}$ 1950-2049 - ' + model_name,fontsize=24)
    else:
        ax1.set_title(r'$\tau_{OHTc \longrightarrow OHCt}$ 1988-2017 - ' + model_name,fontsize=24)
else:
    ax1.set_title(r'$\tau_{Qnet \longrightarrow OHCt}$ 1988-2017 - ' + model_name,fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
if use_residual == 0 or use_residual == 1:
    cbar.set_label(r'$\tau_{OHTc \longrightarrow OHCt}$ ($\%$)',fontsize=18)
else:
    cbar.set_label(r'$\tau_{Qnet \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# tau Qnet->OHCt or Residual->OHCt
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
if use_residual == 0:
    if use_control == True:
        ax2.set_title(r'$\tau_{Qnet \longrightarrow OHCt}$ 1950-2049 - ' + model_name,fontsize=24)
    else:
        ax2.set_title(r'$\tau_{Qnet \longrightarrow OHCt}$ 1988-2017 - ' + model_name,fontsize=24)
else:
    ax2.set_title(r'$\tau_{Residual \longrightarrow OHCt}$ 1988-2017 - ' + model_name,fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
if use_residual == 0:
    cbar.set_label(r'$\tau_{Qnet \longrightarrow OHCt}$ ($\%$)',fontsize=18)
else:
    cbar.set_label(r'$\tau_{Residual \longrightarrow OHCt}$ ($\%$)',fontsize=18)

# Save figure
if save_fig == True:
    if model == 'EC-Earth3P-HR' and upscaling == True:
        if depth == 50:
            fig.savefig(dir_fig + 'fig_a8.png')
    else:
        if use_residual == 0:
            if model == 'EC-Earth3P' and depth == 50:
                fig.savefig(dir_fig + 'fig1.png')
            elif model == 'EC-Earth3P-HR' and depth == 50:
                if use_control == True:
                    fig.savefig(dir_fig + 'fig_a7.png')
                else:
                    fig.savefig(dir_fig + 'fig2.png')
            elif model == 'EC-Earth3P' and depth == 300:
                fig.savefig(dir_fig + 'fig4.png')
            elif model == 'EC-Earth3P-HR' and depth == 300:
                fig.savefig(dir_fig + 'fig5.png')
            elif model == 'HadGEM3-GC31-LL' and depth == 50:
                fig.savefig(dir_fig + 'fig_a1.png')
            elif model == 'HadGEM3-GC31-MM' and depth == 50:
                fig.savefig(dir_fig + 'fig_a3.png')
            elif model == 'HadGEM3-GC31-HM' and depth == 50:
                fig.savefig(dir_fig + 'fig_a4.png')
            elif model == 'HadGEM3-GC31-HH' and depth == 50:
                fig.savefig(dir_fig + 'fig_a5.png')
        elif use_residual == 1:
            if model == 'EC-Earth3P-HR' and depth == 50:
                fig.savefig(dir_fig + 'fig_a9.png')
        elif use_residual == 2:
            if model == 'EC-Earth3P-HR' and depth == 50:
                fig.savefig(dir_fig + 'fig_a10.png')