#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute  correlation coefficients OHCt-OHTc-Qnet
    EC-Earth3P / HadGEM3-GC31 (interpolation of ocean fields onto atmospheric grid)
    Ocean heat budget based on Roberts et al. (2017)
    OHCt: ocean heat content tendency (dOHC/dt)
    OHTc: ocean heat transport convergence
    Qnet: net downward surface heat flux, which includes both solar (shortwave) and non-solar heat fluxes; it is positive downwards
    Non-solar heat flux = net longwave radiation + sensible heat flux + latent heat flux
PROGRAMMER
    D. Docquier
LAST UPDATE
    03/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose

# Options
model = 'EC-Earth3P' # EC-Earth3P; EC-Earth3P-HR; HadGEM3-GC31-LL; HadGEM3-GC31-MM; HadGEM3-GC31-HM
member = 'r1i1p2f1' # r1i1p2f1 for EC-Earth3P; r1i1p1f1 for HadGEM3-GC31
nyears = int(2017-1988+1)
nmy = 12
nm = int(nyears * nmy)
nvar = 3
dt = 1
depth = 50 # OHT and OHC integrations (50m or 300m)
mon_to_sec = 30.4375 * 24. * 60. * 60. # conversion /month to /sec

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'

# Load latitude and longitude from atmospheric grid
if model == 'EC-Earth3P' or model == 'EC-Earth3P-HR':
    grid = 'gr'
else:
    grid = 'gn'
filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape

# Load latitude and longitude from NEMO grid
filename = dir_input + 'grid/thkcello_' + model + '.nc'
fh = Dataset(filename, mode='r')
lat_nemo = fh.variables['latitude'][:]
lon_nemo = fh.variables['longitude'][:]
if model == 'HadGEM3-GC31-LL' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
    lon_nemo[lon_nemo<0.] = lon_nemo[lon_nemo<0.] + 360.
fh.close()

# Load OHT convergence interpolated onto atmospheric grid
filename_interp = dir_input + member + '/OHT/divOHT_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp.npy'
oht_interp = np.load(filename_interp,allow_pickle=True)[0]
oht_interp[oht_interp==0.] = np.nan
oht_interp = -oht_interp

# Load OHC interpolated onto atmospheric grid
filename_interp = dir_input + member + '/OHC/OHC_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp.npy'
ohc_interp = np.load(filename_interp,allow_pickle=True)[0]
ohc_interp[ohc_interp==0.] = np.nan

# Load Qnet
filename = dir_input + member + '/Qnet/Qnet_' + model + '_' + member + '_1988-2017.npy'
Qnet = np.load(filename,allow_pickle=True)[0]

# Initialize variables (with zeroes)
OHCt = np.zeros((nm,ny,nx),dtype='float32')
corrcoef1 = np.zeros((ny,nx))
corrcoef2 = np.zeros((ny,nx))
pval_corrcoef1 = np.zeros((ny,nx))
pval_corrcoef2 = np.zeros((ny,nx))

# Compute OHC tendency, remove trend and seasonality of OHCt, Qnet, OHTc and residual, and compute correlation coefficient in each grid point
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        if np.count_nonzero(np.isnan(ohc_interp[:,y,x])) >= 1 or np.count_nonzero(np.isnan(oht_interp[:,y,x])) >= 1 or np.count_nonzero(np.isnan(Qnet[:,y,x])) >= 1:
            corrcoef1[y,x] = np.nan
            corrcoef2[y,x] = np.nan
            pval_corrcoef1[y,x] = np.nan
            pval_corrcoef2[y,x] = np.nan
        else:
            OHCt[1:nm-1,y,x] = (ohc_interp[2:nm,y,x] - ohc_interp[0:nm-2,y,x]) / (2.*dt*mon_to_sec)
            OHCt_resid = seasonal_decompose(OHCt[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            OHTc_resid = seasonal_decompose(oht_interp[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            Qnet_resid = seasonal_decompose(Qnet[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            corrcoef1[y,x],pval_corrcoef1[y,x] = pearsonr(OHCt_resid,Qnet_resid)
            corrcoef2[y,x],pval_corrcoef2[y,x] = pearsonr(OHCt_resid,OHTc_resid)
    
# Save variables
filename_liang = dir_output + 'OHCbugdet_Corr_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
np.save(filename_liang,[corrcoef1,corrcoef2,pval_corrcoef1,pval_corrcoef2])