#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute Liang index OHCt - OHTc - Qnet (-residual)
    EC-Earth3P / HadGEM3-GC31 (interpolation of ocean fields onto IFS grid)
    Ocean heat budget based on Roberts et al. (2017)
    OHCt: ocean heat content tendency (dOHC/dt)
    OHTc: ocean heat transport convergence
    Qnet: net downward surface heat flux, which includes both solar (shortwave) and non-solar heat fluxes; it is positive downwards
    Non-solar heat flux = net longwave radiation + sensible heat flux + latent heat flux
    Residual: OHCt - (OHTc + Qnet)
PROGRAMMER
    D. Docquier
LAST UPDATE
    23/11/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
from scipy.interpolate import griddata

# Import my functions
sys.path.append('/home/cvaf/Air-Sea/')
from function_liang_nvar2 import compute_liang_nvar

# Options
boot_iter = int(sys.argv[1]) # bootstrap realization (to be used in the job file)
model = 'EC-Earth3P-HR' # EC-Earth3P; EC-Earth3P-HR; HadGEM3-GC31-LL; HadGEM3-GC31-MM; HadGEM3-GC31-HM
upscaling = False # True: interpolate from IFS-HR to IFS-LR; False: no upscaling
interpolation_oht = False # True: interpolate OHTc from NEMO onto IFS; False: use existing file
interpolation_ohc = False # True: interpolate OHC from NEMO onto IFS; False: use existing file
save_qnet = False # True: save Qnet; False: use existing file
use_residual = 0 # 0: no residual; 1: use residual instead of Qnet; 2: use residual instead of OHTc
nvar = 3 # number of variables
var_2D = 1 # if nvar=2 and use_residual=0, var_2D=1 is OHTc and var_2D=2 is Qnet
depth = 50 # depth for OHT and OHC integrations (50m or 300m)

# Other options
if model == 'EC-Earth3P-HR' and upscaling == True:
    model_interp = 'EC-Earth3P'
else:
    model_interp = model
if model == 'EC-Earth3P' or model == 'EC-Earth3P-HR':
    member = 'r1i1p2f1'
else:
    member = 'r1i1p1f1'
    
# Time parameters
nyears = int(2017-1988+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months
dt = 1 # time step
mon_to_sec = 30.4375 * 24. * 60. * 60. # conversion /month to /sec

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/' + model + '/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/' + model + '/'

# Load latitude and longitude from IFS grid
if model == 'EC-Earth3P' or model == 'EC-Earth3P-HR':
    grid = 'gr'
elif model == 'HadGEM3-GC31-LL' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
    grid = 'gn'
if model == 'EC-Earth3P-HR' and upscaling == True:
    filename = '/ec/res4/hpcperm/cvaf/' + model_interp + '/' + member + '/hfls/hfls_Amon_' + model_interp + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
else:
    filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['lat'][:]
lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape

# Load latitude and longitude from IFS-HR grid if upscaling
if model == 'EC-Earth3P-HR' and upscaling == True:
    filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_hist-1950_' + member + '_' + grid + '_198801-198812.nc'
    fh = Dataset(filename, mode='r')
    lat_init = fh.variables['lat'][:]
    lon_init = fh.variables['lon'][:]
    lon_hr,lat_hr = np.meshgrid(lon_init,lat_init)
    fh.close()
    
# Load latitude and longitude from NEMO grid
filename = dir_input + 'grid/thkcello_' + model + '.nc'
fh = Dataset(filename, mode='r')
lat_nemo = fh.variables['latitude'][:]
lon_nemo = fh.variables['longitude'][:]
if model == 'HadGEM3-GC31-LL' or model == 'HadGEM3-GC31-MM' or model == 'HadGEM3-GC31-HM':
    lon_nemo[lon_nemo<0.] = lon_nemo[lon_nemo<0.] + 360.
fh.close()

# Interpolate OHC from NEMO onto IFS grid
if model == 'EC-Earth3P-HR' and upscaling == True:
    filename_interp = dir_input + member + '/OHC/OHC_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp_upscaling.npy'
else:
    filename_interp = dir_input + member + '/OHC/OHC_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp.npy'
if interpolation_ohc == True:
    filename = dir_input + member + '/OHC/OHC_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
    ohc = np.load(filename,allow_pickle=True)[0]
    ohc_interp = np.zeros((nm,ny,nx))
    for i in np.arange(nm):
        print(i)
        ohc_interp[i,:,:] = griddata((lon_nemo.ravel(),lat_nemo.ravel()),ohc[i,:,:].ravel(),(lon,lat),method='linear')
    np.save(filename_interp,[ohc_interp])
    del ohc
else:
    ohc_interp = np.load(filename_interp,allow_pickle=True)[0]
ohc_interp[ohc_interp==0.] = np.nan

# Interpolate OHT convergence from NEMO onto IFS grid
if model == 'EC-Earth3P-HR' and upscaling == True:
    filename_interp = dir_input + member + '/OHT/divOHT_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp_upscaling.npy'
else:
    filename_interp = dir_input + member + '/OHT/divOHT_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017_interp.npy'
if interpolation_oht == True:
    filename = dir_input + member + '/OHT/divOHT_' + model + '_' + member + '_' + str(depth) + 'm_1988-2017.npy'
    oht = np.load(filename,allow_pickle=True)[0]
    oht_interp = np.zeros((nm,ny,nx))
    for i in np.arange(nm):
        print(i)
        oht_interp[i,:,:] = griddata((lon_nemo.ravel(),lat_nemo.ravel()),oht[i,:,:].ravel(),(lon,lat),method='linear')
    np.save(filename_interp,[oht_interp])
    del oht
else:
    oht_interp = np.load(filename_interp,allow_pickle=True)[0]
oht_interp[oht_interp==0.] = np.nan
oht_interp = -oht_interp

# File names
if boot_iter < 10:
    if nvar == 3:
        if model == 'EC-Earth3P-HR' and upscaling == True:
            filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017_upscaling.npy'
        else:
            if use_residual == 0:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017.npy'
            elif use_residual == 1:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017_residual.npy'
            elif use_residual == 2:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017_residual2.npy'
    elif nvar == 2:
        if use_residual == 0:
            if var_2D == 1:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017_2D_OHTc.npy'
            else:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017_2D_Qnet.npy'
        elif use_residual == 1:
            filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_0' + str(boot_iter) + '_1988-2017_2D_residual.npy'
else:
    if nvar == 3:
        if model == 'EC-Earth3P-HR' and upscaling == True:
            filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017_upscaling.npy'
        else:
            if use_residual == 0:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017.npy'
            elif use_residual == 1:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017_residual.npy'
            elif use_residual == 2:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017_residual2.npy'
    elif nvar == 2:
        if use_residual == 0:
            if var_2D == 1:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017_2D_OHTc.npy'
            else:
                filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017_2D_Qnet.npy'
        elif use_residual == 1:
            filename_liang = dir_output + 'OHCbudget_Liang_' + model + '_' + member + '_' + str(depth) + 'm_' + str(boot_iter) + '_1988-2017_2D_residual.npy'

# Initialize variables (with zeroes)
Qnet = np.zeros((nm,ny,nx),dtype='float32')
OHCt = np.zeros((nm,ny,nx),dtype='float32')
residual = np.zeros((nm,ny,nx),dtype='float32')
tau = np.zeros((ny,nx,nvar,nvar))
boot_tau = np.zeros((ny,nx,nvar,nvar))

# Save Qnet - loop over years
if save_qnet == True:
    for year in np.arange(nyears):
        print(1988+year)
        
        # Select scenario
        if (1988+year) < 2015:
            scenario = 'hist-1950'
        else:
            scenario = 'highres-future'
    
        # Load LHF
        filename = dir_input + member + '/hfls/hfls_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(1988+year) + '01-' + str(1988+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        lhf = fh.variables['hfls'][:]
        fh.close()
    
        # Load SHF
        filename = dir_input + member + '/hfss/hfss_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(1988+year) + '01-' + str(1988+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        shf = fh.variables['hfss'][:]
        fh.close()
        
        # Load LWFdown
        filename = dir_input + member + '/rlds/rlds_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(1988+year) + '01-' + str(1988+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        lwfdown = fh.variables['rlds'][:]
        fh.close()
        
        # Load LWFup
        filename = dir_input + member + '/rlus/rlus_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(1988+year) + '01-' + str(1988+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        lwfup = fh.variables['rlus'][:]
        fh.close()
        
        # Load SWFdown
        filename = dir_input + member + '/rsds/rsds_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(1988+year) + '01-' + str(1988+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        swfdown = fh.variables['rsds'][:]
        fh.close()
        
         # Load SWFup
        filename = dir_input + member + '/rsus/rsus_Amon_' + model + '_' + scenario + '_' + member + '_' + grid + '_' + str(1988+year) + '01-' + str(1988+year) + '12.nc'
        fh = Dataset(filename, mode='r')
        swfup = fh.variables['rsus'][:]
        fh.close()
        
        # Compute net surface heat flux (Qnet)
        if model == 'EC-Earth3P-HR' and upscaling == True:
            Qnet_init = lwfdown - lwfup + swfdown - swfup - lhf - shf
        else:
            Qnet[year*nmy:year*nmy+nmy,:,:] = lwfdown - lwfup + swfdown - swfup - lhf - shf
        
        # Delete variables
        del lhf,shf,lwfdown,lwfup,swfdown,swfup
        
        # Interpolate Qnet onto IFS LR grid
        if model == 'EC-Earth3P-HR' and upscaling == True:
            for i in np.arange(nmy):
                Qnet[year*nmy+i,:,:] = griddata((lon_hr.ravel(),lat_hr.ravel()),Qnet_init[i,:,:].ravel(),(lon,lat),method='linear')

    # Save Qnet
    if model == 'EC-Earth3P-HR' and upscaling == True:
        filename_interp = dir_input + member + '/Qnet/Qnet_' + model + '_' + member + '_1988-2017_upscaling.npy'
        np.save(filename_interp,[Qnet])
    else:
        filename = dir_input + member + '/Qnet/Qnet_' + model + '_' + member + '_1988-2017.npy'
        np.save(filename,[Qnet])

else:
    if model == 'EC-Earth3P-HR' and upscaling == True:
        filename_interp = dir_input + member + '/Qnet/Qnet_' + model + '_' + member + '_1988-2017_upscaling.npy'
        Qnet = np.load(filename_interp,allow_pickle=True)[0]
    else:
        filename = dir_input + member + '/Qnet/Qnet_' + model + '_' + member + '_1988-2017.npy'
        Qnet = np.load(filename,allow_pickle=True)[0]

# Compute OHC tendency, remove trend and seasonality of OHCt, Qnet, OHTc and residual, and compute Liang index in each grid point
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        if np.count_nonzero(np.isnan(ohc_interp[:,y,x])) >= 1 or np.count_nonzero(np.isnan(oht_interp[:,y,x])) >= 1 or np.count_nonzero(np.isnan(Qnet[:,y,x])) >= 1:
            tau[y,x,:,:] = np.nan
        else:
            OHCt[1:nm-1,y,x] = (ohc_interp[2:nm,y,x] - ohc_interp[0:nm-2,y,x]) / (2.*dt*mon_to_sec)
            OHCt_resid = seasonal_decompose(OHCt[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            OHTc_resid = seasonal_decompose(oht_interp[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            Qnet_resid = seasonal_decompose(Qnet[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            if use_residual == 1 or use_residual == 2:
                residual[:,y,x] = OHCt[:,y,x] - oht_interp[:,y,x] - Qnet[:,y,x]
                residual_resid = seasonal_decompose(residual[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            if nvar == 3:
                if use_residual == 0:
                    xx = np.array((OHCt_resid,OHTc_resid,Qnet_resid))
                elif use_residual == 1:
                    xx = np.array((OHCt_resid,OHTc_resid,residual_resid))
                elif use_residual == 2:
                    xx = np.array((OHCt_resid,Qnet_resid,residual_resid))
            elif nvar == 2:
                if use_residual == 0:
                    if var_2D == 1:
                        xx = np.array((OHCt_resid,OHTc_resid))
                    else:
                        xx = np.array((OHCt_resid,Qnet_resid))
                elif use_residual == 1:
                    xx = np.array((OHCt_resid,residual_resid))
            notused,tau[y,x,:,:],notused,boot_tau[y,x,:,:] = compute_liang_nvar(xx,dt)
    
# Save variables
if boot_iter == 1:
    np.save(filename_liang,[tau,boot_tau])
else:
    np.save(filename_liang,[boot_tau])