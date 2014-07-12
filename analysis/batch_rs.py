#!/usr/bin/env python

import astropy.constants as const
import re
import galaxy

import shlex
import numpy as np
 


import matplotlib as mpl
import matplotlib.pylab as plt
import pickle

from scipy.interpolate import interp1d
import brewer2mpl

import sol_check as sc

from scipy.misc import derivative

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value



gal_dict=galaxy.nuker_params()
fig,ax=plt.subplots(1, sharex=True, figsize=(10, 8))
ax.set_xlabel('$v_w/\sigma$')
ax.set_ylabel(r'$r_{\rm stag}/r_{\rm soi}$')
linestyles=['-.', '-']
vws=[200., 500., 1000.]
# selection=['NGC3115', 'NGC1172', 'NGC4478']
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors

def glaw(eta):
	return 2./(eta**2)


for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		try:
			saved=np.load(d+'/save.npz')['a']
		except:
			continue
		if not sc.check(d):
			continue

		start=galaxy.prepare_start(saved[-1])
		gal=galaxy.NukerGalaxy(name, gal_dict, init_array=start)
		if gal.params['type']=='Core':
			symbol='<'
		else:
			symbol='s'

		try:
			rsoi=np.genfromtxt(gal_data+'/rsoi')
			sigma=np.genfromtxt(gal_data+'/sigma')
		except:
			continue
		sigma_interp=interp1d(sigma[:,0], sigma[:,1])

		rho_interp=interp1d(np.log(gal.radii), np.log(gal.rho))
		dens_slope=derivative(rho_interp, np.log(gal.rs), dx=gal.delta_log[0])
	
		irs=len(gal.radii[gal.radii<gal.rs])-1
		dens_slope2=gal.radii[irs]*gal.get_spatial_deriv(irs, 'log_rho')

		irs=len(gal.radii[gal.radii<gal.rs])
		dens_slope3=gal.radii[irs]*gal.get_spatial_deriv(irs, 'log_rho')

		dens_slope4=derivative(rho_interp, np.log(gal.radii[irs]), dx=gal.delta_log[0])

		print dens_slope, dens_slope2, dens_slope3
		print dens_slope3, dens_slope4

		ax.loglog(vw*1.E5/sigma_interp(rsoi*pc), gal.rs/pc/rsoi, symbol, color=cols[j], markersize=10)

ax.loglog([10., 1.], [glaw(10.), glaw(1.)])
for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_A2052/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		try:
			saved=np.load(d+'/save.npz')['a']
		except:
			continue
		if not sc.check(d):
			continue

		start=galaxy.prepare_start(saved[-1])
		gal=galaxy.NukerGalaxy(name, gal_dict, init_array=start)
		if gal.params['type']=='Core':
			symbol='<'
		else:
			symbol='s'

		try:
			rsoi=np.genfromtxt(gal_data+'/rsoi')
			sigma=np.genfromtxt(gal_data+'/sigma')
		except:
			continue
		sigma_interp=interp1d(sigma[:,0], sigma[:,1])

		ax.loglog(vw*1.E5/sigma_interp(rsoi*pc), gal.rs/pc/rsoi, symbol, color=cols[j], markersize=10)


plt.savefig('rs.png')