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

linestyles=['-.', '-']
vws=[200., 500., 1000.]
# selection=['NGC3115', 'NGC1172', 'NGC4478']
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors

for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		if not sc.check(d):
			continue
		try:
			saved=np.load(d+'/save.npz')['a']
		except:
			continue
		try:
			rsoi=np.genfromtxt(gal_data+'/rsoi')
			sigma=np.genfromtxt(gal_data+'/sigma')
		except:
			continue

		saved=np.load(d+'/save.npz')['a']
		start=galaxy.prepare_start(saved[-1])
		gal=galaxy.NukerGalaxy(name, gal_dict, init_array=start)
		
		heating=gal.q_grid*(0.5*gal.vel**2+0.5*(vw*1.E5)**2+0.5*sigma[:,1]**2)
		cooling=gal.cooling

		np.savetxt(d+'/hc_ratio', np.transpose([gal.radii, heating/cooling]))


for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_A2052/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		if not sc.check(d):
			continue
		try:
			saved=np.load(d+'/save.npz')['a']
		except:
			continue
		try:
			rsoi=np.genfromtxt(gal_data+'/rsoi')
			sigma=np.genfromtxt(gal_data+'/sigma')
		except:
			continue

		saved=np.load(d+'/save.npz')['a']
		start=galaxy.prepare_start(saved[-1])

		gal=galaxy.NukerGalaxy(name, gal_dict, init_array=start)
		heating=gal.q_grid*(0.5*gal.vel**2+0.5*(vw*1.E5)**2+0.5*sigma[:,1]**2)
		cooling=gal.cooling

		np.savetxt(d+'/hc_ratio', np.transpose([gal.radii, heating/cooling]))


	

