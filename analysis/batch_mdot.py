#!/usr/bin/env python

import astropy.constants as const
import re
import galaxy

import shlex
import numpy as np
 
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

fig1,ax1=plt.subplots(2, sharex=True, figsize=(5,8))
fig2,ax2=plt.subplots()
ax1[0].set_xscale('log')
ax1[1].set_xscale('log')
ax1[1].set_xlabel(r'$\dot{M}/\dot{M_{\rm Edd}}$')
ax1[0].set_ylabel('Number')
ax1[1].set_ylabel('Number')

ax1[1].set_ylim([0,10])
ax1[1].set_ylim([0,10])

ax2.set_xlabel(r'$M_{\bullet}/M_{\odot}$')
ax2.set_ylabel(r'$\dot{M}/\dot{M_{\rm Edd}}$')
ax2.set_xlim([1.E6, 3.E9])

gal_dict=galaxy.nuker_params()
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors
vws=[200., 500., 1000.]

eddr=[[],[],[]]
mass=[[],[],[]]

eddr_core=[[],[],[]]
mass_core=[[],[],[]]
for idx,name in enumerate(gal_dict.keys()):
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	if not (sc.check(base_d+'/vw_200.0') and sc.check(base_d+'/vw_500.0') and sc.check(base_d+'/vw_1000.0')):
		continue
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
		if gal.params['type']=='Cusp':
			eddr[j].append(gal.eddr)
			mass[j].append(gal.M_bh/galaxy.M_sun)
		else:
			eddr_core[j].append(gal.eddr)
			mass_core[j].append(gal.M_bh/galaxy.M_sun)


for idx,name in enumerate(gal_dict.keys()):
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_A2052_unique/'+name
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
		if gal.params['type']=='Cusp':
			continue
			# eddr[j].append(gal.eddr)
			# mass[j].append(gal.M_bh/galaxy.M_sun)
		else:
			eddr_core[j].append(gal.eddr)
			mass_core[j].append(gal.M_bh/galaxy.M_sun)


mass_fit=[1.E6, 1.E9]

for j,vw in enumerate(vws):
	pow_cusp, coeff_cusp=np.polyfit(np.log(mass[j]),np.log(eddr[j]),1)
	print pow_cusp

	ax1[0].hist(eddr[j], color=cols[j], bins=np.logspace(-9, -1, 16), alpha=0.5)
	ax2.loglog(mass_fit, [np.exp(coeff_cusp)*m**pow_cusp for m in mass_fit], color=cols[j])
	ax2.loglog(mass[j], eddr[j], 's', color=cols[j], markersize=10)

ax1[1].hist(eddr_core[2], color=cols[2], bins=np.logspace(-9, -1, 16), histtype='step', linestyle='dashed')
ax2.loglog(mass_core[2], eddr_core[2], '<', color=cols[2], markersize=10)

fig1.savefig('mdot_hist.pdf')
fig2.savefig('mdot_mass.eps')

