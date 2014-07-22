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
fig3,ax3=plt.subplots()

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

mdot=[[],[],[]]
mdot_approx=[[],[],[]]
mdot_bondi=[[],[],[]]

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

		gal=galaxy.NukerGalaxy.from_dir(name, d)
		if gal.params['type']=='Cusp':
			eddr[j].append(gal.eddr)
			mass[j].append(gal.params['M']/galaxy.M_sun)
		else:
			eddr_core[j].append(gal.eddr)
			mass_core[j].append(gal.params['M']/galaxy.M_sun)

		mdot[j].append(gal.mdot)
		mdot_approx[j].append(gal.mdot_approx)
		try:
			mdot_bondi[j].append(gal.mdot_bondi)
		except:
			mdot_bondi[j].append(np.nan)
		# 	break

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

		gal=galaxy.NukerGalaxy.from_dir(name, d)
		if gal.params['type']=='Cusp':
			eddr[j].append(gal.eddr)
			mass[j].append(gal.params['M']/galaxy.M_sun)
		else:
			eddr_core[j].append(gal.eddr)
			mass_core[j].append(gal.params['M']/galaxy.M_sun)

		# mdot[j].append(gal.mdot)
		# mdot_approx[j].append(gal.mdot_approx)
		# try:
		# 	mdot_bondi[j].append(gal.mdot_bondi)
		# except:
		# 	mdot_bondi[j].append(np.nan)


mdot=np.array(mdot)
mdot_approx=np.array(mdot_approx)
mdot_bondi=np.array(mdot_bondi)

for i in range(len(mdot)):
	print len(mdot[i])

mass_fit=[1.E6, 1.E9]

for j,vw in enumerate(vws):
	pow_cusp, coeff_cusp=np.polyfit(np.log(mass[j]),np.log(eddr[j]),1)
	print np.abs((mdot[j]-mdot_bondi[j])/mdot[j])

	ax1[0].hist(eddr[j], color=cols[j], bins=np.logspace(-9, -1, 16), alpha=0.5)
	ax2.loglog(mass_fit, [np.exp(coeff_cusp)*m**pow_cusp for m in mass_fit], color=cols[j])
	ax2.loglog(mass[j], eddr[j], 's', color=cols[j], markersize=10)


	ax3.plot(range(0,len(mdot[j])), [np.abs((mdot[j,i]-mdot_bondi[j,i])/mdot[j,i]) for i in range(0, len(mdot[j]))],'s',color=cols[j])
	#ax3[1].plot(range(0,len(mdot[j])), np.abs((mdot[j]-mdot_approx[j])/mdot[j]),'s',color=cols[j])
ax1[1].hist(eddr_core[2], color=cols[2], bins=np.logspace(-9, -1, 16), histtype='step', linestyle='dashed')
ax2.loglog(mass_core[2], eddr_core[2], '<', color=cols[2], markersize=10)

fig1.savefig('mdot_hist.pdf')
fig2.savefig('mdot_mass.eps')
fig3.savefig('mdot_comp.pdf')

