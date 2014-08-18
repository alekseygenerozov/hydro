#!/usr/bin/env python

import astropy.constants as const
import re
import galaxy

import shlex
import numpy as np
 
import matplotlib.pylab as plt
import dill

from scipy.interpolate import interp1d
import brewer2mpl

import sol_check as sc

from scipy.misc import derivative
import time

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

eddr_cusp=[[],[],[]]
mass_cusp=[[],[],[]]

eddr_core=[[],[],[]]
mass_core=[[],[],[]]

mdot=[[],[],[]]
mdot_approx=[[],[],[]]
mdot_bondi=[[],[],[]]
gammas=[]
names=[]

for idx,name in enumerate(gal_dict.keys()):
	names.append(name)
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_collected/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name

	# if not ((sc.check(base_d+'/vw_200.0') and sc.check(base_d+'/vw_500.0') and sc.check(base_d+'/vw_1000.0'))):
	# 	continue
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		try:
			gal=dill.load(open(d+'/grid.p', 'rb'))
		except:
			print 'could not open pickle'
			continue
		if not gal.check_partial:
			continue
		if not gal.stag_unique:
			continue

		if j==0:
			gammas.append(gal.params['gamma'])

		eddr=gal.eddr
		mass=gal.params['M']/galaxy.M_sun
		if gal.params['gamma']>0.2:
			eddr_cusp[j].append(eddr)
			mass_cusp[j].append(mass)
			marker='s'
			#ax2.loglog(mass[j], eddr[j], 's', color=cols[j], markersize=10)
		else:
			eddr_core[j].append(gal.eddr)
			mass_core[j].append(gal.params['M']/galaxy.M_sun)
			marker='<'
			# if j==1:
			# 	ax2.text(mass_core[j][-1], eddr_core[j][-1], gal.name, fontsize=12)
		ax2.loglog(mass, eddr, marker, color=cols[j], markersize=10)

		# mdot[j].append(gal.mdot)
		# mdot_approx[j].append(gal.mdot_approx)
		# try:
		# 	mdot_bondi[j].append(gal.mdot_bondi)
		# except:
		# 	mdot_bondi[j].append(np.nan)

		# if j==2:
		# 	print gal.name,gal.params['gamma'],gal.params['type'],mdot[2][-1]/mdot[1][-1]

mass_fit=[1.E6, 1.E9]
print eddr






fig1.savefig('mdot_hist.pdf')
fig2.savefig('mdot_mass.eps')
fig3.savefig('mdot_comp.pdf')

