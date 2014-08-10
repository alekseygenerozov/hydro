#!/usr/bin/env python

import astropy.constants as const
import re
import galaxy

import shlex
import numpy as np
 
import matplotlib as mpl
import matplotlib.pylab as plt
import dill

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
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_collected/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		if not sc.check(d):
			print d, 'did not pass cons_check'
			continue
		try:
			gal=dill.load(open(d+'/grid.p','rb'))
		except:
			continue
		print name, vw

		heating=gal.q_grid*(0.5*gal.vel**2+0.5*(vw*1.E5)**2+0.5*gal.sigma_grid**2)
		cooling=gal.cooling

		ax.loglog(gal.radii, heating/cooling, color=cols[j], label=vw)

fig.savefig('cooling.pdf')

	

