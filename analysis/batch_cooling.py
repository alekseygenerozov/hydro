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
from astropy.table import Table
from latex_exp import latex_exp

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
ax.set_xlabel('Radius [cm]')
ax.set_ylabel('H/C')

linestyles=['-.', '-']
vws=[200., 500., 1000.]

eta_max=[[],[],[]]

# selection=['NGC3115', 'NGC1172', 'NGC4478']
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors
i=0
names=gal_dict.keys()
for name in names:
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_collected/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	for j,vw in enumerate(vws):

		d=base_d+'/vw_'+str(vw)
		if not sc.check(d):
			print d, 'did not pass cons_check'
			eta_max[j].append('')
			continue

		gal=dill.load(open(d+'/grid.p','rb'))
		print name, vw

		heating=gal.q_grid*(0.5*gal.vel**2+0.5*(vw*1.E5)**2+0.5*gal.sigma_grid**2)
		cooling=gal.cooling

		eta_max[j].append('%s' % float('%.1g' % min(heating/cooling)))

		ax.loglog(gal.radii, heating/cooling, color=cols[j], label=str(vw))
		i+=1

eta_str=r'$\eta_{\rm max}$'
vw_str=r'$v_{\rm w,0}=$'
eta_str=eta_str+' '+vw_str

eta_tab=Table([names, eta_max[0], eta_max[1], eta_max[2]], names=['Galaxy', eta_str+'200 km/s', eta_str+'500 km/s', eta_str+'1000 km/s'])
for i in range(1,4):
	eta_tab[i].format='%3.2e'
print eta_tab
eta_tab.write('eta_tab.tex', format='latex', latexdict={'header_start':'\hline','header_end':'\hline','data_end':'\hline','caption':r'\label{tab:eta} Table of the maximum\
 $\eta$ for which each of our galaxies would have $H/C$ (Heating Rate/Cooling Rate)$>1$'})

handles, labels = ax.get_legend_handles_labels()
labels=np.array(labels)
handles=np.array(handles)

handles_new=[]
for j in range(3):
	handles_new.append(handles[np.where(labels==str(vws[j]))][0])
vw_str=r'$v_{\rm w,0}=$'
ax.legend(handles_new, [vw_str+'200.0',vw_str+'500.0',vw_str+'1000.0'])
fig.savefig('cooling.eps')

	

