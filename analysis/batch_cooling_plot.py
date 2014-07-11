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

import astropy.table
from astropy.table import Table,Column

from collections import OrderedDict

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value

def check(outdir, tol=40.):
	check=False
	refloat=r'[+-]?\d+\.?\d*[eE]?[+-]?\d*'
	try:
		f=open(outdir+'/check', 'r')
	except:
		return False
	checkf=f.read()
	
	pdiffs=re.findall(re.compile('pdiff='+refloat), checkf)
	try:
		cons1=re.findall(refloat,pdiffs[0])[0]
		cons2=re.findall(refloat,pdiffs[-1])[0]
		if float(cons1)<tol and float(cons2)<tol:
			check=True
	except:
		pass
	
	return check 


gal_dict=galaxy.nuker_params()
fig,ax=plt.subplots(1, sharex=True, figsize=(10, 8))
ax.set_xlabel('Radius [cm]')
ax.set_ylabel(r'$q^{+}/q_{\rm cooling}$')

linestyles=['-.', '-']
vws=[200., 500., 1000.]
# selection=['NGC3115', 'NGC1172', 'NGC4478']
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors

cooling_table=[]

for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)

		try:
			hc_ratio=np.genfromtxt(d+'/hc_ratio')
		except:
			continue


		ax.loglog(hc_ratio[:,0], hc_ratio[:,1], color=cols[j], label=vw)


		
for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_A2052/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)

		try:
			hc_ratio=np.genfromtxt(d+'/hc_ratio')
		except:
			continue

		ax.loglog(hc_ratio[:,0], hc_ratio[:,1], color=cols[j], label=vw)


handles, labels=ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.savefig('batch_cooling_plot.png')
