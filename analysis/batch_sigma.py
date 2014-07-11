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


gal_dict=galaxy.nuker_params(skip=True)
fig,ax=plt.subplots(1, sharex=True, figsize=(10, 8))
ax.set_xlabel('$v_w/\sigma$')
ax.set_ylabel(r'$r_{\rm stag}/r_{\rm soi}$')
linestyles=['-.', '-']
vws=[200., 500., 1000.]
# selection=['NGC3115', 'NGC1172', 'NGC4478']
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors

for name in gal_dict.keys():
	d='/Users/aleksey/Second_Year_Project/hydro/batch/'+name
	try:
		saved=np.load(d+'/vw_1000.0/save.npz')['a']
		start=galaxy.prepare_start(saved[0])
		gal=galaxy.NukerGalaxy(name, gal_dict,init_array=start)
		np.savetxt(d+'/sigma', np.transpose([gal.radii,gal.sigma_grid]))
	except:
		continue






