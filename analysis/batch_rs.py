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


gal_dict=galaxy.nuker_params()
fig,ax=plt.subplots(1, sharex=True, figsize=(10, 8))
ax.set_xlabel('$v_w/\sigma$')
ax.set_ylabel(r'$r_{\rm stag}/r_{\rm soi}$')
linestyles=['-.', '-']
vws=[200., 500., 1000.]
# selection=['NGC3115', 'NGC1172', 'NGC4478']
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors

for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		if not check(d):
			continue
		try:
			saved=np.load(d+'/save.npz')['a']
		except:
			continue


		start=galaxy.prepare_start(saved[-1])
		gal=galaxy.NukerGalaxy(name, gal_dict, init_array=start)

		if gal.params['type']=='Core':
			symbol='x'
		else:
			symbol='s'

		try:
			rsoi=np.genfromtxt(base_d+'/rsoi')
			sigma=np.genfromtxt(base_d+'/sigma')
		except:
			pass
		sigma_interp=interp1d(sigma[:,0], sigma[:,1])

		ax.loglog(vw*1.E5/sigma_interp(rsoi*pc), gal.rs/pc/rsoi, symbol, color=cols[j])

for name in gal_dict.keys():
	for j,vw in enumerate(vws):
		d='/Users/aleksey/Second_Year_Project/hydro/batch_A2052/'+name+'/vw_'+str(vw)
		if not check(d):
			continue
		try:
			saved=np.load(d+'/save.npz')['a']
		except:
			continue


		start=galaxy.prepare_start(saved[-1])
		gal=galaxy.NukerGalaxy(name, gal_dict, init_array=start)


		if gal.params['type']=='Core':
			symbol='x'
		else:
			symbol='s'
		try:
			rsoi=np.genfromtxt(base_d+'/rsoi')
			sigma=np.genfromtxt(base_d+'/sigma')
		except:
			pass
			
		sigma_interp=interp1d(sigma[:,0], sigma[:,1])


		ax.loglog(vw*1.E5/gal.sigma(gal.rinf), gal.rs/pc/gal.rinf, symbol, color=cols[j])


plt.savefig('rs.png')