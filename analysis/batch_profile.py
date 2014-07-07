#!/usr/bin/env python

import astropy.constants as const
import re
import galaxy

import shlex
import numpy as np
import nuker 
import sol_check as sc

import matplotlib as mpl
import matplotlib.pylab as plt
import pickle

from scipy.interpolate import interp1d

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


dirs=galaxy.bash_command('echo NGC*')
galaxies=shlex.split(dirs)
gnames=np.copy(galaxies)
gal_dict=galaxy.nuker_params()
fig,ax=plt.subplots(3, sharex=True, figsize=(10, 24))


ax[0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
ax[1].set_ylabel(r'T [K]')
ax[2].set_ylabel('Cumulative x-ray luminosity [ergs/s]')
ax[2].set_xlabel('r [cm]')
# ax.set_ylabel(r'$r_{\rm stag}/r_{\rm soi}$')


linestyles=['-.', '-']
vws=[200., 1000.]
selection=['NGC3115', 'NGC1172', 'NGC4478']

cols=mpl.rcParams['axes.color_cycle']
gal_dict=galaxy.nuker_params()
for i,name in enumerate(selection):
	for j,vw in enumerate(vws):
		d=name+'/vw_'+str(vw)
		if not check(d):
			continue
		try:
			saved=np.load(d+'/save.npz')['a']
		except:
			continue
		if j!=1 and name!='NGC4478':
			continue

		start=galaxy.prepare_start(saved[-1])
		gal=galaxy.NukerGalaxy(name, gal_dict, init_array=start)
		#gal.set_param('vw_extra', vw)

		rho_interp=interp1d(gal.radii,gal.rho)
		temp_interp=interp1d(gal.radii,gal.temp)
		x_ray_interp=interp1d(gal.radii, gal.x_ray_lum)

		rho_rb=rho_interp(gal.rb)
		rho_rs=rho_interp(gal.rs)

		temp_rb=temp_interp(gal.rb)
		temp_rs=temp_interp(gal.rs)

		x_ray_rb=x_ray_interp(gal.rb)
		x_ray_rs=x_ray_interp(gal.rs)
		
		ax[0].loglog(gal.radii, gal.rho, linestyle=linestyles[j],color=cols[i%len(cols)], label=gal.name)
		ax[0].loglog(gal.rb, rho_rb, 'o', color=cols[i%len(cols)], markersize=10)
		ax[0].loglog(gal.rs, rho_rs, 's', color=cols[i%len(cols)],  markersize=10)
		if j==1:
			ax[0].text(1.1*gal.rs, rho_rs,gal.name, color=cols[i%len(cols)])

		ax[1].loglog(gal.radii, gal.temp, linestyle=linestyles[j],color=cols[i%len(cols)])
		ax[1].loglog(gal.rb, temp_rb, 'o',color=cols[i%len(cols)], markersize=10)
		ax[1].loglog(gal.rs, temp_rs, 's',color=cols[i%len(cols)], markersize=10)
		if j==1:
			ax[1].text(1.1*gal.rs, temp_rs,gal.name, color=cols[i%len(cols)])

		ax[2].loglog(gal.radii, gal.x_ray_lum, linestyle=linestyles[j],color=cols[i%len(cols)])
		ax[2].loglog(gal.rb, x_ray_rb, 'o',color=cols[i%len(cols)], markersize=10)
		ax[2].loglog(gal.rs, x_ray_rs, 's',color=cols[i%len(cols)], markersize=10)
		if j==2:
			ax[2].text(1.1*gal.rs, x_ray_rs,gal.name, color=cols[i%len(cols)])

plt.savefig('profiles.png')