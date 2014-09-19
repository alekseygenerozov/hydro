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
import dill

from scipy.interpolate import interp1d

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value




fig,ax=plt.subplots(3, sharex=True, figsize=(10, 24))
ax[0].set_ylabel(r'$\mathbf{\rho}$ [g cm$^{-3}$]')
ax[1].set_ylabel(r'T [K]')
ax[2].set_ylabel('|v| [cm/s]')
ax[2].set_xlabel('Radius [cm]')
# ax.set_ylabel(r'$r_{\rm stag}/r_{\rm soi}$')


linestyles=['-.', '-']
vws=[200., 500.]
selection=['NGC4365', 'NGC1172', 'NGC4478']
base_dir='/Users/aleksey/Second_Year_Project/hydro/batch_collected/'

cols=mpl.rcParams['axes.color_cycle']
for i,name in enumerate(selection):
	base_d=base_dir+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		try:
			gal=dill.load(open('{0}/{1}/vw_{2}/grid.p'.format(base_dir, name, vw),'rb'))
		except:
			continue
		gal.cons_check(write=False)
		if not gal.check_partial:
			continue
		if j==0 and gal.name!='NGC4478':
			continue

		rho_interp=interp1d(gal.radii,gal.rho)
		temp_interp=interp1d(gal.radii,gal.temp)
		x_ray_interp=interp1d(gal.radii, gal.x_ray_lum)

		rho_rb=rho_interp(gal.rb)
		rho_rs=rho_interp(gal.rs)

		temp_rb=temp_interp(gal.rb)
		temp_rs=temp_interp(gal.rs)

		x_ray_rb=x_ray_interp(gal.rb)
		x_ray_rs=x_ray_interp(gal.rs)

		print gal.dens_pow_slope_rs
		
		ax[0].loglog(gal.radii, gal.rho, linestyle=linestyles[j],color=cols[i%len(cols)], label=gal.name)
		ax[0].loglog(gal.rb, rho_rb, 'o', color=cols[i%len(cols)], markersize=10)
		ax[0].loglog(gal.rs, rho_rs, 's', color=cols[i%len(cols)],  markersize=10)
		if j==1:
			if name=='NGC4478':
				ax[0].text(0.1*gal.radii[0], gal.rho[0],gal.name, color=cols[i%len(cols)])
			elif name=='NGC1172':
				ax[0].text(gal.radii[0], 0.9*gal.rho[0],gal.name, color=cols[i%len(cols)])
			else:
				ax[0].text(gal.radii[0], gal.rho[0],gal.name, color=cols[i%len(cols)])

		ax[1].loglog(gal.radii, gal.temp, linestyle=linestyles[j],color=cols[i%len(cols)])
		ax[1].loglog(gal.rb, temp_rb, 'o',color=cols[i%len(cols)], markersize=10)
		ax[1].loglog(gal.rs, temp_rs, 's',color=cols[i%len(cols)], markersize=10)
		if j==1:
			ax[1].text(gal.radii[0], gal.temp[0],gal.name, color=cols[i%len(cols)])


		# ax[2].loglog(gal.radii, gal.x_ray_lum, linestyle=linestyles[j],color=cols[i%len(cols)])
		# ax[2].loglog(gal.rb, x_ray_rb, 'o',color=cols[i%len(cols)], markersize=10)
		# ax[2].loglog(gal.rs, x_ray_rs, 's',color=cols[i%len(cols)], markersize=10)
		# if j==1:
		# 	ax[2].text(1.1*gal.rs[0], x_ray_rs,gal.name, color=cols[i%len(cols)])
		ax[2].loglog(gal.radii, abs(gal.vel), linestyle=linestyles[j],color=cols[i%len(cols)])
		ax[2].loglog(gal.rb, abs(gal.vel_interp(gal.rb)), 'o',color=cols[i%len(cols)], markersize=10)
		#ax[2].loglog(gal.rs, gal.vel_interp(gal.rs), 's',color=cols[i%len(cols)], markersize=10)
		if j==1:
			if name=='NGC4478':
				ax[2].text(0.5*gal.radii[0], gal.temp[0],gal.name, color=cols[i%len(cols)])
			elif name=='NGC1172':
				ax[2].text(gal.radii[0], 0.9*gal.temp[0],gal.name, color=cols[i%len(cols)])
			else:
				ax[2].text(gal.radii[0], abs(gal.vel[0]),gal.name, color=cols[i%len(cols)])


plt.savefig('/Users/aleksey/Second_Year_Project/SMBH_env/Figures/profiles.eps')