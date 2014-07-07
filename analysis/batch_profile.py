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


linestyles=['-', '-.',':']
vws=[200., 500.,1000.]
selection=['NGC3115', 'NGC1172', 'NGC4478']

cols=mpl.rcParams['axes.color_cycle']
for i in range(len(galaxies)):
	dirs=galaxy.bash_command('echo '+galaxies[i]+'/*')
	dirs=np.array(shlex.split(dirs))
	filt=np.array(map(check, dirs))
	print dirs,filt
	dirs=dirs[filt]
	vws_present=[float(d.split('_')[-1]) for d in dirs]

	if not np.in1d(gnames[i],selection):
		continue
	# filt=np.in1d(200., vws_present) and np.in1d(1000., vws_present)
	# if not filt:
	# 	continue

	print vws_present
	for j in range(len(vws)):
		if j!=2 and 
		
		filt=np.array([np.allclose(vw,vws[j]) for vw in vws_present])
		print filt
		try:
			print dirs[filt]
			d=dirs[filt][0]
			print d
		except:
			continue

		# grid2=pickle.load(open(d+'/grid.p', 'rb'))
		saved=np.load(d+'/save.npz')['a']
		start=galaxy.prepare_start(saved[-1])

		
		gal=galaxy.NukerGalaxy(galaxies[i], gal_dict, init_array=start)
		rho_interp=interp1d(gal.radii,gal.rho)
		rho_rb=rho_interp(gal.rb)
		rho_rs=rho_interp(gal.rs)

		
		ax[0].loglog(gal.radii, gal.rho, linestyle=linestyles[j],color=cols[i%len(cols)], label=gal.name)
		ax[0].loglog(gal.rb, rho_rb, 'o',color=cols[i%len(cols)])
		ax[0].loglog(gal.rs, rho_rs, 's',color=cols[i%len(cols)])
		#handles, labels = ax[0].get_legend_handles_labels()
		#ax[0].legend(handles, labels)
		if j==2:
			ax[0].text(gal.rs, rho_rs,gal.name, color=cols[i%len(cols)])

		ax[1].loglog(gal.radii, gal.temp, linestyle=linestyles[j],color=cols[i%len(cols)])
		ax[2].loglog(gal.radii, gal.x_ray_lum, linestyle=linestyles[j],color=cols[i%len(cols)])


plt.savefig('profiles.png')