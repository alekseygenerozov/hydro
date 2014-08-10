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

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value



gal_dict=galaxy.nuker_params()
fig,ax=plt.subplots(2, sharex=False, figsize=(10,  16))
ax[0].set_xlabel(r'$v_w/\sigma$')
ax[0].set_ylabel(r'$r_{\rm stag}/r_{\rm soi}$')

ax[1].set_ylabel('Frational difference\n from  analytic prediction')

vws=[200., 500., 1000.]
# selection=['NGC3115', 'NGC1172', 'NGC4478']
cols=brewer2mpl.get_map('Set2', 'qualitative', 3).mpl_colors

def glaw(eta, omega=1., dens_slope=1, ad_index=5./3., gamma=1.):
	A=(4.*ad_index-(1+gamma)*(ad_index-1.))/(4.*(ad_index-1.))
	return 1./(dens_slope*eta**2)*((0.5)*(2*A-dens_slope)*(1+omega)-(2.-gamma)/4.)

dens_slopes=[]

eta_core=[]
eta_cusp=[]
x_core=[]
x_cusp=[]
dens_slope_cusp=[]
dens_slope_core=[]
idx=0
for name in gal_dict.keys():
	base_d='/Users/aleksey/Second_Year_Project/hydro/batch_collected/'+name
	gal_data='/Users/aleksey/Second_Year_Project/hydro/gal_data/'+name
	for j,vw in enumerate(vws):
		d=base_d+'/vw_'+str(vw)
		if not sc.check(d):
			continue

		gal=dill.load(open(d+'/grid.p', 'rb'))
		
		try:
			rsoi=np.genfromtxt(gal_data+'/rsoi')
			sigma=np.genfromtxt(gal_data+'/sigma')
		except:
			continue
		sigma_interp=interp1d(sigma[:,0], sigma[:,1])

		rho_interp=interp1d(np.log(gal.radii), np.log(gal.rho))
		dens_slope=np.abs(derivative(rho_interp, np.log(gal.rs), dx=gal.delta_log[0]))
		dens_slopes.append(dens_slope)

		x=gal.rs/pc/rsoi
		vw_eff=(sigma_interp(gal.rs)**2.+(vw*1.E5)**2.)**0.5
		eta=vw*1.E5/sigma_interp(rsoi*pc)

		M_enc_rs=(sigma_interp(gal.rs)**2*gal.rs/G)-gal.params['M']
		omega=M_enc_rs/gal.params['M']

		predicted=glaw(eta)
		residual=(predicted-x)/predicted
		predicted2=glaw(eta, omega=omega, dens_slope=dens_slope, gamma=gal.params['gamma'])
		residual2=(predicted2-x)/predicted2

		if gal.params['gamma']<0.2:
			symbol='<'
			eta_core.append(eta)
			x_core.append(x)
			dens_slope_core.append(dens_slope)
		else:
			symbol='s'
			eta_cusp.append(eta)
			x_cusp.append(x)
			dens_slope_cusp.append(dens_slope)

		ax[0].loglog(eta, x, symbol, color=cols[j], markersize=10)
		# etas=[1.,10.]
		ax[1].plot(idx, residual2, symbol, color=cols[j], markersize=10)
		idx+=1



pow_cusp, coeff_cusp=np.polyfit(np.log(eta_cusp),np.log(x_cusp),1)
pow_core, coeff_core=np.polyfit(np.log(eta_core),np.log(x_core),1)

print np.mean(dens_slope_cusp)
print np.mean(dens_slope_core)
print len(x_core)

# =np.polyfit(np.log(x_core),np.log(eta_core),1)
etas=[0.2,20.]
print [coeff_cusp*eta**pow_cusp for eta in etas]
ax[0].loglog(etas, [np.exp(coeff_cusp)*eta**pow_cusp for eta in etas], 'k')
ax[0].loglog(etas, [np.exp(coeff_core)*eta**pow_core for eta in etas], 'k--')

ax[1].tick_params(\
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.savefig('rs.pdf')