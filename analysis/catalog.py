import re
from .. import galaxy
from .. import gal_properties as gp

from bash_command import bash_command as bc
import numpy as np
 
import matplotlib.pylab as plt
import dill

from scipy.interpolate import interp1d
import brewer2mpl

from scipy.misc import derivative
import time
from mpldatacursor import datacursor

import operator

import astropy.constants as const
from astropy.table import Table
from astropy.io import ascii

import warnings

from custom_collections import LastUpdatedOrderedDict as od
from latex_exp import latex_exp

import sys
import shlex


#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value
th=4.35*10**17
year=3.15569E7

class Catalog(object):
	def __init__(self, base_d, masses=[1.E6, 1.E7, 1.E8], vws=[200., 500., 1000.], gammas=[0.1,0.8], rbs=[100.], bad_gals=False, extra_dirs=[],exclude_dirs=[]):
		'''Catalog of designed to store group of galaxy solutions

		:param vws: list of vws to include in our grid
		:param bad_gals: flag specifying whether or not to include galaxies which did not satisfy energy/mass conservation.
		:param extra_dirs: list of manually specified extra directories to include. 
		'''
		self.base_d=base_d
		# gal_dict=galaxy.nuker_params()
		# self.base_names=gal_dict.keys()
		self.vws=vws
		if len(vws)<3:
			len_cols=3
		elif len(vws)>8:
			len_cols=8
		else:
			len_cols=len(vws)	
		#Use colors to distinguish different choices of vw_0, 
		self.cols=brewer2mpl.get_map('Set2', 'qualitative', len_cols).mpl_colors
		#And use different shapes to distinguish cores from cusps.
		self.symbols=['s','>']
		self.cusp_thres=0.2

		self.gals_full=[]
		self.gal_vws_full=[]
		self.index_full={}
		self.dirs=[]

		for mass in masses: 
			for j,vw in enumerate(vws):
				for gamma in gammas:
					for rb in rbs:
						d=base_d+'/M{0:2.1g}_vw{1}_gamma{2:.1f}_rb{3}'.format(mass, vw, gamma, rb)
						try:
							gal=dill.load(open(d+'/grid.p', 'rb'))
						except:
							continue
						if d in exclude_dirs:
							continue

						if not hasattr(gal, 'check_partial'):
							gal.cons_check(write=False)

						if (not bad_gals and not gal.check_partial):
							print 'bad!', d
							continue
						if gal.vw_extra!=vw*1.E5:
							print gal.name
							continue
						if len(gal.rs)!=1:
							print gal.name+' '+str(gal.params['M'])+' '+str(gal.vw_extra/1.E5)+' has more than 1 stagnation point'
							continue

						self.gals_full.append(gal)
						self.gal_vws_full.append(j)
						self.index_full[(gal.name,vw)]=len(self.gals_full)-1
						self.dirs.append(d)


		for name in extra_dirs:
			d=base_d+'/'+name
			try:
				gal=dill.load(open(d+'/grid.p', 'rb'))
			except:
				continue

			if not hasattr(gal, 'check_partial'):
				gal.cons_check(write=False)

			if (not bad_gals and not gal.check_partial):
				continue
			if len(gal.rs)!=1:
				print gal.name+' '+str(gal.params['M'])+' '+str(gal.vw_extra/1.E5)+' has more than 1 stagnation point'
				continue

			self.gals_full.append(gal)
			try:
				j=np.where(np.isclose(vws,gal.vw_extra/1.E5))[0][0]
			except IndexError:
				j=len(vws)-1
			self.gal_vws_full.append(j)
			self.index_full[(gal.name,gal.vw_extra/1.E5)]=len(self.gals_full)-1
			self.dirs.append(d)

		self.gals_full=np.array(self.gals_full)
		self.gal_vws_full=np.array(self.gal_vws_full)
		self.filt=np.array(range(len(self.gals_full)))
		# self.restore_saved()

	@property 
	def gal_symbols_full(self):
		symbols=np.empty(len(self.gals_full),dtype=int)
		for idx, gal in enumerate(self.gals_full):
			if gal.params['gamma']>self.cusp_thres:
				symbols[idx]=0
			else:
				symbols[idx]=1
		return symbols

	@property
	def gal_symbols(self):
		return self.gal_symbols_full[self.filt]

	def restore_saved(self):
		[gal.restore_saved(self.dirs[idx]) for idx,gal in enumerate(self.gals)]
			
	def select_subset(self, param, val, compare=operator.eq):
		'''Select a subset of the galaxy catalog based on some condition

		:param string param: galaxy parameter to compare
		:param val: value to compare the 
		'''
		params=np.array([gal.get_param(param) for gal in self.gals_full])
		val=np.array([val]).flatten()
		filt=np.empty(0, dtype=int)
		for v in val:
			filt=np.append(filt, np.where(compare(params,v))[0])
		filt=filt.flatten()
		self.filt=np.intersect1d(filt,self.filt)

	def subset_off(self):
		self.filt=range(len(self.gals_full))

	@property
	def gals(self):
		return self.gals_full[self.filt]

	@property
	def names(self):
		return [gal.name for gal in self.gals]

	@property
	def gal_vws(self):
		return self.gal_vws_full[self.filt]

	def mdot_mass(self, gammas=[1.], eta=0.1, lines=['-']):
		fig,ax=plt.subplots(figsize=(10,8))
		mass_anal=np.logspace(5.,9.,30)

		gammas=np.array([gammas]).flatten()
		for idx0,vw in enumerate(np.array(self.vws)):
			for g,gamma in enumerate(gammas):
				ax.loglog(mass_anal, [gp.eddr_analytic(m*M_sun, vw*1.E5, gamma=gamma, eta=eta) for m in mass_anal], lines[g%len(lines)], color=self.cols[idx0], linewidth=5)
		for idx, gal in enumerate(self.gals):
			ax.set_xlabel(r'$\mathbf{M_{\bullet}/M_{\odot}}$')
			ax.set_ylabel(r'$\mathbf{\dot{M}/\dot{M}_{Edd}}$')
			ax.loglog(gal.params['M']/M_sun, (eta/gal.eta)*gal.eddr,  marker=self.symbols[self.gal_symbols[idx]], color=self.cols[self.gal_vws[idx]], label=gal.name)
		datacursor(formatter='{label}'.format)
		plt.close()
		return fig

	def rs_mass(self, gamma=1., eta=0.1):
		fig,ax=plt.subplots(figsize=(10,8))
		mass_anal=np.logspace(6,9.3,30)
		for idx0,vw in enumerate(np.array(self.vws)):
			ax.loglog(mass_anal, [gp.rs_approx(m*M_sun, vw*1.E5) for m in mass_anal], color=self.cols[idx0], linestyle='--')
		for idx, gal in enumerate(self.gals):
			ax.set_xlabel(r'$M_{\bullet}/M_{\odot}$')
			ax.set_ylabel(r'$r_s$ [cm]')
			ax.loglog(gal.params['M']/M_sun, gal.rs[0],  marker=self.symbols[self.gal_symbols[idx]], color=self.cols[self.gal_vws[idx]], label=gal.name+str(gal.dens_pow_slope_rs))
		return fig

	def rs(self):
		fig,ax=plt.subplots(2, sharex=False, figsize=(10,16))
		ax[1].tick_params(\
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelbottom='off')

		ax[0].set_xlabel(r'$\mathbf{v_w/\sigma}$')
		ax[0].set_ylabel(r'$\mathbf{r_{s}/r_{soi}}$')
		ax[1].set_ylabel('Frational difference\n from  analytic prediction')
		#ax[1].set_ylim(-0.1, 0.8)

		eta_analytic=[15.,0.3]
		ax[0].loglog(eta_analytic, [gp.rs_approx_rinf(eta) for eta in eta_analytic])
		for idx, gal in enumerate(self.gals):
				x=gal.rs[0]/gal.rinf
				vw_eff=(gal.sigma_inf**2.+gal.vw_extra**2.)**0.5
				#eta=vw_eff/gal.sigma_inf
				eta=gal.vw_extra/gal.sigma_inf

				ax[0].loglog(eta, x, marker=self.symbols[self.gal_symbols[idx]], label=gal.name, color=self.cols[self.gal_vws[idx]])
				ax[1].plot(idx, gal.rs_residual,  marker=self.symbols[self.gal_symbols[idx]], label=gal.name, color=self.cols[self.gal_vws[idx]])
				#ax[2].loglog(eta2, x, marker=self.symbols[self.gal_symbols[idx]], label=gal.name, color=self.cols[self.gal_vws[idx]])
		datacursor(formatter='{label}'.format)
		plt.close()
		return fig

	def cooling(self, eta=0.1):
		fig,ax=plt.subplots(1, sharex=True, figsize=(10, 8))
		ax.set_xlabel('Radius [cm]')
		ax.set_ylabel('H/C')

		for idx, gal in enumerate(self.gals):
			ax.loglog(gal.radii, (gal.eta/eta)*gal.heating_pos/gal.cooling, color=self.cols[self.gal_vws[idx]],linewidth=5)
			ax.loglog(gal.radii, (gal.eta/eta)*gal.tcool_tff, '--', color=self.cols[self.gal_vws[idx]], linewidth=5)
			ax.loglog(gal.rs[0], (gal.eta/eta)*gal.tcool_tff_rs, 's', color=self.cols[self.gal_vws[idx]], linewidth=5, label=r'$v_{w,0}$='+str(gal.vw_extra/1.E5)+'km/s')
		ax.legend()
		plt.close()
		return fig

	def bh_xray(self, eps1=5.E-7, eps2=2.E-4):
		fig,ax=plt.subplots(2, figsize=(10,16))

		ax[0].set_xlabel(r'$\mathbf{M_{gal}}$')
		ax[0].set_ylabel(latex_exp.latex_exp(eps1, precision=0)+r' $\mathbf{\dot{M} c^2}$ [ergs/s]')
		ax[1].set_xlabel(r'$\mathbf{M_{gal}}$')
		ax[1].set_ylabel(latex_exp.latex_exp(eps2, precision=0)+r' $\mathbf{\left(\dot{M}/\dot{M}_{Edd}\right)^2 \dot{M}_{Edd} c^2}$ [ergs/s]')
		#ax[1].set_ylabel(r'$\mathbf{'+latex_exp.latex_exp(eps2, precision=0)+r'}$'+r' $\mathbf{\left(\dot{M}/\dot{M}_{Edd}\right)^2 \dot{M}_{Edd} c^2}$ [ergs/s]')

		for idx, gal in enumerate(self.gals):
			if not (gal.vw_extra==2.E7):
				continue
			stellar_mass=gal.mstar_bulge/galaxy.M_sun

			ax[0].loglog([stellar_mass], [eps1*gal.mdot*c**2],  marker=self.symbols[self.gal_symbols[idx]], color=self.cols[0], label=(gal.name,'{0:3.2e}'.format(gal.eddr)))
			ax[1].loglog([stellar_mass], [eps2*(gal.eddr)**2*gal.mdot_edd*c**2],  marker=self.symbols[self.gal_symbols[idx]],color=self.cols[0], label=(gal.name,'{0:3.2e}'.format(gal.eddr)))

		m1=1.E8
		m2=1.E12
		lum1=galaxy.pow_extrap(m1,10.**11.96, 10.**11.48, 10.**39.62, 10.**39.23)
		lum2=galaxy.pow_extrap(m2,10.**11.96, 10.**11.48, 10.**39.62, 10.**39.23)
		for i in range(2):
			ax[i].loglog([m1, m2], [lum1, lum2])
			ax[i].loglog([m1, m2], [10.**38.25, 10.**38.25])

		plt.close()
		return fig

	def profiles(self, labels=[]):
		fig,ax=plt.subplots(3, sharex=True, figsize=(10, 24))
		ax[0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
		ax[1].set_ylabel(r'T [K]')
		ax[2].set_ylabel('Cumulative x-ray luminosity [ergs/s]')
		ax[2].set_xlabel('r [cm]')
		# ax.set_ylabel(r'$r_{\rm stag}/r_{\rm soi}$')

		for idx, gal in enumerate(self.gals):
			gal_label=gal.name
			try:
				labels=np.array([labels]).flatten()
			except:
				pass
			try:
				for label in labels:
					gal_label=gal_label+' '+str(gal.get_param(label))
			except:
				print sys.exc_info()[0]
				pass

			col=self.cols[self.gal_vws[idx]]

			rho_interp=interp1d(gal.radii,gal.rho)
			temp_interp=interp1d(gal.radii,gal.temp)
			x_ray_interp=interp1d(gal.radii, gal.x_ray_lum)
			rho_rs=gal.rho_interp(gal.rs[0])
			temp_rs=gal.temp_interp(gal.rs[0])
			x_ray_rs=gal.x_ray_lum_interp(gal.rs[0])

			ax[0].loglog(gal.radii, gal.rho, color=col, label=gal_label)
			ax[0].loglog(gal.rs, rho_rs, 's', color=col,  markersize=10)
			ax[1].loglog(gal.radii, gal.temp, color=col, label=gal_label)
			ax[1].loglog(gal.rs, temp_rs, 's',color=col, markersize=10)
			ax[2].loglog(gal.radii, gal.x_ray_lum, color=col, label=gal_label)
			ax[2].loglog(gal.rs, x_ray_rs, 's',color=col, markersize=10)
			try:
				rb=gal.rb
			except:
				continue

			try:
				rho_rb=gal.rho_interp(rb)
				temp_rb=gal.temp_interp(rb)
				x_ray_rb=gal.x_ray_lum_interp(rb)
			except:
				continue
				
			ax[0].loglog(gal.rb, rho_rb, 'o', color=col, markersize=10)
			ax[1].loglog(gal.rb, temp_rb, 'o',color=col, markersize=10)
			ax[2].loglog(gal.rb, x_ray_rb, 'o',color=col, markersize=10)

		return fig,ax

	def rs_rIa(self, analytic=True):
		fig,ax=plt.subplots(1, figsize=(10, 8))
		ax.set_xlabel(r'$M_{\bullet}/M_{\odot}$')
		ax.set_ylabel(r'$r_s/r_{\rm Ia}$')

		for idx, gal in enumerate(self.gals):
			ratio=gal.rs/gal.r_Ia
			mass=gal.params['M']/galaxy.M_sun
			if gal.params['gamma']<0.2:
				symbol='<'
			else:
				symbol='s'

			ax.loglog(mass, ratio, marker=symbol, color=self.cols[self.gal_vws[idx]])
			if analytic:
				mass_anal=[1.E6, 1.E10]
				rs_anal=[(7./4.)*G*M*M_sun/((gal.vw_extra)**2./2.)/gp.r_Ia(M*M_sun) for M in mass_anal]

				ax.loglog(mass_anal, rs_anal, color=self.cols[self.gal_vws[idx]], markersize=10)

		return fig,ax

	def cooling_table(self):
		eta_max=[[],[],[]]
		for name in self.base_names:
			for j,vw in enumerate(self.vws):
				try:
					idx=self.index[(name,vw)]
				except KeyError:
					eta_max[j].append('')
					continue

				gal=self.gals[idx]
				heating=gal.q_grid*(0.5*gal.vel**2+0.5*(gal.vw_extra)**2+0.5*gal.sigma_grid**2)
				cooling=gal.cooling
				eta_max[self.gal_vws[idx]].append('%s' % float('%.1g' % min(heating/cooling)))

		eta_str=r'$\eta_{\rm max}$'
		vw_str=r'$v_{\rm w,0}=$'
		eta_str=eta_str+' '+vw_str
		
		eta_tab=Table([self.base_names, eta_max[0], eta_max[1], eta_max[2]], names=['Galaxy', eta_str+'200 km/s', eta_str+'500 km/s', eta_str+'1000 km/s'])
		for i in range(1,4):
			eta_tab[i].format='%3.2e'
		print eta_tab
		eta_tab.write('eta_tab.tex', format='latex', latexdict={'header_start':'\hline','header_end':'\hline','data_end':'\hline','caption':r'\label{tab:eta} Table of the maximum\
		 $\eta$ for which each of our galaxies would have $H/C$ (Heating Rate/Cooling Rate)$>1$'})

		return eta_tab

	def gen_table(self, fields):
		'''Generate table of properties for catalog'''
		cols=od({})
		cols[r'$M_{bh}$']=[]
		cols[r'$v_{w,0}$']=[]
		for field in fields:
			cols[field]=[]

		for i, gal in enumerate(self.gals):
			cols[r'$M_{bh}$'].append(gal.params['M']/M_sun)
			cols[r'$v_{w,0}$'].append(gal.vw_extra/10.**5)
			for j,field in enumerate(fields):
				cols[field].append(gal.get_param(field))

		return Table(cols)

	def gen_plot(self, field1, field2):
		'''Generate plot of field1 vs. field2 for galaxies in catalog. Field1 and field2 are 2 galaxy properties'''
		fig=plt.figure()
		for i, gal in enumerate(self.gals):
			plt.loglog(gal.get_param(field2), gal.get_param(field1),'ks',label=gal.name)
		return fig

	def cond_plots(self):
		bc('mkdir -p conduction')
		for gal in self.gals:
			gal.set_param('eps_cond',1.)
			gal.set_param('cond_scheme', 'spitzer')
			gal.set_param('phi_cond',0.05)
			gal._update_aux()

			fig=gal.cond_plot
			fig.savefig('conduction/'+gal.name+'_'+str(gal.vw_extra/1.E5)+'_conduction.pdf')
			plt.close()

	def cond_ratios(self):
		bc('mkdir -p conduction')
		fig,ax=plt.subplots(1, figsize=(10, 8))
		for gal in self.gals:
			gal.set_param('eps_cond',1.)
			gal.set_param('cond_scheme', 'spitzer')
			gal.set_param('phi_cond',0.05)
			gal._update_aux()

			try:
				ax.loglog(gal.M_bh_8, gal.field_interp('cond_grid')(1.E18)/gal.field_interp('heating_pos')(1.E18), 'ks')
			except:
				continue
		
		plt.close()
		return fig

	def cond_ratios_temp(self):
		bc('mkdir -p conduction')
		fig,ax=plt.subplots(1, figsize=(10, 8))
		for gal in self.gals:
			gal.set_param('eps_cond',1.)
			gal.set_param('cond_scheme', 'spitzer')
			gal.set_param('phi_cond',0.05)
			gal._update_aux()

			try:
				ax.loglog(gal.temp_interp(1.E18), gal.field_interp('cond_grid')(1.E18)/gal.field_interp('heating_pos')(1.E18), 'ks')
			except:
				continue
		
		plt.close()
		return fig

	def cond_ratios_gamma(self):
		bc('mkdir -p conduction')
		fig,ax=plt.subplots(1, figsize=(10, 8))
		for gal in self.gals:
			gal.set_param('eps_cond',1.)
			gal.set_param('cond_scheme', 'spitzer')
			gal.set_param('phi_cond',0.05)
			gal._update_aux()

			try:
				ax.loglog(gal.M_bh_8, gal.field_interp('cond_grid')(1.E18)/gal.field_interp('heating_pos')(1.E18), 'ks')
			except:
				continue
		plt.close()
		return fig

	def q_plot(self, scale_radius=True):
		fig,ax=plt.subplots(1, figsize=(10, 8))
		ax.set_xlabel(r'$r/r_{\rm inf}$')
		ax.set_ylabel(r'$q $ g cm$^{-3}$ s$^{-1}$')

		for idx, gal in enumerate(self.gals):
			if scale_radius:
				ax.loglog(gal.radii/gal.rinf, gal.q_grid, label=gal.name)
			else:
				ax.loglog(gal.radii, gal.q_grid, label=gal.name)

		return fig, ax

	def convergence(self):
		props=['sol_plot_seq', 'conv_plot_cons', 'conv_plot_sol']
		for prop in props:
			bc('mkdir -p '+self.base_d+'/'+prop)
		
		for idx, gal in enumerate(self.gals):
			for prop in props:
				fig=getattr(gal, prop)
				fig.suptitle(gal.name+'_gamma{0}_M{1:2.1e}_vw{2:2.1e}'.format(gal.params['gamma'],\
					gal.params['M'], gal.vw_extra)) 
				fig.savefig(self.base_d+'/'+prop+'/'+str(idx)+'.pdf')

	def chandra_compare(self):
		fig, ax = plt.subplots(ncols=2, figsize=(10,5))
		ax[0].set_xscale('log')
		ax[1].set_xscale('log')
		
		ratio=np.array([gal.mdot_bondi_ratio for gal in self.gals])
		ratio2=np.array([gal.chandra_mdot_ratio for gal in self.gals])
		ax[0].hist(ratio, bins=np.logspace(-3,3,60), color='b', label=self.names)
		ax[1].hist(ratio2, bins=np.logspace(-3,3,60), color='b', label=self.names)

		return fig


	# def input_gen(self, param, target):
	# 	bc('mkdir -p input/')
	# 	col_names=['pickle', 'param', 'target', 'outdir']
	# 	for idx, gal in enumerate(self.gals):
	# 		d=gal.name+'/vw_{0}'.format(gal.vw_extra/1.E5)
	# 		d2=gal.name+'/vw_{0}_{1}_{2}'.format(gal.vw_extra/1.E5, param, target)
	# 		tab=Table([[d+'/grid.p'], [param], [target], [d2]], names=col_names)
	# 		ascii.write(tab,'input/input_{0}'.format(idx))

	def paper_plot_gen(self, outdir, gammas=[1.], eta=0.1, lines=['-']):
		'''Generate plots for our sample'''
		fig_rs=self.rs()
		fig_rs.savefig(outdir+'/rs.eps')
		fig_cooling=self.cooling(eta=eta)
		fig_cooling.savefig(outdir+'/cooling.eps')
		fig_mdot=self.mdot_mass(gammas=gammas, eta=eta, lines=lines)
		fig_mdot.savefig(outdir+'/mdot_mass.eps')
		# fig_bh_xray=self.bh_xray()
		# fig_bh_xray.savefig(outdir+'/bh_xray.eps')

