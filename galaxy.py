import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import copy
import warnings
import pickle
import sys
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import e

import astropy.constants as const
from astropy.io import ascii
import astropy.table as table
from astropy.table import Table,Column
import astropy.units as u

import tde_jet

import progressbar as progress
import ipyani

import os.path

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value
th=4.35*10**17
#params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7.)


def bash_command(cmd):
	'''Run command from the bash shell'''
	process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	return process.communicate()[0]

def lambda_c(temp):
	'''Power law approximation to cooling function for solar abundance plasma (taken from figure 34.1 of Draine)'''
	if temp>2.E7:
		return 2.3E-24*(temp/1.E6)**0.5
	else:
		return 1.1E-22*(temp/1.E6)**-0.7

def nuker(r, Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., **kwargs):
	'''Nuker parameterization'''
	return Ib*2.**((beta-gamma)/alpha)*(rb/r)**gamma*(1.+(r/rb)**alpha)**((gamma-beta)/alpha)

##Derivative of nuker parameterized surface brightness profile
def nuker_prime(r, Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., **kwargs):
	'''First derivative of nuker paramerization'''
	return -((2**((beta - gamma)/alpha)*Ib*(gamma + beta*(r/rb)**alpha)*(rb/r)**gamma)/ (r*(1 + (r/rb)**alpha)**((alpha + beta - gamma)/alpha)))
	
##Return the inverse abel transform of a function 
def inverse_abel(i_prime, r, **kwargs):
	'''Inverse abel transform

	:param i_prime: First derivative of surface brightness profile
	'''
	f=lambda y: i_prime(y, **kwargs)/np.sqrt(y**2-r**2)
	return -(1/np.pi)*integrate.quad(f, r, np.inf)[0]

##Convert from magnitudes per arcsec^2 to luminosities per parsec^2
def mub_to_Ib(mub):
	'''Convert from magnitudes per arcsec^2 to luminosities per pc^2'''
	Msun=4.83
	return 10**(-(mub-Msun-21.572)/2.5)

def nuker_params(skip=False):
	'''Reads Wang & Merritt parameter table into python dictionary

	:param skip: skip over galaxies discarded by Wang and Merritt
	'''
	table=ascii.read('wm')
	galaxies=dict()
	for i in range(len(table)):
		d=dict()
		if table['$\\log_{10}\\dot{N}$\\tablenotemark{f}'][i]=='-' and skip:
			continue
		d['Name']=table[i]['Name']
		d['Uv']=table[i]['$\\Upsilon_V $']
		d['alpha']=table[i]['$\\alpha$']
		d['beta']=table[i]['$\\beta$']
		d['gamma']=table[i]['$\\Gamma$']
		d['rb']=10.**table[i]['$\log_{10}(r_{b})$']
		d['Ib']=mub_to_Ib(table[i]['$\mu_b$'])
		d['mub']=table[i]['$\mu_b$']
		d['d']=table[i]['Distance']
		d['M']=M_sun*10.**table[i]['$\\log_{10}(M_{\\bullet}/M_{\\odot})$\\tablenotemark{e}']
		d['type']=table[i][r'Profile\tablenotemark{b}']
		if d['type']=='$\\cap$':
			d['type']='core'
		else:
			d['type']='cusp'
		galaxies[table[i]['Name']]=d

	return galaxies


#Compute logarithmic slope given values and radii
def get_slope(r1, r2, f1, f2):
	return np.log(f2/f1)/np.log(r2/r1)


def extend_to_ss(dat):
	'''Extend inner grid boundary to be supersonic'''
	slope=np.empty(5)
	field=np.copy(dat[0])
	for i in range(len(field)):
		slope[i]=get_slope(dat[0,0], dat[3,0], dat[0,i], dat[3,i])
	#Getting target radius
	r_end=dat[0,0]*np.exp((1./slope[2])*np.log(-2./dat[0,2]))

	#Computing the grid spacing
	delta_log=np.log(dat[1,0]/dat[0,0])
	r=dat[0,0]
	logr=np.log(dat[0,0])
	dat_extend=np.copy(dat)
	while r>r_end:
		logr=logr-delta_log
		r=np.exp(logr)
		for i in range(len(field)):
			field[i]=field[i]*(r/dat_extend[0,0])**slope[i]
		#Stacking to the end of the grid
		dat_extend=np.vstack([field, dat_extend])
	return dat_extend
		
		
def prepare_start(end_state, rescale=1):

	# end_state=dat[-70:]
	end_state[:,2]=end_state[:,2]*end_state[:,-1]
	end_state[:,1]=np.log(end_state[:,1])
	start=end_state[:,:4]
	#rescaling radial grid. This will be useful for going from one mass black hole to another. 
	start[:,0]=start[:,0]*rescale

	return start


class Galaxy(object):
	'''Class to representing galaxy--Corresponds to Quataert 2004. Can be initialized either using an analytic function or a
	numpy ndarray object

	:param init: list contain minimum radius of the grid [cm], maximum radius of the grid [cm], and function to evaluate 
		initial conditions
	:param init_array: array containing initial condition for grid.
	'''

	def __init__(self, init=None, init_array=None):
		self.logr=True
		self.init_params=dict()
		try:
			self.params
		except:
			self.params={'M':3.6E6*M_sun}
		try:
			self.rmin_star
		except:
			self.rmin_star=0.
		try:
			self.rmax_star
		except:
			self.rmax_star=0.


		#Initializing the radial grid
		if init!=None:
			assert len(init)==3 or len(init)==4
			r1,r2,f_initial=init[0],init[1],init[2]
			if len(init)==4:
				self.init_params=init[3]

			assert r2>r1
			assert hasattr(f_initial, '__call__')
			self.length=70

			if self.logr:
				self.radii=np.logspace(np.log(r1), np.log(r2), self.length, base=e)
			else:
				self.radii=np.linspace(r1, r2, self.length)
			prims=[f_initial(r, **self.init_params) for r in self.radii]
			prims=np.array(prims)
			# for i in range(len(self.radii)):
			# 	prims[i]=f_initial(self.radii[i], **params)

		elif type(init_array)==np.ndarray:
			self.radii=init_array[:,0]
			prims=init_array[:,1:]
			self.length=len(self.radii)
		else:
			raise Exception("Not enough initialization info entered!")

		#Attributes to store length of the list as well as start and end indices (useful for ghost zones)
		self.start=0
		self.end=self.length-1
		self.tcross=(self.radii[-1]-self.radii[0])/(kb*1.E7/mp)**0.5
		self._num_ghosts=3
		self.tinterval=0.05*self.tcross
		self.sinterval=100

		self.isot=False
		self.gamma=5./3.
		self.fields=['log_rho', 'vel', 's']
		self.mu=1.

		self.visc_scheme='default'
		self.Re=90.
		self.Re_s=1.E20
		self.floor=0.
		self.safety=0.6
		self.bdry='default'
		self.bdry_fixed=False
		
		# self.q_grid=np.array([self.q(r) for r in self.radii/pc])/pc**3
		self.vw_extra=1.E8
		# self.vw=np.array([(self.sigma(r/pc)**2+(self.vw_extra)**2)**0.5 for r in self.radii])

		self.eps=1.
		# self.place_mass()

		self.out_fields=['radii', 'rho', 'vel', 'temp', 'frho', 'bernoulli', 's', 'cs']
		self.cons_fields=['frho', 'bernoulli', 's', 'fen']
		self.src_fields=['src_rho', 'src_v', 'src_s', 'src_en']
		self.movies=False
		self.outdir='.'

		#Will store values of time derivatives at each time step
		self.time_derivs=np.zeros(self.length, dtype={'names':['log_rho', 'vel', 's'], 'formats':['float64', 'float64', 'float64']})
		#Initializing the grid using the initial value function f_initial
		self.log_rho=prims[:,0]
		self.rho=np.exp(self.log_rho)
		self.vel=prims[:,1]
		self.temp=prims[:,2]
		self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))


		self._add_ghosts()
		#Computing differences between all of the grid elements 
		delta=np.diff(self.radii)
		self.delta=np.insert(delta, 0, delta[0])
		delta_log=np.diff(np.log(self.radii))
		self.delta_log=np.insert(delta_log, 0, delta_log[0])
		#Coefficients use to calculate the derivatives 
		first_deriv_weights=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.
		second_deriv_weights=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.)
		if not self.logr:
			self.first_deriv_coeffs=first_deriv_weights/self.delta[0]
			self.second_deriv_coeffs=second_deriv_weights/self.delta[0]**2
		else: 
			self.first_deriv_coeffs=np.array([first_deriv_weights/(r*self.delta_log[0]) for r in self.radii])
			self.second_deriv_coeffs=np.array([(1./r**2)*(second_deriv_weights/(self.delta_log[0]**2)-(first_deriv_weights)/(self.delta_log[0]))\
				for r in self.radii])

		self.delta_t=0
		self.time_cur=0
		self.total_time=0
		self.time_target=0

		self.saved=np.empty([0, self.length, len(self.out_fields)])
		self.fdiff=np.empty([0, self.length-1, 2*len(self.cons_fields)+1])
		self.output_prep()
		self.time_stamps=[]

		self.nsolves=0

	def update_aux(self):
		self.rho=np.exp(self.log_rho)
		self.r2vel=self.radii**2*self.vel
		self.frho=self.radii**2*self.vel*self.rho
		# if not self.isot:
		# 	self.temp=(np.exp(self.log_rho)*np.exp(self.mu*mp*self.s/kb))**(2./3.)
		self.eos()

		self.sp_heating=(0.5*self.vel**2+0.5*self.vw**2-(self.gamma)/(self.gamma-1)*(self.pres/self.rho))
		u=self.pres/(self.rho*(self.gamma-1.))
		self.bernoulli=0.5*self.vel**2+(self.pres/self.rho)+u+self.phi_grid
		self.fen=self.rho*self.radii**2*self.vel*self.bernoulli

		self.src_rho=self.q_grid*self.radii**2
		self.src_en=self.radii**2.*self.q_grid*(self.vw**2/2.+self.phi_grid)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.src_v=-(self.q_grid*self.vel/self.rho)+(self.q_grid*self.sp_heating/(self.rho*self.vel))
		if self.isot:
			self.src_s=0.
		else:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				self.src_s=self.q_grid*self.sp_heating/(self.rho*self.vel*self.temp)

	#Equation of state. Note this could be default in the future there could be functionality to override this.
	def eos(self):
		if not self.isot:
			self.temp=(np.exp(self.log_rho)*np.exp(self.mu*mp*self.s/kb))**(2./3.)
			self.cs=np.sqrt(self.gamma*kb*self.temp/(self.mu*mp))
		else:                                                                                                     
			self.cs=np.sqrt(kb*self.temp/(self.mu*mp))
		self.alpha_max=np.max([np.abs(self.vel+self.cs), np.abs(self.vel-self.cs)])
		self.pres=self.rho*kb*self.temp/(self.mu*mp)
		
	def M_enc(self,r):
		return 0.

	def q(self, r):
		r1=2.4e17
		r2=1.2e18
		mdotw=6.7*10.**22
		eta=-2.

		a=mdotw/(4.*np.pi)/((r2**(eta+3)-r1**(eta+3))/(eta+3))
		if r*pc<r2 and r*pc>r1:
			return a*(r*pc)**(eta)*pc**3
		else:
			return 0.

	def sigma(self, r):
		return 0.

	def phi_s(self, r):
		return 0.

	def phi_bh(self, r):
		return -G*self.params['M']/r

	def phi(self,r):
		'''Total potential
		'''
		return self.phi_bh(r)+self.phi_s(r)
	
	def output_prep(self):
		bash_command('mkdir -p '+self.outdir)
		f=open(self.outdir+'/params', 'w')
		f.write(str(vars(self)))
		log=open(self.outdir+'/log', 'w')
		log.close()
		save=open(self.outdir+'/save', 'w')
		save.close()
		times=open(self.outdir+'/times', 'w')
		times.close()


	#Calculate hearting in cell
	def sp_heating(self, i):
		return (0.5*self.grid[i].vel**2+0.5*self.vw[i]**2-(self.gamma)/(self.gamma-1)*(self.grid[i].pres/self.grid[i].rho))
	

	#Update array of conseerved quantities	
	def _cons_update(self):
		#differences in fluxes and source terms
		fdiff=np.empty([2*len(self.cons_fields)+1, self.length-1])
		fdiff[0]=self.radii[1:]
		for i in range(3):
			flux=getattr(self, self.cons_fields[i])
			#get differences in fluxes for all of the cells.
			fdiff[i+1]=np.diff(flux)
			#get the source terms for all of the cells.
			fdiff[i+4]=(getattr(self, self.src_fields[i])*self.delta)[1:]
			
		self.fdiff=np.append(self.fdiff,[np.transpose(fdiff)],0)

	#Check how well conservation holds on grid as a whole.
	def _cons_check(self):
		'''Check level of conservation at end of run'''
		check=file(self.outdir+'/check', 'w')
		self.check=True
		for i in range(len(self.cons_fields)):
			flux=4.*np.pi*getattr(self, self.cons_fields[i])
			fdiff=flux[self.end]-flux[self.start]
			src=4.*np.pi*getattr(self, self.src_fields[i])*self.delta
			integral=np.sum(src[self.start:self.end+1])
			with warnings.catch_warnings():
				pdiff=(fdiff-integral)*100./integral
			if (pdiff>40.) or np.isnan(pdiff):
				self.check=False

			check.write(self.cons_fields[i]+'\n')
			pre=['flux1=','flux2=','diff=','src=','pdiff=']
			vals=[flux[self.start],flux[self.end],fdiff,integral,pdiff]
			for j in range(len(vals)):
				check.write(pre[j])
				s='{0:4.3e}'.format(vals[j])
				check.write(s+'\n')
			check.write('____________________________________\n\n')

	#Adding ghost zones onto the edges of the grid (moving the start of the grid)
	def _add_ghosts(self):
		self.start=self._num_ghosts
		self.end=self.end-self._num_ghosts
	
	#Interpolating field (using zones wiht indices i1 and i2) to radius rad 
	def _interp_zones(self, rad, i1, i2, field):
		'''Interpolate field between zone with indices i1 and i2'''
		rad1=self.radii[i1]
		rad2=self.radii[i2]

		field_arr=getattr(self, field)
		val1=field_arr[i1]
		val2=field_arr[i2]
		return np.interp(np.log(rad), [np.log(rad1), np.log(rad2)], [val1, val2])

	#Applying boundary conditions
	def _update_ghosts(self):
		'''Method to update boundaries'''
		if self.bdry=='bp':
			self._s_adjust()
			self._dens_extrapolate()
			self._mdot_adjust()
		else:
			self._extrapolate('rho')
			self._extrapolate('vel')
			self._extrapolate('s')
		self.update_aux()


	#Constant entropy across the ghost zones
	def _s_adjust(self):
		s_start=self.s[self.start]
		for i in range(0, self.start):
			self.s[i]=s_start
		s_end=self.s[self.end]
		for i in range(self.end+1, self.length):
			self.s[i]=s_end

	#Power law extrapolation of quantities into ghost zones 
	def _extrapolate(self, field):
		'''Perform power law extrapolation of quantities on grid to boundaries'''
		r1=self.radii[self.start]
		r2=self.radii[self.start+3]
		field_arr=getattr(self.grid, field)
		field1=field_arr[self.start]
		field2=field_arr[self.start+3]
		slope=np.log(field2/field1)/np.log(r2/r1)

		for i in range(0, self.start):
			val=field1*np.exp(slope*np.log(self.grid[i].rad/r1))
			field_arr[i]=val
			if field=='rho':
				self.log_rho[i]=np.log(val)
			
		#Updating the end ghost zones
		r1=self.radii[self.end]
		r2=self.radii[self.end-3]
		field1=getattr(self.grid[self.end], field)
		field2=getattr(self.grid[self.end-3], field)
		slope=np.log(field2/field1)/np.log(r2/r1)
		
		#Updating the end ghost zones, extrapolating using a power law density
		for i in range(self.end+1, self.length):
			val=field1*np.exp(slope*np.log(self.grid[i].rad/r1))
			field_arr[i]=val
			if field=='rho':
				self.log_rho[i]=np.log(val)

	#Extrapolate densities to the ghost zones; a lot of this method is redundant 
	def _dens_extrapolate(self):
		'''Extrapolate $\rho$ assuming that the $\rho\sim r^{-3/2}$ on inner boundary and $\rho\sim r^{-3/2}$'''
		r_start=self.radii[self.start]
		r_start2=self.radii[self.start+3]
		log_rho_start=self.log_rho[self.start]
		log_rho_start2=self.log_rho[self.start+3]

		#If the inner bdry is fixed...(appropriate for Parker wind)
		if self.bdry_fixed:
			for i in range(1, self.start):
				self.log_rho[i]=self._interp_zones(self.radii[i], 0, self.start, 'log_rho')
				self.rho[i]=np.exp(self.log_rho[i])
		#Updating the starting ghost zones, extrapolating using rho prop r^-3/2
		else:
			for i in range(0, self.start):
				slope=-3./2.
				#slope=(log_rho_start2-log_rho_start)/np.log(r_start2/r_start)
				log_rho=slope*np.log(self.radii[i]/r_start)+log_rho_start
				self.log_rho[i]=log_rho
				self.rho[i]=np.exp(log_rho)
		#Updating the end ghost zones
		r_end=self.radii[self.end]
		log_rho_end=self.log_rho[self.end]
		r_end2=self.radii[self.end-3]
		log_rho_end2=self.log_rho[self.end-3]
		#Updating the end ghost zones, extrapolating using a power law density
		for i in range(self.end+1, self.length):
			slope=-2.
			#slope=(log_rho_end-log_rho_end2)/np.log(r_end/r_end2)
			log_rho=slope*np.log(self.radii[i]/r_end)+log_rho_end
			self.log_rho[i]=log_rho
			self.rho[i]=np.exp(log_rho)

	#Enforce constant mdot across the boundaries (bondary condition for velocity)
	def _mdot_adjust(self):
		#Start zones 
		frho=self.frho[self.start]
		for i in range(0, self.start):
			vel=frho/self.rho[i]/self.radii[i]**2
			self.vel[i]=vel
		#End zones
		frho=self.frho[self.end]
		for i in range(self.end+1, self.length):
			vel=frho/self.rho[i]/self.radii[i]**2
			self.vel[i]=vel
		self.update_aux()


	#Evaluate terms of the form div(kappa*df/dr)-->Diffusion like terms (useful for something like the conductivity)
	# def get_diffusion(self, i, coeff, field):
	# 	dkappa_dr=self.get_spatial_deriv(i, coeff)
	# 	dfield_dr=self.get_spatial_deriv(i, field)
	# 	d2field_dr2=self.get_spatial_deriv(i, field, second=True)

	# 	if hasattr(coeff, '__call__'):
	# 		kappa=coeff(self.grid[i])
	# 	else:	
	# 		kappa=getattr(self.grid[i], coeff)

	# 	return kappa*d2field_dr2+dkappa_dr*dfield_dr+(2./self.radii[i])*(kappa*dfield_dr)


	#Getting derivatives for a given field (density, velocity, etc.). If second is set to be true then the discretized 2nd
	#deriv is evaluated instead of the first
	def get_spatial_deriv(self, i, field, second=False):
		field_list=getattr(self,field)[i-3:i+4]

		if second:
			return np.sum(field_list*self.second_deriv_coeffs[i])		
		else:
			return np.sum(field_list*self.first_deriv_coeffs[i])


	#Calculate laplacian in spherical coords. 
	def get_laplacian(self, i, field):
		return self.get_spatial_deriv(i, field, second=True)+(2./self.radii[i])*(self.get_spatial_deriv(i, field))

	#Evaluate Courant condition for the entire grid. This gives us an upper bound on the time step we may take 
	def _cfl(self):
		# alpha_max=0.
		delta_t=np.zeros(self.length)
		delta_t=self.safety*self.delta/self.alpha_max

		#Setting the time step
		cfl_delta_t=np.min(delta_t)
		target_delta_t=self.time_target-self.time_cur
		self.delta_t=min([target_delta_t, cfl_delta_t])

	#Wrapper for evaluating the time derivative of all fields
	def dfield_dt(self, i, field):
		if field=='rho':
			return self.drho_dt(i)
		elif field=='vel':
			return self.dvel_dt(i)
		elif field=='log_rho':
			return self.dlog_rho_dt(i)
		elif field=='s':
			return self.ds_dt(i)
		else:
			return 0.

	#Partial derivative of density with respect to time
	def dlog_rho_dt(self, i):
		rad=self.radii[i]
		rho=self.rho[i]
		vel=self.vel[i]

		#return -cs*drho_dr+art_visc*drho_dr_second
		return -vel*self.get_spatial_deriv(i, 'log_rho')-(1/rad**2)*self.get_spatial_deriv(i, 'r2vel')+self.q_grid[i]/rho

	#Partial derivative of density with respect to time
	def drho_dt(self, i):
		rad=self.radii[i]

		#return -cs*drho_dr+art_visc*drho_dr_second
		return -(1./rad)**2*self.get_spatial_deriv(i, 'frho')+self.q_grid[i]

	#Evaluating the partial derivative of velocity with respect to time
	def dvel_dt(self, i):
		rad=self.radii[i]
		vel=self.vel[i]
		rho=self.rho[i]
		temp=self.temp[i]

		#If the density zero of goes negative return zero to avoid numerical issues
		if rho<=self.floor:
			return 0

		# dpres_dr=self.get_spatial_deriv(i, 'pres')
		dlog_rho_dr=self.get_spatial_deriv(i, 'log_rho')
		dtemp_dr=self.get_spatial_deriv(i, 'temp')
		dv_dr=self.get_spatial_deriv(i, 'vel')
		d2v_dr2=self.get_spatial_deriv(i, 'vel', second=True)
		drho_dr=dlog_rho_dr*(rho)

		lap_vel=self.get_laplacian(i, 'vel')
		art_visc=min(self.cs[i],  np.abs(self.vel[i]))*(self.radii[self.end]-self.radii[self.start])*lap_vel/self.Re
		if self.visc_scheme=='const_visc':
			pass
		elif self.visc_scheme=='cap_visc':
			art_visc=art_visc*min(1., (self.delta[i]/np.mean(self.delta)))
		else:
			art_visc=art_visc*(self.delta[i]/np.mean(self.delta))

		return -vel*dv_dr-dlog_rho_dr*kb*temp/(self.mu*mp)-(kb/(self.mu*mp))*dtemp_dr-self.grad_phi_grid[i]+art_visc-(self.q_grid[i]*vel/rho)


	#Evaluating the partial derivative of entropy with respect to time
	def ds_dt(self, i):
		rho=self.rho[i]
		temp=self.temp[i]
		vel=self.vel[i]
		rad=self.radii[i]
		cs=self.cs[i]
		ds_dr=self.get_spatial_deriv(i, 's')
		lap_s=self.get_laplacian(i, 's')
		art_visc=min(self.cs[i],  np.abs(self.vel[i]))*(self.radii[self.end]-self.radii[self.start])*lap_s/self.Re_s
		if self.visc_scheme=='const_visc':
			pass
		elif self.visc_scheme=='cap_visc':
			art_visc=art_visc*min(1., (self.delta[i]/np.mean(self.delta)))
		else:
			art_visc=art_visc*(self.delta[i]/np.mean(self.delta))


		#return self.q(rad, **self.params_delta)*(0.5*self.vw**2+0.5*vel**2-self.gamma*cs**2/(self.gamma-1))/(rho*temp)-vel*def s_dr#+art_visc*lap_s
		return self.q_grid[i]*self.sp_heating[i]/(rho*temp)-vel*ds_dr+art_visc


	def isot_off(self):
		'''Switch off isothermal evolution'''
		self.isot=False
		self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))
		self.eos()
		self.fields=['log_rho', 'vel', 's']
		# log.write('isot off time:'+str(self.total_time)+'\n')
		# for zone in self.grid:
		# 	zone.isot=False
		# 	zone.entropy()

	def isot_on(self):
		'''Switch on isothermal evolution'''
		self.isot=True
		self.eos()
		self.fields=['log_rho', 'vel']

		log=open(self.outdir+'/log', 'a')
		log.write('isot on time:'+str(self.total_time)+'\n')
		log.close()
		# for zone in self.grid:
		# 	zone.isot=True

	#Set all mass dependent quantities
	def place_mass(self):
		self.M_bh=self.params['M']
		self.M_enc_grid=np.array(map(self.M_enc, self.radii/pc))

		self.M_tot=self.M_enc_grid+self.M_bh
		#Potential and potential gradient (Note that we have to be careful of the units here.)
		self.phi_s_grid=np.array(map(self.phi_s, self.radii/pc))/pc
		self.phi_bh_grid=np.array(map(self.phi_bh, self.radii/pc))/pc
		self.phi_grid=self.eps*self.phi_s_grid+self.phi_bh_grid
		self.grad_phi_grid=G*(self.M_bh+self.eps*self.M_enc_grid)/self.radii**2

	def _solve_prep(self):
		self.q_grid=np.array([self.q(r) for r in self.radii/pc])/pc**3
		self.vw=np.array([(self.sigma(r/pc)**2+(self.vw_extra)**2)**0.5 for r in self.radii])
		self.place_mass()
		self.update_aux()

	def solve(self, time, max_steps=np.inf):
		'''Controller for solution. Several possible stop conditions (max_steps is reached, time is reached)

		:param max_steps: Maximum number of time steps to take in the solution
		'''
		try:
			self.q_grid
		except:
			print 'Running _solve_prep'
			self._solve_prep()
			
		self.time_cur=0
		self.time_target=time
		if not os.path.isfile(self.outdir+'/params') or self.nsolves==0:
			self.output_prep()
		#For purposes of easy backup
		if len(self.saved)==0:
			self.save_pt=0
		else:
			self.save_pt=len(self.saved)-1

		self._evolve(max_steps=max_steps)

		self.write_sol()
		self.nsolves+=1

	def set_param(self, param, value):
		'''Reset parameter

		:param param: parameter to reset
		:param value: value to reset it to
		'''

		old=getattr(self,param)
		if param=='eps':
			self.eps=value
			try:
				self.phi=self.eps*self.phi_s_grid+self.phi_bh_grid
				self.grad_phi_grid=G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii**2
			except AttributeError:
				pass
		elif param=='vw_extra':
			self.vw_extra=value
			self.vw=np.array([(self.sigma(r/pc)**2+(self.vw_extra)**2)**0.5 for r in self.radii])
		elif param=='gamma' or param=='mu':
			setattr(self,param,value)
			self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))
			self.eos()
		elif param=='outdir':
			self.outdir=value
			bash_command('mkdir -p '+value)
			bash_command('mv '+old+'/log '+old+'/cons '+old+'/save '+value)
		elif param=='isot':
			print 'Warning! Changing the isothermal flag is done through the isot_on and isot_off methods!'
		else:
			setattr(self,param,value)

		log=open(self.outdir+'/log', 'a')
		new=getattr(self, param)
		log.write(param+' old:'+str(old)+' new:'+str(value)+' time:'+str(self.total_time)+'\n')
		log.close()

	#Gradually perturb a given parameter (param) to go to the desired value (target). 
	def solve_adjust(self, time, param, target, n=10, max_steps=np.inf):
		'''Run solver for time adjusting the value of params to target in the process

		:param str param: parameter to adjust
		:param target: value to which we would like to adjust the parameter
		:param int n: Number of time intervals to divide time into for the purposes of parameter adjustment
		:param int max_steps: Maximum number of steps for solver to take
		'''
		try:
			self.q_grid
		except:
			self._solve_prep()
			
		if len(self.saved==0):
			self.save_pt=0
		else:
			self.save_pt=len(self.saved)-1
		self.time_cur=0
		param_cur=getattr(self, param)
	
		interval=time/float(n)
		self.time_target=interval
		delta_param=(target-param_cur)/float(n)
		while not np.allclose(param_cur, target):
			self._evolve(max_steps=max_steps)
			param_cur+=delta_param
			self.time_target+=interval
			self.set_param(param,param_cur)
		self.write_sol()
		self.nsolve+=1

			
	#Method to write solution info to file
	def write_sol(self):
		self._cons_check()

		np.savez(self.outdir+'/save', a=self.saved, b=self.time_stamps)
		np.savez(self.outdir+'/cons', a=self.fdiff)

	def backup(self):
		bash_command('mkdir -p '+self.outdir)
		pickle.dump(self, open( self.outdir+'/grid.p', 'wb' ) )

	#Lower level evolution method
	def _evolve(self, max_steps=np.inf):
		#Initialize the number of steps and the progress
		num_steps=0
		pbar=progress.ProgressBar(maxval=self.time_target, fd=sys.stdout).start()
		ninterval=0

		#While we have not yet reached the target time
		while self.time_cur<self.time_target:
			if (self.tinterval>0 and (self.time_cur/self.tinterval)>=ninterval) or (self.tinterval<=0 and num_steps%self.sinterval==0):
				pbar.update(self.time_cur)
				self._cons_update()
				self.save()
				ninterval=ninterval+1

				# if len(self.saved)>2:
				# 	max_change=np.max(np.abs((self.saved[-1]-self.saved[-2])/self.saved[-2]))
				# 	self.max_change.append(max_change)
				# 	# if max_change<self.tol:
				# 	# 	break
				# #np.savetxt(fout, self.saved[-1])
				# #print self.total_time/self.time_target

			#Take step and increment current time
			self._step()
			#Increment the time and steps variables.
			self.time_cur+=self.delta_t
			self.total_time+=self.delta_t
			num_steps+=1
			#If we have exceeded the max number of allowed steps then break
			if num_steps>max_steps:
				print "exceeded max number of allowed steps"
				break
		pbar.finish()
		print

	# #Reverts grid to earlier state. Previous solution
	# def revert(self, index=None):
	# 	if index==None:
	# 		index=self.save_pt
	# 	for i in range(len(self.grid)):
	# 		self.grid[i].log_rho=np.log(self.saved[index,i,1])
	# 		self.grid[i].vel=self.saved[index,i,2]*self.saved[index,i,-1]
	# 		self.grid[i].temp=self.saved[index,i,3]
	# 		if not self.isot:
	# 			self.grid[i].entropy()
	# 		self.grid[i].update_aux()
	# 	self.saved=self.saved[:index]
	# 	self.time_stamps=self.time_stamps[:index]
	# 	self.fdiff=self.fdiff[:index]
	# 	#Overwriting previous saved files
	# 	np.savetxt(self.outdir+'/save', self.saved.reshape((-1, len(self.out_fields))))
	# 	np.savetxt(self.outdir+'/cons', self.fdiff.reshape((-1, 7)))
	# 	np.savetxt(self.outdir+'/times',self.time_stamps)

	#Create movie of solution
	def animate(self,  analytic_func=None, index=1):
		if analytic_func:
			vec_analytic_func=np.vectorize(analytic_funcs)
		def update_img(n):
			time=self.time_stamps[n]
			sol.set_ydata(self.saved[n*50,:,index])
			label.set_text(str(time))

		#Setting up for plotting
		ymin=np.min(self.saved[:,:,index])
		ymax=np.max(self.saved[:,:,index])
		fig,ax=plt.subplots()
		label=ax.text(0.02, 0.95, '', transform=ax.transAxes)	
		ax.set_xscale('log')
		if index==0:
			ax.set_yscale('log')
		ax.set_ylim(ymin-0.1*np.abs(ymin), ymax+0.1*np.abs(ymax))

		#Plot solution/initial condition/(maybe) analytic solution
		sol,=ax.plot(self.radii, self.saved[0,:,index], self.symbol)
		ax.plot(self.radii, self.saved[0,:,index], 'b')
		if analytic_func:
			analytic_sol,=ax.plot(self.radii, vec_analytic_func(self.radii))
		
		#Exporting animation
		sol_ani=animation.FuncAnimation(fig,update_img,len(self.saved)/50,interval=50, blit=True)
		sol_ani.save('sol_'+self.out_fields[index]+'.mp4', dpi=100)
		plt.clf()

	#Save the state of the grid
	def save(self):
		grid_prims=[getattr(self, field) for field in self.out_fields]
		grid_prims[2]=grid_prims[2]/grid_prims[-1]

		#Saving the state of the grid within list
		#self.saved.append((self.total_time, np.transpose(grid_prims)))
		self.saved=np.append(self.saved,[np.transpose(grid_prims)],0)
		self.time_stamps.append(self.total_time)
		#Dump to file
		save_handle=file(self.outdir+'/save','a')
		np.savetxt(save_handle, self.saved[-1])
		cons_handle=file(self.outdir+'/cons','a')
		np.savetxt(cons_handle, self.fdiff[-1])
		t_handle=file(self.outdir+'/times', 'a')
		np.savetxt(t_handle, [self.time_stamps[-1]])

	#Clear all of the info in the saved list
	def clear_saved(self):
		self.saved=np.empty([0, self.length, len(self.out_fields)])
		self.fdiff=np.empty([0, self.length-1, 7])
		self.time_stamps=[]
		self.save_pt=0
		self.total_time=0

	#Take a single step in time
	def _step(self):
		gamma=[8./15., 5./12., 3./4.]
		zeta=[-17./60.,-5./12.,0.]

		self._cfl()

		for substep in range(3):
			self._sub_step(gamma[substep], zeta[substep])
			self.update_aux()
			self._update_ghosts()

	#Substeps
	def _sub_step(self, gamma, zeta):
		#Calculating the derivatives of the field
		for field in self.fields:
			for i in range(self.start,self.end+1):
				self.time_derivs[i][field]=self.dfield_dt(i, field)

		#Updating the values in the grid: have to calculate the time derivatives for all relevant fields before this step
		for field in self.fields:
			field_arr=getattr(self,field)
			for i in range(self.start, self.end+1):
				f=field_arr[i]+gamma*self.time_derivs[i][field]*self.delta_t
				g=f+zeta*self.time_derivs[i][field]*self.delta_t
				field_arr[i]=g

	#Extracting the array corresponding to a particular field from the grid as well as an array of radii
	def get_field(self, field):
		'''Extract field list at all grid points

		:param str field: Field to extract
		:return: list containing two numpy arrays: one containing list of radii and the other containing the field list.
		'''
		# field_list=np.zeros(self.length)
		# rad_list=np.zeros(self.length)
		# for i in range(0, self.length):
		# 	field_list[i]=getattr(self.grid[i], field)
		# 	rad_list[i]=getattr(self.grid[i],'rad')

		return [self.radii, getattr(self,field)]

	@property
	def rs(self):
		'''Find stagnation point in the flow'''
		guess=self.radii[0]*1.1
		self.interp_vel=interp1d(self.radii, self.vel)
		try:
			return fsolve(self.interp_vel, guess)[0]
		except:
			return None

	#Get accretion rate  for galaxy by integrating source from stagnation radius
	@property
	def mdot(self):
		'''Mass accretion rate based on stagnation radius 
		'''
		rs_pc=self.rs/pc
		mdot=4.*np.pi*integrate.quad(lambda r: r**2*self.q(r), self.rmin_star, rs_pc)[0]
		return mdot

	@property
	def eddr(self, eta=0.1):
		'''Compute the Eddington ratio

		:param float eta: The assumed radiarive efficiency
		'''
		l_edd=4.*np.pi*G*self.params['M']*c/(0.4)
		mdot_edd=l_edd/(eta*c**2)

		return self.mdot/mdot_edd

	@property	
	def cooling(self):
		'''Cooling luminosity'''
		lambdas=np.array([lambda_c(temp) for temp in self.temp])
		return lambdas*(self.rho/(self.mu*mp))**2

	@property
	def x_ray_lum(self):
		'''Integrated x-ray luminosity at each grid radius'''
		return np.cumsum(4.*np.pi*self.cooling*self.delta*self.radii**2)

class NukerGalaxy(Galaxy):
	'''Sub-classing galaxy above to represent Nuker parameterized galaxies'''
	def __init__(self, gname, gdata,  init=None, init_array=None):
		try:
			self.params=gdata[gname]
			self.params_table=Table([self.params])
			# self.params_table['M'].format='3.2E'
		except KeyError:
			print 'Error! '+gname+' is not in catalog!'
			raise
		names=['Name', 'Type','M', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$I_b$', r'$r_b$', 'Uv']
		self.params_table=Table(self.params_table['Name', 'type', 'M', 'alpha', 'beta', 'gamma', 'Ib', 'rb', 'Uv'], names=names)
		self.params_table['M'].format='{0:3.2e}'
		self.params_table[r'$I_b$'].format='{0:3.2}'
		self.params_table[r'$r_b$'].format='{0:3.2}'
		self.params_table['M']=self.params_table['M']/M_sun
		self.params_table['M'].unit=u.MsolMass

		self.name=gname
		print self.name
		self.eta=1.
		self.rmin_star=1.E-3
		self.rmax_star=1.E5
		self.rg=G*self.params['M']/c**2

		Galaxy.__init__(self, init=init, init_array=init_array)

	def rho_stars(self,r):
		'''Stellar density
		:param r: radius in parsecs
		'''
		if r<self.rmin_star:
			return 0.
		else:
			return M_sun*self.params['Uv']*inverse_abel(nuker_prime, r, **self.params)

	def M_enc(self,r):
		'''Mass enclosed within radius r
		:param r: radius in parsecs
		'''
		if r<self.rmin_star:
			return 0.
		# elif self.menc=='circ':
		#     return integrate.quad(lambda r1:2.*np.pi*r1*M_sun*self.params['Uv']*nuker(r1, **self.params),self.rmin)
		else:
			return integrate.quad(lambda r1:4.*np.pi*r1**2*self.rho_stars(r1), self.rmin_star, r)[0]

	def sigma(self, r):
		'''Velocity dispersion of galaxy
		:param r: radius in parsecs
		'''
		return (c**2*self.rg/pc/r*(self.M_enc(r)+self.params['M'])/self.params['M'])**0.5

	def phi_s(self,r):
		'''Potential from the stars
		:param r: radius in parsecs
		'''
		return (-G*self.M_enc(r)/r)-4.*np.pi*G*integrate.quad(lambda r1:self.rho_stars(r1)*r1, r, self.rmax_star)[0]

	def phi_bh(self,r):
		'''Potential from the black hole
		:param r: radius in parsecs
		'''
		return -G*self.params['M']/r

	def q(self, r):
		'''Source term representing mass loss from stellar winds'''
		return self.eta*self.rho_stars(r)/th


	##Getting the radius of influence: where the enclosed mass begins to equal the mass of the central BH. 
	@property 
	def rinf(self):
		'''Get radius of influence for galaxy'''
		def mdiff(r):
			return self.params['M']-self.M_enc(r)

		return fsolve(mdiff, 1)[0]

	@property
	def rb(self):
		'''Nominal bondi radius--computed using black hole mass and tempearture on the edge of the grid'''
		try:
			rb=G*self.params['M']/self.cs[-1]**2
		except AttributeError:
			self.eos()
			rb=G*self.params['M']/self.cs[-1]**2

		return rb

	@property
	def tde_table(self):
		'''Get crossing radius for jet'''
		m6=self.params['M']/(1.E6*M_sun)
		jet=tde_jet.Jet()
		jet.m6=m6

		self.rho_interp=interp1d(self.radii, self.rho)
		r=integrate.ode(jet.vj)

		r.set_integrator('vode')
		r.set_initial_value(0., t=jet.delta).set_f_params(self.rho_interp)
		try:
			while r.y<r.t:
				r.integrate(r.t+0.01*pc)
			rc=r.y[0]
		except Exception as inst:
			print inst
			rc=np.nan

		f=jet.rho(rc)/self.rho_interp(rc)
		gamma=jet.gamma_j*(1.+2.*jet.gamma_j*f**(-0.5))**(-0.5)   

		rc_col=Column([rc/pc], name=r'$r_c$', format='{0:3.2}')
		n_rc_col=Column([self.rho_interp(rc)/(self.mu*mp)], name=r'$n_{rc}$',format='{0:3.2}')
		gamma_rc_col=Column([gamma], name=r'$\Gamma_{rc}$',format='{0:3.2}')
		tde_table=Table([rc_col, n_rc_col, gamma_rc_col])

		return tde_table


	@property 
	def summary(self):
		'''Summary of galaxy properties'''
		return table.hstack([self.params_table, self.tde_table, Table([[self.rs/pc]], names=['$r_{s}$']), Table([[self.eddr]],\
			names=[r'$\dot{M}/\dot{M_{\rm edd}}$'])])


		
































