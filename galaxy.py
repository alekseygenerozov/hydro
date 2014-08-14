import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.misc import derivative

import copy
import warnings
import pickle
import dill
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

from latex_exp import latex_exp

import tde_jet
import gal_properties

import progressbar as progress
import ipyani

import os.path
import re
import functools

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


def pow_extrap(r, r1, r2, field1, field2):
	'''Power law extrapolation given radius of interest, and input radii and fields'''
	slope=np.log(field2/field1)/np.log(r2/r1)

	return field1*np.exp(slope*np.log(r/r1))

def extrap1d(interpolator):
	'''Modify interpolation function to allow for extrapolation--constant extrapolation beyond bondaries'''
	xs = interpolator.x
	ys = interpolator.y

	def pointwise(x):
		if x < xs[0]:
			return ys[0]
		elif x > xs[-1]:
			return ys[-1]
		else:
			return interpolator(x)

	def ufunclike(xs):
		try:
			return np.array(map(pointwise, xs))
		except TypeError:
			return pointwise(xs)

	return ufunclike

def extrap1d_pow(interpolator):
	'''Modify interpolation function to allow for extrapolation--power law extrapolation beyond boundaries'''
	xs = interpolator.x
	ys = interpolator.y

	#This will not work if we have less than 2 interpolation x values. Also, this is somewhat different from the extrapolation used for updating 
	#the boundaries of the grid, where I go three zones on either side of the edge non-ghost zones in order to compute the power law slope.
	def pointwise(x):
		if x < xs[0]:
			return pow_extrap(x, xs[0], xs[1], ys[0], ys[1])
		elif x > xs[-1]:
			return pow_extrap(x, xs[-1], xs[-2], ys[-1], ys[-2])
		else:
			return interpolator(x)

	def ufunclike(xs):
		try:
			return np.array(map(pointwise, xs))
		except TypeError:
			return pointwise(xs)

	return ufunclike

class memoize(object):
	'''Memoization intended for class methods.'''
	#Exclude first argument which is just info on the object instance, and is not persistent through pickling. 
	def __init__(self, func):
	#print "Init"
		self.func = func

	def __call__(self, *args):
	#print "Call"
		name=self.func.__name__
		if not name in self.cache:
			self.cache[name] = {}
		try:
			return self.cache[name][args[1:]]
		except KeyError:
			value = self.func(*args)
			self.cache[name][args[1:]] = value
			return value
		except TypeError:
			# uncachable -- for instance, passing a list as an argument.
			# Better to not cache than to blow up entirely.
			return self.func(*args)

	def __repr__(self):
		"""Return the function's docstring."""
		return self.func.__doc__

	def __get__(self, obj, objtype):
		"""Support instance methods."""
		#print "Get", obj, objtype
		fn = functools.partial(self.__call__, obj)
		try:
			self.cache = obj.cache
		except:
			obj.cache = {}
			self.cache = obj.cache
		#print self.cache
		return fn

def lazyprop(fn):
	attr_name = '_lazy_' + fn.__name__
	@property
	def _lazyprop(self):
		if not hasattr(self, attr_name):
			setattr(self, attr_name, fn(self))
		return getattr(self, attr_name)
	return _lazyprop

class StepsError(Exception):
	pass
	
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
			d['type']='Core'
		else:
			d['type']='Cusp'
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
	end_state[:,2]=end_state[:,2]*end_state[:,7]
	#end_state[:,1]=np.log(end_state[:,1])
	start=end_state[:,:4]
	#rescaling radial grid. This will be useful for going from one mass black hole to another. 
	start[:,0]=start[:,0]*rescale

	return start

def background(rad, rho_0=2.E-31, temp=1.E6, r0=5.E11, n=0.):
	rho_0=rho_0*(rad/r0)**n
	return np.array([rho_0, 0., temp])
	

class Galaxy(object):
	'''Class to representing galaxy--Corresponds to Quataert 2004. Can be initialized either using an analytic function or a
	numpy ndarray object

	:param init: list contain minimum radius of the grid [cm], maximum radius of the grid [cm], and function to evaluate 
		initial conditions
	:param init_dir: name of directory containing previous from which the current run should be initialized
	'''

	def __init__(self, init={}):
		self.params={'M':3.6E6*M_sun}
		self.isot=False
		self.gamma=5./3.
		self.fields=['log_rho', 'vel', 's']
		self.mu=1.

		init_def={'rmin':0.1*pc,'rmax':10.*pc,'f_initial':background, 'length':70, 'func_params':{}, 'logr':True}
		for key in init_def.keys():
			try:
				init[key]
			except KeyError:
				init[key]=init_def[key]
		self.init=init
		# self.init_array=init_array
		self._init_grid()

		#Attributes to store length of the list as well as start and end indices (useful for ghost zones)
		self.start=0
		self.end=self.length-1
		self.tcross=(self.radii[-1]-self.radii[0])/(kb*1.E7/mp)**0.5
		self._num_ghosts=3
		self.tinterval=0.05*self.tcross
		self.sinterval=100

		self.visc_scheme='default'
		self.Re=90.
		self.Re_s=1.E20
		self.floor=0.
		self.safety=0.6
		self.bdry='default'
		self.bdry_fixed=False
	
		self.vw_extra=1.E8
		self.sigma_heating=True
		self.eps=1.

		self.out_fields=['radii', 'rho', 'vel', 'temp', 'frho', 'bernoulli', 's', 'cs', 'q_grid', 'M_enc_grid', 'phi_grid', 'sigma_grid','vw']
		self.cons_fields=['frho', 'bernoulli', 's', 'fen']
		self.src_fields=['src_rho', 'src_v', 'src_s', 'src_en']
		self.movies=False
		self.outdir='.'

		#Will store values of time derivatives at each time step
		self.time_derivs=np.zeros(self.length, dtype={'names':['log_rho', 'vel', 's'], 'formats':['float64', 'float64', 'float64']})
		#Initializing the grid using the initial value function f_initial


		self._add_ghosts()
		#Coefficients use to calculate the derivatives 
		first_deriv_weights=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.
		second_deriv_weights=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.)
		if not self._logr:
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
		self.tmax=20.*self.tcross

		self.saved=np.empty([0, self.length, len(self.out_fields)])
		self.fdiff=np.empty([0, self.length-1, 2*len(self.cons_fields)+1])
		self.non_standard={}
		self.time_stamps=[]

		self.log=''
		self.nsolves=0

	@classmethod
	def from_dir(cls, loc, index=-1, rescale=1, length=None):
		init={}

		init_array=prepare_start(np.load(loc+'/save.npz')['a'][index])
		init_array[:,0]=rescale*init_array[:,0]
		radii=init_array[:,0]
		
		init['rmin']=radii[0]
		init['rmax']=radii[-1]

		delta=np.diff(radii)
		delta_log=np.diff(np.log(radii))

		if np.allclose(np.diff(delta),[0.]):
			init['logr']=False
		else:
			init['logr']=True
		init['f_initial']=extrap1d(interp1d(init_array[:,0], init_array[:,1:4], axis=0))
		if not length:
			init['length']=len(radii)
		else:
			init['length']=length

		return cls(init=init)


	def _init_grid(self):
		#Initializing the radial grid
		rmin,rmax,f_initial=self.init['rmin'],self.init['rmax'],self.init['f_initial']
		self.func_params=self.init['func_params']
		self._logr=self.init['logr']

		assert rmax>rmin
		assert hasattr(f_initial, '__call__')
		self.length=self.init['length']

		if self._logr:
			self.radii=np.logspace(np.log(rmin), np.log(rmax), self.length, base=e)
		else:
			self.radii=np.linspace(rmin, rmax, self.length)
		prims=[f_initial(r, **self.func_params) for r in self.radii]
		prims=np.array(prims)

		delta=np.diff(self.radii)
		self.delta=np.insert(delta, 0, delta[0])
		delta_log=np.diff(np.log(self.radii))
		self.delta_log=np.insert(delta_log, 0, delta_log[0])

		self.log_rho=np.log(prims[:,0])
		self.vel=prims[:,1]
		self.temp=prims[:,2]
		self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))

	def M_enc(self,r):
		return 0.

	def q(self, r):
		r1=2.4e17
		r2=1.2e18
		mdotw=6.7*10.**22
		eta=-2.

		a=mdotw/(4.*np.pi)/((r2**(eta+3)-r1**(eta+3))/(eta+3))
		if r<r2 and r>r1:
			return a*(r)**(eta)
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
		return self.phi_bh(r)+self.eps*self.phi_s(r)

	@property
	def q_grid(self):
		return np.array([self.q(r) for r in self.radii])

	@property
	def M_enc_grid(self):
		return np.array(map(self.M_enc, self.radii))

	@property
	def sigma_grid(self):
		return np.array([self.sigma(r) for r in self.radii])

	@property
	def phi_s_grid(self):
		return np.array(map(self.phi_s, self.radii))

	@property
	def  phi_bh_grid(self): 
		return np.array(map(self.phi_bh, self.radii))

	@property
	def phi_grid(self):
		return self.eps*self.phi_s_grid+self.phi_bh_grid

	@property
	def grad_phi_grid(self):
		return G*(self.params['M']+self.eps*self.M_enc_grid)/self.radii**2

	def vw(self, r):
		if not self.sigma_heating:
			return self.vw_extra
		else:
			return (self.sigma(r)**2+(self.vw_extra)**2)**0.5

	@property 
	def vw_grid(self):
		[self.vw(r) for r in self.radii]
		
	@property
	def rho(self):
		return np.exp(self.log_rho)

	def rho_interp(self, r):
		return interp1d(self.radii, self.rho)(r)

	def vel_interp(self, r):
		return interp1d(self.radii, self.vel)(r)

	def temp_interp(self, r):
		return interp1d(self.radii, self.temp)(r)

	def pres_interp(self,r):
		return interp1d(self.radii, self.pres)(r)

	def cs_interp(self, r):
		return interp1d(self.radii, self.cs)(r)

	def M_enc_interp(self, r):
		return interp1d(self.radii, self.M_enc_grid)(r)

	def phi_interp(self, r):
		return interp1d(self.radii, self.phi_grid)(r)

	def sigma_interp(self,r):
		return interp1d(self.radii, self.sigma_grid)(r)

	def q_interp(self, r):
		return interp1d(self.radii, self.q_grid)(r)

	@property 
	def r2vel(self):
		return self.radii**2*self.vel

	@property 
	def frho(self):
		return self.radii**2*self.vel*self.rho

	@property
	def pres(self):
		return (kb*self.temp*self.rho)/(self.mu*mp)

	@property
	def cs(self):
		if not self.isot:
			return np.sqrt(self.gamma*kb*self.temp/(self.mu*mp))
		else:                                                                                                     
			return np.sqrt(kb*self.temp/(self.mu*mp))

	@property 
	def alpha_max(self):
		return np.max([np.abs(self.vel+self.cs), np.abs(self.vel-self.cs)],axis=0)

	def sp_heating(self, r):
		return (0.5*self.vel_interp(r)**2+0.5*self.vw(r)**2-(self.gamma)/(self.gamma-1)*(self.pres_interp(r)/self.rho_interp(r)))

	@property 
	def sp_heating_grid(self):
		return [self.sp_heating(r) for r in self.radii]

	@property 
	def u(self):
		return self.pres/(self.rho*(self.gamma-1.))

	@property 
	def bernoulli(self):
		return 0.5*self.vel**2+(self.pres/self.rho)+self.u+self.phi_grid

	@property 
	def fen(self):
		return self.rho*self.radii**2*self.vel*self.bernoulli

	def src_rho(self, r):
		return self.q_interp(r)*r**2

	@property 
	def src_rho_grid(self):
		return [self.src_rho(r) for r in self.radii]

	def src_en(self, r):
		return r**2.*self.q_interp(r)*(0.5*(self.sigma_interp(r)**2.+self.vw_extra**2.)+self.phi_interp(r))

	@property 
	def src_en_grid(self):
		return [self.src_en(r) for r in self.radii]

	def src_v(self, r):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return -(self.q_interp(r)*self.vel_interp(r)/self.rho_interp(r))+(self.q_interp(r)*self.sp_heating(r)/(self.rho_interp(r)*self.vel_interp(r)))

	@property
	def src_v_grid(self):
		return [self.src_v(r) for r in self.radii]

	def src_s(self, r):
		if self.isot:
			src_s=0.
		else:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				src_s=self.q_interp(r)*self.sp_heating(r)/(self.rho_interp(r)*self.vel_interp(r)*self.temp_interp(r))
		return src_s

	@property
	def src_s_grid(self):
		return [self.src_s(r) for r in self.radii]

	def _update_temp(self):
		self.temp=(np.exp(self.log_rho)*np.exp(self.mu*mp*self.s/kb))**(2./3.)

	#Update array of conseerved quantities	
	def _cons_update(self):
		#differences in fluxes and source terms
		fdiff=np.empty([2*len(self.cons_fields)+1, self.length-1])
		fdiff[0]=self.radii[1:]
		for i in range(len(self.cons_fields)+1):
			flux=getattr(self, self.cons_fields[i])
			#get differences in fluxes for all of the cells.
			fdiff[i+1]=np.diff(flux)
			#get the source terms for all of the cells.
			fdiff[i+4]=(getattr(self, self.src_fields[i]+'_grid')*self.delta)[1:]
			
		self.fdiff=np.append(self.fdiff,[np.transpose(fdiff)],0)

	@property
	def rs_outside(self):
		return (np.where(self.radii>self.rs))[0][0]

	@property
	def rs_inside(self):
		return (np.where(self.radii<self.rs))[0][-1]

	def src_integral_outside(self, src_field):
		if src_field not in self.src_fields:
			print 'Not a source field'
			return 
		src=getattr(self, src_field)
		return 4.*np.pi*integrate.quad(lambda r:src(r), self.radii[self.rs_outside], self.radii[self.end])[0]

	def src_integral_inside(self, src_field):
		if src_field not in self.src_fields:
			print 'Not a source field'
			return 
		src=getattr(self, src_field)
		return 4.*np.pi*integrate.quad(lambda r:src(r), self.radii[self.start], self.radii[self.rs_inside])[0]

	#Check how well conservation holds on grid as a whole.
	def cons_check(self, write=True, tol=40., skip=True):
		'''Check level of conservation'''
		self.check=True
		self.check_partial=True

		check_str=''
		#self._cons_update()
		for i in range(len(self.cons_fields)):
			if (i==1 or i==2) and skip:
				continue
			flux=4.*np.pi*getattr(self, self.cons_fields[i])
			fdiff1=flux[self.rs_inside]-flux[self.start]
			fdiff2=flux[self.end]-flux[self.rs_outside]

			integral1=self.src_integral_inside(self.src_fields[i])
			integral2=self.src_integral_outside(self.src_fields[i])
			with warnings.catch_warnings():
				pdiff1=(fdiff1-integral1)*100./integral1
				pdiff2=(fdiff2-integral2)*100./integral2
				pdiff=max(abs(pdiff1), abs(pdiff2))
			if (abs(pdiff)>tol) or np.isnan(pdiff):
				self.check=False
			if ((abs(pdiff)>tol) or np.isnan(pdiff)) and (i==0 or i==3):
				self.check_partial=False

			check_str=check_str+self.cons_fields[i]+'\n'
			pre=['diff1=','diff2=','src1=','src2=','pdiff1=','pdiff2=']
			vals=[fdiff1,fdiff2,integral1,integral2,pdiff1,pdiff2]
			for j in range(len(vals)):
				check_str=check_str+pre[j]
				s='{0:4.3e}'.format(vals[j])
				check_str=check_str+s+'\n'
			check_str=check_str+'____________________________________\n\n'

		if write:
			checkf=open(self.outdir+'/check','w')
			checkf.write(check_str)
		return check_str

	def cons_plot(self, dict1={}, dict2={}):
		if len(self.fdiff)==0:
			self._cons_update()

		fig1,ax1=plt.subplots(3, sharex=True, figsize=(10,24))
		plt.title(self.name)
		for i in range(3):
			ax1[i].set_xscale('log')
			ax1[i].set_yscale('log')
		ax1[2].set_xlabel('Radius [cm]')

		for i in range(1,4):
			ax1[i-1].loglog(self.fdiff[-1,:,0],self.fdiff[-1,:,i], **dict1)
			ax1[i-1].loglog(self.fdiff[-1,:,0],self.fdiff[-1,:,i+3], **dict2)
		plt.close()
		return fig1

	def sol_plot(self, init=False, dict1={}, dict2={}, index=-1):
		fig,ax=plt.subplots()

		for k in range(1,4):
			ax.loglog(self.saved[index,:,0], self.saved[index,:,j], **dict1)
			if init:
				ax.loglog(self.saved[0,:,0], self.saved[0,:,j], **dict2)
		plt.show()
		
	#Adding ghost zones onto the edges of the grid (moving the start of the grid)
	def _add_ghosts(self):
		self.start=self._num_ghosts
		self.end=self.end-self._num_ghosts
	
	def _power_zones(self, rad, i1, i2, field):
		'''Power law extrapolation from zones'''
		r1=self.radii[i1]
		r2=self.radii[i2]
		field_arr=getattr(self, field)
		field1=field_arr[i1]
		field2=field_arr[i2]
		slope=np.log(field2/field1)/np.log(r2/r1)

		return field1*np.exp(slope*np.log(rad/r1))

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
			self._dens_adjust()
			self._mdot_adjust()
		else:
			self._extrapolate('rho')
			self._extrapolate('vel')
			self._extrapolate('s')
		if not self.isot:
			self._update_temp()

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
		field_arr=getattr(self, field)
		for i in range(0, self.start):
			val=self._power_zones(self.radii[i], self.start, self.start+3, field)
			field_arr[i]=val
			if field=='rho':
				self.log_rho[i]=np.log(val)
			
		
		#Updating the end ghost zones, extrapolating using a power law density
		for i in range(self.end+1, self.length):
			val=self._power_zones(self.radii[i], self.end, self.end-3, field)
			field_arr[i]=val
			if field=='rho':
				self.log_rho[i]=np.log(val)

	#Extrapolate densities to the ghost zones; a lot of this method is redundant 
	def _dens_adjust(self):
		'''Extrapolate $\rho$ assuming that the $\rho\sim r^{-3/2}$ on inner boundary and $\rho\sim r^{-3/2}$'''
		r_start=self.radii[self.start]
		r_start2=self.radii[self.start+3]
		log_rho_start=self.log_rho[self.start]
		log_rho_start2=self.log_rho[self.start+3]

		#If the inner bdry is fixed...(appropriate for Parker wind)
		if self.bdry_fixed:
			for i in range(1, self.start):
				self.log_rho[i]=self._interp_zones(self.radii[i], 0, self.start, 'log_rho')
		#Updating the starting ghost zones, extrapolating using rho prop r^-3/2
		else:
			for i in range(0, self.start):
				slope=-3./2.
				#slope=(log_rho_start2-log_rho_start)/np.log(r_start2/r_start)
				log_rho=slope*np.log(self.radii[i]/r_start)+log_rho_start
				self.log_rho[i]=log_rho
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
		return self.q_grid[i]*self.sp_heating_grid[i]/(rho*temp)-vel*ds_dr+art_visc

	def isot_off(self):
		'''Switch off isothermal evolution'''
		self.isot=False
		self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))
		#The next line should not be necessary...
		self._update_temp()
		self.fields=['log_rho', 'vel', 's']

	def isot_on(self):
		'''Switch on isothermal evolution'''
		self.isot=True
		self.fields=['log_rho', 'vel']

	def set_param(self, param, value):
		'''Reset parameter

		:param param: parameter to reset
		:param value: value to reset it to
		'''
		pat=re.compile('params\[\w+\]')
		try:
			old=getattr(self,param)
		except AttributeError:
			old=''

		if param=='gamma' or param=='mu':
			setattr(self,param,value)
			self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))
			if not self.isot:
				self._update_temp()
		elif param=='outdir':
			self.outdir=value
			bash_command('mkdir -p '+value)
			#bash_command('mv '+old+'/log '+old+'/cons.npz '+old+'/save.npz '+old+'/params '+value)
		elif param=='isot':
			if value==True:
				self.isot_on()
			elif value==False:
				self.isot_off()
			else:
				print 'Warning! Invalid value for the passed for parameter isot'
		elif re.findall(pat, param):
			param=param[7:-1]
			try:
				old=self.params[param]
				self.params[param]=value
			except KeyError:
				print 'This Nuker parameter does not exist'
				return
			#Clear cached values if the Nuker parameters have been reset
			self.cache={}
		else:
			setattr(self,param,value)

		if param!='outdir':
			self.non_standard[param]=value

		self.log=self.log+param+' old:'+str(old)+' new:'+str(value)+' time:'+str(self.total_time)+'\n'
			
	#Method to write solution info to file
	def write_sol(self):
		np.savez(self.outdir+'/save', a=self.saved, b=self.time_stamps)
		np.savez(self.outdir+'/cons', a=self.fdiff)
		log=open(self.outdir+'/log', 'w')
		log.write(self.log)
		dill.dump(self.non_standard, open(self.outdir+'/non_standard.p','wb'))

	def backup(self):
		bash_command('mkdir -p '+self.outdir)
		dill.dump(self, open(self.outdir+'/grid.p', 'wb' ) )

	def grid(self):
		self.q_grid
		self.M_enc_grid
		self.phi_grid
		self.sigma_grid
		self.rinf

	def solve(self, time=None, max_steps=np.inf):
		'''Solve hydro equations for galaxy

		:param float time: Time to run solver for. If none run for 20 tcross, but break if certain level of conservation has been reached.
		:param int max_steps: Maximum number of steps for solver to take
		'''
		# the number of steps and the progress
		self.time_cur=0
		self.ninterval=0
		self.num_steps=0
		self.max_steps=max_steps
		if not time:
			self.time_target=self.tmax
		else:
			self.time_target=time
		pbar=progress.ProgressBar(maxval=self.time_target, fd=sys.stdout).start()
		self.check=False


		#While we have not yet reached the target time
		while (self.time_cur<self.time_target):
			if not time and self.check:
				print 'Conservation check satisfied!'
				break

			if (self.tinterval>0 and (self.time_cur/self.tinterval)>=self.ninterval) or (self.tinterval<=0 and self.num_steps%self.sinterval==0):
				pbar.update(self.time_cur)
				self.save()

			#Take step and increment current time
			self._step()
			#If we have exceeded the max number of allowed steps then break
			if self.num_steps>self.max_steps:
				print "Exceeded max number of allowed steps"
				return 1

		pbar.finish()
		self.write_sol()
		return 0

	#Gradually perturb a given parameter (param) to go to the desired value (target). 
	def solve_adjust(self, time, param, target, n=10, max_steps=np.inf):
		'''Run solver for time adjusting the value of params to target in the process

		:param str param: parameter to adjust
		:param target: value to which we would like to adjust the parameter
		:param int n: Number of time intervals to divide time into for the purposes of parameter adjustment
		:param int max_steps: Maximum number of steps for solver to take
		'''
		param_cur=getattr(self, param)
		print target, param_cur
		interval=time/float(n)
		delta_param=(target-param_cur)/float(n)
		while not np.allclose(param_cur, target):
			param_cur+=delta_param
			self.set_param(param,param_cur)

			steps=self.solve(time=interval, max_steps=max_steps)
			if steps==1:
				break
			
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

	def save(self):
		'''Write state of grid to file'''
		grid_prims=[getattr(self, field) for field in self.out_fields]
		grid_prims[2]=grid_prims[2]/grid_prims[7]
		self.time_stamps.append(self.total_time)
		self.saved=np.append(self.saved,[np.transpose(grid_prims)],0)

		self._cons_update()
		self.cons_check()
		self.ninterval+=1
		self.write_sol()

		if np.any(np.isnan(grid_prims)):
			sys.exit(3)

	def revert(self, index):
		'''Revert grid to an earlier state'''
		self.log_rho=np.log(self.saved[index,:,1])
		self.vel=self.saved[index,:,2]*self.saved[index,:,7]
		self.s=self.saved[index,:,6]
		if not self.isot:
			self._update_temp()

		index=index%self.length
		self.saved=self.saved[:index+1]
		self.time_stamps=self.time_stamps[:index+1]
		self.fdiff=self.fdiff[:index+1]
		#Overwriting previous saved files
		self.write_sol()


	#Clear all of the info in the saved list
	def clear_saved(self):
		self.saved=np.empty([0, self.length, len(self.out_fields)])
		self.fdiff=np.empty([0, self.length-1,  2*len(self.cons_fields)+1])
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
			if not self.isot:
				self._update_temp()
			self._update_ghosts()

		self.time_cur+=self.delta_t
		self.total_time+=self.delta_t
		self.num_steps+=1

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
	def mdot_from_v(self):
		'''Mass accretion rate based on stagnation radius 
		'''
		mdot=4.*np.pi*self.radii[self.start]**2*self.rho[self.start]*abs(self.vel[self.start])
		return mdot

	@property
	def mdot(self):
		return self.eta*self.M_enc_interp(self.rs)/th

	@property
	def mdot_bondi(self):
		return 4.*np.pi*0.25*(G*self.params['M'])**2.*self.cs_interp(self.rb)**-3*self.rho_interp(self.rb)

	@property
	def mdot_approx(self):
		(4./3.)*np.pi*self.rs**2*self.rho_interp(self.rs)*(G*self.params['M']/self.rs)**0.5

	@property
	def eddr(self):
		'''Compute the Eddington ratio assuming a 10 percent radiative efficiency
		'''
		return self.mdot/gal_properties.mdot_edd(self.params['M'], efficiency=0.1)

	@property
	def xray(self):
		'''Simple prescription for x-ray luminosity, If we are below a particular edd. ratio<0.03  Lx~(mdot)^2 c^2; otherwise Lx~0.1 mdot c^2'''
		l_0=0.03*gal_properties.l_edd(self.params['M'])
		if self.eddr<0.03:
			xray=(self.eddr/0.03)**2*l_0
		else:
			xray=(gal.eddr/0.03)*l_0

		return xray

	@property	
	def cooling(self):
		'''Cooling luminosity'''
		lambdas=np.array([lambda_c(temp) for temp in self.temp])
		return lambdas*(self.rho/(self.mu*mp))**2

	@property
	def x_ray_lum(self):
		'''Integrated x-ray luminosity at each grid radius'''
		return np.cumsum(4.*np.pi*self.cooling*self.delta*self.radii**2)

	def x_ray_lum_interp(self, r):
		return interp1d(self.radii, self.x_ray_lum)(r)

class NukerGalaxy(Galaxy):
	'''Sub-classing galaxy above to represent Nuker parameterized galaxies'''
	def __init__(self, gname, gdata=None, init={}):
		Galaxy.__init__(self, init=init)
		if not gdata:
			gdata=nuker_params()
		try:
			self.params=gdata[gname]
		except KeyError:
			print 'Error! '+gname+' is not in catalog!'
			raise

		names=['Name', 'Type','M', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$I_b$', r'$r_b$', 'Uv']
		
		self.params_table=Table([self.params])
		self.params_table=Table(self.params_table['Name', 'type', 'M', 'alpha', 'beta', 'gamma', 'Ib', 'rb', 'Uv'], names=names)
		self.params_table['M'].format=latex_exp.latex_exp
		self.params_table[r'$I_b$'].format=latex_exp.latex_exp
		#self.params_table[r'$r_b$'].format=latex_exp.latex_exp
		self.params_table['M']=self.params_table['M']/M_sun
		self.params_table['M'].unit=u.MsolMass

		self.name=gname
		self.eta=1.
		self.rmin_star=1.E-3
		self.rmax_star=1.E5

		self.rg=G*self.params['M']/c**2


	@classmethod
	def from_dir(cls, name, loc, index=-1, rescale=1., gdata=None, length=None):
		init={}
		init_array=prepare_start(np.load(loc+'/save.npz')['a'][index])
		if rescale=='auto':
			a=96.5
			tmp=cls(name, gdata=gdata)
			rescale=tmp.rinf/init_array[0,0]/a
			print rescale

		init_array[:,0]=rescale*init_array[:,0]
		radii=init_array[:,0]
		init['rmin']=radii[0]
		init['rmax']=radii[-1]
		delta=np.diff(radii)
		delta_log=np.diff(np.log(radii))

		if np.allclose(np.diff(delta),[0.]):
			init['logr']=False
		else:
			init['logr']=True
		init['f_initial']=extrap1d(interp1d(init_array[:,0], init_array[:,1:4], axis=0))
		if not length:
			init['length']=len(radii)
		else:
			init['length']=length

		return cls(name, init=init, gdata=gdata)

	@memoize
	def rho_stars(self,r):
		'''Stellar density
		:param r: radius 
		'''
		rpc=r/pc
		if rpc<self.rmin_star or rpc>self.rmax_star:
			return 0.
		else:
			return M_sun*self.params['Uv']*inverse_abel(nuker_prime, rpc, **self.params)/pc**3

	@memoize
	def _get_rho_stars_interp(self):
		_rho_stars_rad=np.logspace(np.log10(self.rmin_star), np.log10(self.rmax_star),1000)*pc
		_rho_stars_grid=[self.rho_stars(r) for r in _rho_stars_rad]
		return interp1d(_rho_stars_rad,_rho_stars_grid)

	@property
	def _rho_stars_interp(self):
		return self._get_rho_stars_interp()

	@memoize
	def M_enc(self,r):
		'''Mass enclosed within radius r
		:param r: radius 
		'''
		rpc=r/pc
		if rpc<self.rmin_star:
			return 0.
		elif rpc>self.rmax_star:
			return self.M_enc(self.rmax_star*pc)
		else:
			return integrate.quad(lambda r1:4.*np.pi*r1**2*self._rho_stars_interp(r1*pc)*pc**3, self.rmin_star, rpc)[0]

	@memoize
	def sigma(self, r):
		'''Velocity dispersion of galaxy
		:param r: radius 
		'''
		return (G*(self.M_enc(r)+self.params['M'])/(r))**0.5

	@memoize
	def phi_s(self,r):
		'''Potential from the stars
		:param r: radius 
		'''
		rpc=r/pc
		return (-G*self.M_enc(r)/r)+4.*G*self.params['Uv']*M_sun*integrate.quad(lambda r1:nuker_prime(r1, **self.params)*(r1**2-rpc**2)**0.5, rpc, self.rmax_star)[0]/pc

	def phi_bh(self,r):
		'''Potential from the black hole
		:param r: radius 
		'''
		return -G*self.params['M']/r

	def q(self, r):
		'''Source term representing mass loss from stellar winds'''
		return self.eta*self.rho_stars(r)/th

	@property
	def sigma_200(self):
		'''Reverse engineering the velocity dispersion from Mbh-sigma relationship and BH mass--using the relationship in WM04'''
		return gal_properties.sigma_200(self.params['M'])

	@property
	def r_Ia(self):
		return gal_properties.r_Ia(self.params['M'])

	##Getting the radius of influence: where the enclosed mass begins to equal the mass of the central BH. 
	@memoize
	def _get_rinf(self):
		'''Get radius of influence for galaxy'''
		def mdiff(r):
			return self.params['M']-self.M_enc(r)

		jac=lambda r: -4.*np.pi*r**2*self.rho_stars(r)
		return fsolve(mdiff, 0.9*pc*(self.params['M']/(1.E6*M_sun))**0.4, fprime=jac)[0]

	@property
	def rinf(self):
		return self._get_rinf()

	@property
	def sigma_inf(self):
		'''Velocity dispersion at the radius of influence.'''
		return self.sigma(self.rinf)

	@property
	def rb(self):
		'''Bondi radius computed from G M/cs(rb)^2=rb'''
		f=lambda r:G*self.params['M']/(self.cs_interp(r))**2-r
		rb=fsolve(f, G*self.params['M']/self.cs[-1]**2)

		return rb

	def get_prop(self,field,r):
		return getattr(self, field)(r)

	@property
	def rs_analytic(self):
		'''Analytic formula for the stagnation radius'''
		A=(4.*self.gamma-(1+self.params['gamma'])*(self.gamma-1.))/(4.*(self.gamma-1.))
		eta=self.vw_extra/self.sigma(self.rinf)
		omega=self.M_enc(self.rs)/self.params['M']

		lrho_interp=interp1d(np.log(self.radii),self.log_rho)
		dens_slope=np.abs(derivative(lrho_interp, np.log(self.rs), dx=self.delta_log[0]))

		return 1./(dens_slope*eta**2)*((0.5)*(2*A-dens_slope)*(1+omega)-(2.-self.params['gamma'])/4.)

	@property 
	def rs_residual(self):
		'''Residual of the stagnation radius from the analytic result'''
		return (self.rs_analytic-(self.rs/self.rinf))/self.rs_analytic

	def rho_func(self, r):
		try:
			return np.interp(r, self.radii, self.rho)
		except:
			if r<self.radii[0]:
				return self._power_zones(r, 0, 3, 'rho')
			else:
				return self._power_zones(r, self.length-1, self.length-4, 'rho')

	@property
	def tde_table(self):
		'''Get crossing radius for jet'''
		m6=self.params['M']/(1.E6*M_sun)
		jet=tde_jet.Jet()
		jet.m6=m6

		r=integrate.ode(jet.vj)

		r.set_integrator('vode')
		r.set_initial_value(0., t=jet.delta).set_f_params(self.rho_func)
		time=0.
		try:
			while r.y<r.t:
				old=r.y[0]
				r.integrate(r.t+0.01*pc)
				new=r.y[0]
				time+=(new-old)/(jet.beta_j*c)
			rc=r.y[0]
		except Exception as inst:
			print inst
			rc=np.nan

		f=jet.rho(rc)/self.rho_func(rc)
		gamma=jet.gamma_j*(1.+2.*jet.gamma_j*f**(-0.5))**(-0.5)   

		rc_col=Column([rc/pc], name=r'$r_c$', format=latex_exp.latex_exp)
		n_rc_col=Column([self.rho_func(rc)/(self.mu*mp)], name=r'$n_{rc}$',format=latex_exp.latex_exp)
		gamma_rc_col=Column([gamma], name=r'$\Gamma_{rc}$',format=latex_exp.latex_exp)
		t_rc_col=Column([time/year], name=r't$_{rc}$',format=latex_exp.latex_exp)
		M_col=self.params_table['M']

		type_col=self.params_table['Type']
		name_col=self.params_table['Name']

		tde_table=Table([name_col, M_col, type_col, rc_col, n_rc_col, gamma_rc_col, t_rc_col])

		return tde_table


	@property 
	def summary(self):
		'''Summary of galaxy properties'''
		return table.hstack([self.params_table, self.tde_table, Table([[self.rs/pc]], names=['$r_{s}$']), Table([[self.eddr]],\
			names=[r'$\dot{M}/\dot{M_{\rm edd}}$'])])


		
































