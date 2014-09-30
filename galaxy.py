import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, brentq
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

import os.path
import re
import functools

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
me=const.m_e.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value
kpc=1.E3*pc
eV=u.eV.to('erg')
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
			return interpolator([x])[0]

	def ufunclike(xs):
		try:
			return np.array(map(pointwise, xs))
		except TypeError:
			return pointwise(xs)

	return ufunclike

def zero_crossings(a):
	'''Identify zero crossings of an array'''
	return np.where(np.diff(np.sign(a)))[0]

def s(temp, rho, mu):
	return (kb/(mu*mp))*np.log(1./rho*(temp)**(3./2.))

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

def _check_format(vals):
	check_str=''
	pre=['diff1=','diff2=','src1=','src2=','pdiff1=','pdiff2=']
	for j in range(len(vals)):
		check_str=check_str+pre[j]
		s='{0:4.3e}'.format(vals[j])
		check_str=check_str+s+'\n'
	check_str=check_str+'____________________________________\n\n'
	return check_str

def bash_command(cmd):
	'''Run command from the bash shell'''
	process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	return process.communicate()[0]

def temp_kev(temp):
	'''Convert temperature (K) to keV'''
	return kb*temp/eV/1.E3

def kev_temp(en):
	return en*eV*1.E3/kb

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
	#vsig=ascii.read('vsig.csv', delimiter=',',names=['gal', 'vsig','eps'])
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
		d['M2']=M_sun*10.**table[i]['$\\log_{10}(M_{\\bullet}/M_{\\odot})$\\tablenotemark{c}']
		d['type']=table[i][r'Profile\tablenotemark{b}']
		if d['type']=='$\\cap$':
			d['type']='Core'
		else:
			d['type']='Cusp'
		# vsig_idx=np.where(vsig['gal']==table[i]['Name'])[0]
		# if vsig['vsig'][vsig_idx]:
		# 	d['vsig']= vsig['vsig'][vsig_idx][0]
		# else:
		# 	d['vsig']=None

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

		self.tol=40.
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
	def from_dir(cls, loc, index=-1, rescale=1, length=None, rmin=None, rmax=None):
		init={}

		init_array=prepare_start(np.load(loc+'/save.npz')['a'][index])
		init_array[:,0]=rescale*init_array[:,0]
		radii=init_array[:,0]
		if not rmin:
			init['rmin']=radii[0]
		else:
			init['rmin']=rmin
		if not rmax:
			init['rmax']=radii[-1]
		else:
			init['rmax']=rmax

		delta=np.diff(radii)
		delta_log=np.diff(np.log(radii))

		if np.allclose(np.diff(delta),[0.]):
			init['logr']=False
		else:
			init['logr']=True
		init['f_initial']=extrap1d_pow(interp1d(init_array[:,0], init_array[:,1:4], axis=0))
		if not length:
			init['length']=len(radii)
		else:
			init['length']=length

		return cls(init=init)

	def restore_saved(self, loc):
		'''Restored saved data to galaxy'''
		saved=np.load(loc+'/save.npz')
		self.saved=saved['a']
		self.time_stamps=saved['b']

	def _init_grid(self):
		#Initializing the radial grid
		rmin,rmax,f_initial=self.init['rmin'],self.init['rmax'],self.init['f_initial']
		self.func_params=self.init['func_params']
		self._logr=self.init['logr']

		assert rmax>rmin
		assert hasattr(f_initial, '__call__')
		self.length=self.init['length']

		if self._logr:
			radii=np.logspace(np.log(rmin), np.log(rmax), self.length, base=e)
		else:
			radii=np.linspace(rmin, rmax, self.length)
		prims=[f_initial(r, **self.func_params) for r in radii]

		self.radii=radii
		prims=np.array(prims)
		delta=np.diff(self.radii)
		self.delta=np.insert(delta, 0, delta[0])
		delta_log=np.diff(np.log(self.radii))
		self.delta_log=np.insert(delta_log, 0, delta_log[0])

		self.log_rho=np.log(prims[:,0])
		self.vel=prims[:,1]
		self.temp=prims[:,2]
		self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))

	def re_grid(self, rmin, rmax, length=None):
		#If length kw arg is unspecified then leave the number of grid points unchanged
		if not length:
			length=self.length

		#Clearing past saved information
		self.clear_saved()
		#New initialization parameters
		self.init={'rmin':rmin, 'rmax':rmax, 'length':length, 'logr':self._logr, 'f_initial':self.profile, 'func_params':{}}
		self._init_grid()

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

	@property
	def M_bh_8(self):
		return self.params['M']/1.E8/M_sun

	@property 
	def vw_extra_500(self):
		return self.vw_extra/5.E7

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

	def q_gridpt(self, i):
		return self.q(self.radii[i])

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

	def vw_func(self, r):
		if not self.sigma_heating:
			return self.vw_extra
		else:
			return (self.sigma(r)**2+(self.vw_extra)**2)**0.5

	@property 
	def vw(self):
		return np.array([self.vw_func(r) for r in self.radii])
		
	@property
	def rho(self):
		return np.exp(self.log_rho)

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
	def mach(self):
		return self.vel/self.cs

	@property 
	def tcross_local(self):
		return self.radii/self.cs

	@property 
	def r_ss(self):
		'''Radius at which the velocity would become supersonic on the inner boundary'''
		slope=get_slope(self.radii[0], self.radii[3], self.mach[0], self.mach[3])
		r_ss=self.radii[0]*np.exp((1./slope)*np.log(-1./self.mach[0]))

		return r_ss
  
	@property 
	def alpha_max(self):
		return np.max([np.abs(self.vel+self.cs), np.abs(self.vel-self.cs)],axis=0)

	@property 
	def sp_heating(self):
		return (0.5*self.vel**2+0.5*self.vw**2-(self.gamma)/(self.gamma-1)*(self.pres/self.rho))

	@property
	def heating_pos(self):
		return 0.5*self.q_grid*(self.vel**2+self.vw**2)

	@property 
	def u(self):
		return self.pres/(self.rho*(self.gamma-1.))

	@property 
	def bernoulli(self):
		return 0.5*self.vel**2+(self.pres/self.rho)+self.u+self.phi_grid

	@property 
	def fen(self):
		return self.rho*self.radii**2*self.vel*self.bernoulli+self.radii**2*self.f_cond

	@property 
	def src_rho(self):
		return self.q_grid*self.radii**2

	@property 
	def src_en(self):
		return self.radii**2.*self.q_grid*(self.vw**2/2.+self.phi_grid)

	@property
	def src_v(self):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return -(self.q_grid*self.vel/self.rho)+(self.q_grid*self.sp_heating/(self.rho*self.vel))

	@property
	def src_s(self):
		if self.isot:
			src_s=np.zeros(self.length)
		else:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				src_s=(self.q_grid*self.sp_heating+self.cond_grid)/(self.rho*self.vel*self.temp)
		return src_s

	@property
	def spitzer(self):
		'''Spitzer conductivity'''
		kappa0=2*10.**-6.
		kappa=kappa0*self.temp**(5./2.)
		return kappa

	@property
	def shcherba(self):
		'''Conductivity from Shcherbakov'''
		kappa=0.1*kb*np.sqrt(kb*self.temp/me)*self.radii*(self.rho/(self.mu*mp))
		return kappa

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

	def src_rho_interp(self, r):
		return interp1d(self.radii, self.src_rho)(r)

	def src_s_interp(self, r):
		return interp1d(self.radii, self.src_s)(r)

	def src_v_interp(self, r):
		return interp1d(self.radii, self.src_v)(r)

	def src_en_interp(self, r):
		return interp1d(self.radii, self.src_en)(r)	

	def rho_profile(self, r):
		'''Calculate density at any radius from our profile--use power law extrapolations beyond the boundary'''
		return extrap1d_pow(interp1d(self.radii, self.rho))(r)

	def vel_profile(self,r):
		'''Calculate density at any radius from our profile--use power law extrapolations beyond the boundary'''
		return extrap1d_pow(interp1d(self.radii, self.vel))(r)		

	def vel_profile_from_mdot(self,r):
		rho=self.rho_profile(r)
		if r>self.radii[-1]:
			return self.frho[-1]/r**2/rho
		elif r<self.radii[0]:
			return self.frho[0]/r**2/rho
		else:
			return self.vel_interp(r)

	def temp_profile(self, r):
		'''Calculate temperature at any radius from our profile--use power law extrapolations beyond the boundary'''
		return extrap1d_pow(interp1d(self.radii, self.temp))(r)

	def profile(self, r):
		return (self.rho_profile(r), self.vel_profile_from_mdot(r),self.temp_profile(r))

	def cooling_profile(self,r):
		'''Calculate cooling rate at any radius from our profile--use power law extrapolations beyond the boundary'''
		return extrap1d_pow(interp1d(self.radii, self.cooling))(r)

	def heating_pos_profile(self,r):
		'''Calculate heating profile of the grid'''
		return extrap1d_pow(interp1d(self.radii, self.heating_pos))(r)

	def cs_profile(self,r):
		'''sound speed at any radius'''
		return gal_properties.cs(self.temp_profile(r),mu=self.mu, gamma=self.gamma)

	def _update_temp(self):
		self.temp=(np.exp(self.log_rho)*np.exp(self.mu*mp*self.s/kb))**(2./3.)

	def set_temp(self, temp):
		'''set temperature to a particular constant value--to help with conduction'''
		self.s=np.array([s(temp, self.rho[i], self.mu) for i in range(self.length)])
		self._update_temp()

	#Update array of conseerved quantities	
	def _cons_update(self):
		self._update_aux()
		#differences in fluxes and source terms
		fdiff=np.empty([2*len(self.cons_fields)+1, self.length-1])
		fdiff[0]=self.radii[1:]
		for i in range(len(self.cons_fields)):
			flux=getattr(self, self.cons_fields[i])
			#get differences in fluxes for all of the cells.
			fdiff[i+1]=np.diff(flux)
			#get the source terms for all of the cells.
			fdiff[i+4]=(getattr(self, self.src_fields[i])*self.delta)[1:]
			
		self.fdiff=np.append(self.fdiff,[np.transpose(fdiff)],0)

	@property
	def stag_unique(self):
		if len(self.rs)==1:
			return True
		else:
			return False

	@property
	def rs_outside(self):
		if self.stag_unique:
			return (np.where(self.radii>self.rs))[0][0]
	@property
	def rs_inside(self):
		if self.stag_unique:
			return (np.where(self.radii<self.rs))[0][-1]

	def src_integral(self, src_field, i1, i2):
		if src_field in self.src_fields:
			src=getattr(self, src_field)
			return 4.*np.pi*np.trapz(src[i1:i2+1], x=self.radii[i1:i2+1])

	def fdiff_seg(self, cons_field, i1, i2):
		if cons_field in self.cons_fields:
			flux=4.*np.pi*getattr(self, cons_field)
			return flux[i2]-flux[i1]

	def src_integral_outside(self, src_field):
		if self.stag_unique and src_field in self.src_fields:
			src=getattr(self, src_field)
			return 4.*np.pi*np.trapz(src[self.rs_outside:self.end+1], x=self.radii[self.rs_outside:self.end+1])
	
	def src_integral_inside(self, src_field):
		if self.stag_unique and src_field in self.src_fields:
			src=getattr(self, src_field)
			return 4.*np.pi*np.trapz(src[self.start:self.rs_inside+1], x=self.radii[self.start:self.rs_inside+1])

	def fdiff_inside(self, cons_field):
		if self.stag_unique and  cons_field in self.cons_fields:
			flux=4.*np.pi*getattr(self, cons_field)
			return flux[self.rs_inside]-flux[self.start]	

	def fdiff_outside(self, cons_field):
		if self.stag_unique and cons_field in self.cons_fields:
			flux=4.*np.pi*getattr(self, cons_field)
			return flux[self.end]-flux[self.rs_outside]	

	def _pdiff(self, i):
		fdiff=np.array([self.fdiff_inside(self.cons_fields[i]), self.fdiff_outside(self.cons_fields[i])])
		integral=np.array([self.src_integral_inside(self.src_fields[i]),self.src_integral_outside(self.src_fields[i])])
		if fdiff[0]==None:
			pdiff=[None,None]
		else:
			with warnings.catch_warnings():
				pdiff=(fdiff-integral)*100./integral
		return [fdiff, integral, pdiff]
		
	def pdiff_max(self):
		pdiff_max=0.
		for i in range(len(self.cons_fields)):
			fdiff,integral,pdiff=self._pdiff(i)
			pdiff_max=max([pdiff_max, max(np.abs(pdiff))])
		return pdiff_max
		
	def cons_check(self, write=True):
		'''Check level of conservation'''
		self.check=True
		self.check_partial=True
		check_str=''
		if not hasattr(self, 'tol'):
			self.tol=40.
		try:
			for i in range(len(self.cons_fields)):
				fdiff,integral,pdiff=self._pdiff(i)
				pdiff_max=abs(np.max(np.abs(pdiff)))
				if (pdiff_max>self.tol) or np.isnan(pdiff_max):
					self.check=False
					if i==0 or i==3:
						self.check_partial=False
				
				vals=[fdiff[0],fdiff[1],integral[0],integral[1],pdiff[0],pdiff[1]]
				check_str=check_str+self.cons_fields[i]+'\n'
				check_str=check_str+_check_format(vals)
		except:
			self.check=False
			self.check_partial=False
			check_str='Could not perform conservation check.'
			if not self.stag_unique:
				print 'Stagnation point is not uniquely defined.'
			
		if write:
			checkf=open(self.outdir+'/check','w')
			checkf.write(check_str)
		return check_str

	def cons_plot_rhr(self, dict1={}, dict2={}):
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

	def cons_plot(self, dict1={}, dict2={}):
		'''Plot integrated source vs. flux difference for pairs of grid points'''
		fig1,ax1=plt.subplots(4, sharex=True, figsize=(10,32))
		for i in range(4):
			src=[self.src_integral(self.src_fields[i], j, j+1) for j in range(self.length-1)]
			fdiff=[self.fdiff_seg(self.cons_fields[i], j, j+1) for j in range(self.length-1)]
			ax1[i].loglog(self.radii[1:],src)
			ax1[i].loglog(self.radii[1:],fdiff)
		plt.close()
		return fig1
		

	def sol_plot(self, init=False, dict1={}, dict2={}, index=-1):
		fig1,ax1=plt.subplots(3, sharex=True, figsize=(10,24))

		for k in range(1,4):
			ax1[k-1].loglog(self.saved[index,:,0], abs(self.saved[index,:,k]), **dict1)
			if init:
				ax1[k-1].loglog(self.saved[0,:,0], abs(self.saved[0,:,k]), **dict2)
		plt.close()
		return fig1

	@property 
	def plot_marker(self):
		if self.params['gamma']<0.2:
			return '<'
		else:
			return 's'

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
		'''Method to update boundaries--bp=bondi-parker; s_fixed=fixed entropy on the inner grid (entropy is simply left alone. Otherwise do power law
			extrapolations for all variables.'''
		if self.bdry=='bp':
			self._update_ghosts_bp()
		elif self.bdry=='s_fixed':
			self._update_ghosts_s_fixed()
		elif self.bdry=='temp_fixed':
			self._update_ghosts_temp_fixed()
		elif self.bdry=='non_cond':
			self._update_ghosts_non_cond()
		else:
			self._update_ghosts_default()
			
		if not self.isot:
			self._update_temp()

	def _update_ghosts_default(self):
		self._extrapolate('rho')
		self._extrapolate('vel')
		self._extrapolate('s')

	def _update_ghosts_s_fixed(self):
		self._extrapolate('rho')
		self._extrapolate('vel')

	def _update_ghosts_bp(self):
		self._s_adjust()
		self._dens_adjust()
		self._mdot_adjust()

	def _update_ghosts_temp_fixed(self):
		temp_inner,temp_outer=self.temp[0],self.temp[-1]
		self._extrapolate('rho')
		self._extrapolate('vel')
		self.s[0],self.s[-1]=s(temp_inner,self.rho[0],self.mu),s(temp_outer,self.rho[-1],self.mu)
		self._bdry_interp('s')

	def _update_ghosts_non_cond(self):
		self._extrapolate('rho')
		self._extrapolate('vel')
		for i in range(0, self.start):
			self.s[i]=s(self.temp[self.start],self.rho[i],self.mu)
		for i in range(self.end+1,self.length):
			self.s[i]=s(self.temp[self.end],self.rho[i],self.mu)


	def _bdry_interp(self,field):
		field_arr=getattr(self, field)
		for i in range(1, self.start):
			field_arr[i]=self._interp_zones(self.radii[i], 0, self.start, field)

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
		# if self.bdry_fixed:
		# 	for i in range(1, self.start):
		# 		self.log_rho[i]=self._interp_zones(self.radii[i], 0, self.start, 'log_rho')
		#Updating the starting ghost zones, extrapolating using rho prop r^-3/2

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

	#Evaluate terms of the form div(kappa*df/dr)-->Diffusion like terms (useful for something like the conductivity)
	def get_diffusion(self, i, coeff, field):
		dkappa_dr=self.get_spatial_deriv(i, coeff)
		dfield_dr=self.get_spatial_deriv(i, field)
		d2field_dr2=self.get_spatial_deriv(i, field, second=True)

		kappa=getattr(self,coeff)[i]
		return kappa*d2field_dr2+dkappa_dr*dfield_dr+(2./self.radii[i])*(kappa*dfield_dr)

	#Getting derivatives for a given field (density, velocity, etc.). If second is set to be true then the discretized 2nd
	#deriv is evaluated instead of the first
	def get_spatial_deriv(self, i, field, second=False):
		if i<self.start or i>self.end:
			return np.nan

		field_list=getattr(self,field)[i-3:i+4]
		if second:
			return np.sum(field_list*self.second_deriv_coeffs[i])		
		else:
			return np.sum(field_list*self.first_deriv_coeffs[i])


	#Calculate laplacian in spherical coords. 
	def get_laplacian(self, i, field):
		return self.get_spatial_deriv(i, field, second=True)+(2./self.radii[i])*(self.get_spatial_deriv(i, field))

	def _cfl(self):
		# alpha_max=0.
		delta_t=np.zeros(self.length)
		delta_t=self.safety*self.delta/self.alpha_max

		#Setting the time step
		cfl_delta_t=np.min(delta_t)
		target_delta_t=self.time_target-self.time_cur
		self.delta_t=min([target_delta_t, cfl_delta_t])

	# @property
	# def _delta_t_cfl(self):
	# 	return np.min(self.safety*self.delta/self.alpha_max)

	# @property 
	# def _delta_t_cond(self):
	# 	if not hasattr(self, 'phi_cond'):
	# 		self.set_param('phi_cond',1.)
	# 	return np.min(np.abs(self.safety*self.rho[self.start:self.end+1]*self.cs[self.start:self.end+1]**2*
	# 		self.delta[self.start:self.end+1]/(self.f_cond[self.start:self.end+1])))

	# #Evaluate Courant condition for the entire grid. This gives us an upper bound on the time step we may take 
	# def _timestep(self):
	# 	# alpha_max=0.
	# 	delta_t_allowed=min([self._delta_t_cfl,self._delta_t_cond])
	# 	#Setting the time step
	# 	delta_t_target=self.time_target-self.time_cur
	# 	self.delta_t=min([delta_t_target, delta_t_allowed])

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
		return -(1./rad)**2*self.get_spatial_deriv(i, 'frho')+self.q_gridpt(i)

	@property
	def art_visc_vel(self):
		art_visc=[min(self.cs[i],  np.abs(self.vel[i]))*(self.radii[self.end]-self.radii[self.start])* self.get_laplacian(i, 'vel')/self.Re for i in range(self.length)]
		art_visc=np.array(art_visc)
		if self.visc_scheme=='const_visc':
			pass
		elif self.visc_scheme=='cap_visc':
			art_visc=art_visc*min(1., (self.delta[i]/np.mean(self.delta)))
		else:
			art_visc=art_visc*(self.delta[i]/np.mean(self.delta))
		return art_visc

	@property
	def art_visc_s(self):
		art_visc=[min(self.cs[i],  np.abs(self.vel[i]))*(self.radii[self.end]-self.radii[self.start])* self.get_laplacian(i, 's')/self.Re_s for i in range(self.length)]
		art_visc=np.array(art_visc)
		if self.visc_scheme=='const_visc':
			pass
		elif self.visc_scheme=='cap_visc':
			art_visc=art_visc*min(1., (self.delta[i]/np.mean(self.delta)))
		else:
			art_visc=art_visc*(self.delta[i]/np.mean(self.delta))
		return art_visc

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

		return -vel*dv_dr-dlog_rho_dr*kb*temp/(self.mu*mp)-(kb/(self.mu*mp))*dtemp_dr-self.grad_phi_grid[i]+art_visc-(self.q_gridpt(i)*vel/rho)

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

		return self.q_gridpt(i)*self.sp_heating[i]/(rho*temp)-vel*ds_dr+art_visc+self.cond(i)/(rho*temp)

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

	def get_param(self, param):
		pat=re.compile('params\[\w+\]')
		if re.findall(pat, param):
			param=param[7:-1]
			return self.params[param]
		else:
			try:
				return getattr(self, param)
			except AttributeError:
				return None

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
		self.backup()

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
			#If we have exceeded the max number of allowed steps then break--might 
			# want to raise exception here instead.
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

	def refine(self):
		'''Reset and clear saved info--useful capturing how a solution blows up'''
		self.set_param('outdir', 'refine')

		self.reset(-1)
		self.clear_saved()
		self.set_param('sinterval',1)
		self.set_param('tinterval',-1)

		self.solve()
	
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
			print 'nan detected in solution'
			sys.exit(3)
			
	def reset(self, index):
		'''Reset density, temperature, and velocity of grid'''
		self.log_rho=np.log(self.saved[index,:,1])
		self.vel=self.saved[index,:,2]*self.saved[index,:,7]
		self.temp=self.saved[index,:,3]
		self.s=self.saved[index,:,6]
		
	def revert(self, index):
		'''Revert grid, saved, fdiff array and output files to an earlier state.'''
		self.reset(index)
		
		index=index%len(self.time_stamps)
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
			self._update_aux()
			self._sub_step(gamma[substep], zeta[substep])
			if not self.isot:
				self._update_temp()
			self._update_ghosts()
		grid_prims=[getattr(self, field) for field in self.out_fields]
		if np.any(np.isnan(grid_prims)):
			print 'nan detected in solution'
			sys.exit(3)

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
		crossings=zero_crossings(self.vel)
		zeros=[]
		for cross in crossings:
			zeros.append(brentq(self.vel_interp, self.radii[cross], self.radii[cross+1]))
		return zeros

	#Get accretion rate  for galaxy by integrating source from stagnation radius
	@property
	def mdot_from_v(self):
		'''Mass accretion rate based on stagnation radius 
		'''
		mdot=4.*np.pi*self.radii[self.start]**2*self.rho[self.start]*abs(self.vel[self.start])
		return mdot

	@property
	def cons_indices(self):
		cons_indices={}
		for idx, field in enumerate(self.cons_fields):
			cons_indices[field]=idx
		return cons_indices
		
	def cons_index(self,cons_field):
		try:
			return self.cons_indices[cons_field]
		except KeyError:
			return None
	
	def convergence(self, cons_field):
		'''Check for convergence of conserved quantity'''
		cons_idx=self.cons_index(cons_field)
		if cons_idx==None:
			return
		sol,src,sol_out,src_out=[],[],[],[]

		for i in range(len(self.saved)):
			self.reset(i)
			fdiff,integral,pdiff=self._pdiff(cons_idx)
			src.append(fdiff[0])
			sol.append(integral[0])
			src_out.append(fdiff[1])
			sol_out.append(integral[1])
					
		return [src, sol, src_out, sol_out]

	def convergence_plot(self):
		fig,ax=plt.subplots(nrows=2, ncols=2, figsize=(10,8))
		series=[self.convergence('frho'),self.convergence('fen')]
		fig.suptitle(self.name+','+str(self.vw_extra/1.E5)) 
		
		ax[0,0].set_yscale('log')
		ax[0,1].set_yscale('log')
		for j in range(2):
			ax[j,0].plot(self.time_stamps, series[j][0])
			ax[j,0].plot(self.time_stamps, series[j][1])
			ax[j,1].plot(self.time_stamps, series[j][2])
			ax[j,1].plot(self.time_stamps, series[j][3])
		
		plt.close()
		return fig

	@property
	def mdot(self):
		if self.stag_unique:
			return self.eta*self.M_enc_interp(self.rs[0])/th

	@property
	def mdot_bondi(self):
		return gal_properties.mdot_bondi(self.params['M'], self.cs_profile(self.rb), self.rho_profile(self.rb))[0]

	@property
	def mdot_bondi_ratio(self):
		return self.mdot_bondi/self.mdot

	@property
	def mdot_approx(self):
		(4./3.)*np.pi*self.rs**2*self.rho_interp(self.rs)*(G*self.params['M']/self.rs)**0.5

	@property
	def mdot_edd(self):
		'''Calculate Eddington accretion rate assuming a 10 percent radiative efficiency'''
		return gal_properties.mdot_edd(self.params['M'], efficiency=0.1)

	@property
	def eddr(self):
		'''Compute the Eddington ratio assuming a 10 percent radiative efficiency
		'''
		return self.mdot/self.mdot_edd

	@property
	def bh_xray(self):
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

	@property 
	def chandra_obs(self):
		'''"Observation" of rho/T Chandra resolution limit (~0.1 kpc at 17 Mpc--fixed for now)'''
		r1,r2=0.1*kpc,0.2*kpc
		rho1,rho2=self.rho_profile(r1),self.rho_profile(r2)
		temp1,temp2=self.temp_profile(r1),self.temp_profile(r2)

		return {'rad':[r1,r2],'rho':[rho1,rho2], 'temp':[temp1, temp2]}
		
	@property 
	def chandra_rb(self):
		'''Bondi radius which would be inferred for Chandra observation.'''
		co=self.chandra_obs
		temp1=co['temp'][0]
		cs=gal_properties.cs(temp1,mu=self.mu,gamma=self.gamma)

		return G*self.params['M']/cs**2.

	@property
	def chandra_extrap(self):
		'''Extrapolating the quantities that would be observed by Chandra to rb'''
		co=self.chandra_obs
		r1,r2=co['rad']
		rho1,rho2=co['rho']
		temp1,temp2=co['temp']

		temp_rb=temp1
		rho_rb=pow_extrap(self.chandra_rb,r1,r2,rho1,rho2)
		return rho_rb,temp_rb

	@property
	def chandra_plot(self):
		'''Showing points on density profile that would be used for Chandra simulated observation'''
		fig,ax=plt.subplots()
		
		co=self.chandra_obs
		r1,r2=np.array(co['rad'])
		rho1,rho2=np.array(co['rho'])
		rbc=self.chandra_rb
		rho_rbc=self.chandra_extrap[0]

		ax.loglog([r1,r2],[rho1,rho2], 'rs')
		ax.loglog(self.radii, self.rho)
		ax.loglog(rbc, rho_rbc, 'g<')
		ax.loglog(self.rb, self.rho_profile(self.rb), 'k<')
		fig.suptitle(self.name+',{0},{1:3.2e},{2:3.2e}'.format(self.vw_extra/1.E5, self.chandra_mdot_ratio,self.mdot_bondi_ratio[0]))

		plt.close()
		return fig

	@property
	def chandra_mdot_bondi(self):
		'''mdot_bondi which would be inferred from the Chandra observation'''
		rho_rb, temp_rb=self.chandra_extrap
		cs_rb=gal_properties.cs(temp_rb,mu=self.mu,gamma=self.gamma)

		return gal_properties.mdot_bondi(self.params['M'],cs_rb,rho_rb)

	@property 
	def chandra_mdot_ratio(self):
		return self.chandra_mdot_bondi/self.mdot

	@property
	def kappa_cond(self):
		if not hasattr(self, 'cond_scheme'):
			self.set_param('cond_scheme','spitzer')
		if not hasattr(self, 'eps_cond'):
			self.eps_cond=0.

		if self.cond_scheme=='shcherba':
			return self.eps_cond*self.shcherba
		elif self.cond_scheme=='simple':
			return self.eps_cond*np.array([(2.*10**-6)*(1.E7)**(5./2.) for i in range(self.length)])
		else:
			return self.eps_cond*self.spitzer
		# else:
		# 	return np.zeros(self.length) 


	@property
	def f_cond_unsat(self):
		return np.array([self.kappa_cond[i]*self.get_spatial_deriv(i, 'temp') for i in range(0,self.length)])

	@property 
	def f_cond_sat(self):
		if not hasattr(self, 'phi_cond'):
			self.set_param('phi_cond',1.)
		return self.temp_deriv_signs*self.phi_cond*5.*self.rho*self.cs**3

	@property 
	def sigma_cond(self):
		sigma_cond=np.abs(self.f_cond_unsat[self.start:self.end+1]/self.f_cond_sat[self.start:self.end+1])
		start=[sigma_cond[0] for i in range(0,self.start)]
		end=[sigma_cond[-1] for i in range(self.end+1,self.length)]

		return np.concatenate([start, sigma_cond, end])

	# @property
	# def kappa_cond_eff(self):
	# 	return self.kappa_cond/(1.+self.sigma_cond)

	def _update_aux(self):
		self.kappa_cond_eff=self.kappa_cond/(1.+self.sigma_cond)
		
	@property 
	def temp_deriv_signs(self):
		temp_derivs=([self.get_spatial_deriv(i,'temp') for i in range(self.length)])
		return temp_derivs/np.abs(temp_derivs)

	@property
	def f_cond(self):
		'''conductive flux'''
		return (self.f_cond_unsat*self.f_cond_sat)/(self.f_cond_unsat+self.f_cond_sat)

	@property 
	def tcond_unsat(self):
		return abs((4.*self.rho*self.cs**2*self.radii**3)/(4.*np.pi*self.f_cond_unsat*self.radii**2))

	@property 
	def tcond_sat(self):
		'''Conductive time-scale obtained from the saturated conductive flux'''
		return abs((4.*self.rho*self.cs**2*self.radii**3)/(4.*np.pi*self.f_cond_sat*self.radii**2))

	@property 
	def cond_spitzer(self):
		return np.array([self.get_diffusion(i, 'spitzer', 'temp') for i in range(0, self.length)])

	@property 
	def cond_shcherba(self):
		return np.array([self.get_diffusion(i, 'shcherba', 'temp') for i in range(0, self.length)])

	def cond(self,i):
		if not(hasattr(self,'cond_simple')) or self.cond_simple==False:
			return self.get_diffusion(i, 'kappa_cond_eff', 'temp')
		else:
			return self.kappa_cond_eff[i]*self.get_spatial_deriv(i, 'temp', second=True) 

	@property 
	def cond_grid(self):
		np.array([self.cond(i) for i in range(self.length)])

	@property
	def cond_spitzer_ratio(self):
		return self.cond_spitzer[self.start:self.end]/self.heating_pos[self.start:self.end]

	@property 
	def cond_shcherba_ratio(self):
		return self.cond_shcherba[self.start:self.end]/self.heating_pos[self.start:self.end]

	@property
	def cond_plot(self):
		'''Plot conductivity vs. heating rate.'''
		fig,ax=plt.subplots()
		ax.loglog(self.radii[self.start:self.end],abs(self.cond_spitzer_ratio))
		ax.loglog(self.radii[self.start:self.end],abs(self.cond_shcherba_ratio))
		plt.close()

		return fig


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
	def from_dir(cls, name, loc, index=-1, rescale=1., rmin=None, rmax=None, gdata=None, length=None, params=False):
		init={}
		init_array=prepare_start(np.load(loc+'/save.npz')['a'][index])
		if rescale=='auto':
			a=96.5
			tmp=cls(name, gdata=gdata)
			rescale=tmp.rinf/init_array[0,0]/a
			print rescale

		init_array[:,0]=rescale*init_array[:,0]
		radii=init_array[:,0]
		if not rmin:
			init['rmin']=radii[0]
		else:
			init['rmin']=rmin
		if not rmax:
			init['rmax']=radii[-1]
		else:
			init['rmax']=rmax

		delta=np.diff(radii)
		delta_log=np.diff(np.log(radii))

		if np.allclose(np.diff(delta),[0.]):
			init['logr']=False
		else:
			init['logr']=True
		init['f_initial']=extrap1d_pow(interp1d(init_array[:,0], init_array[:,1:4], axis=0))
		if not length:
			init['length']=len(radii)
		else:
			init['length']=length

		gal=cls(name, init=init, gdata=gdata)
		if params:
			try:
				params=dill.load(open(loc+'/non_standard.p','rb'))
				[gal.set_param(param, params[param]) for param in params]
			except:
				pass 
		else:
			pass

		return gal

	@property
	def mstar_bulge(self):
		'''Get the mbulge from WM04 table'''
		params=nuker_params()
		return params[self.name]['M2']/0.006

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
		return self.sigma_interp(self.rinf)

	@property
	def rb(self):
		'''Bondi radius computed from G M/cs(rb)^2=rb'''
		f=lambda r:G*self.params['M']/(self.cs_profile(r))**2-r
		rb=fsolve(f, G*self.params['M']/self.cs_profile(self.rinf)**2)

		return rb

	def get_prop(self,field,r):
		return getattr(self, field)(r)

	@property
	def dens_pow_slope_rs(self):
		lrho_interp=interp1d(np.log(self.radii),self.log_rho)
		return derivative(lrho_interp, np.log(self.rs[0]), dx=self.delta_log[0])

	@property
	def rs_analytic(self):
		'''Analytic formula for the stagnation radius--normalized by the influence radius'''
		if self.stag_unique:
			A=(4.*self.gamma-(1+self.params['gamma'])*(self.gamma-1.))/(4.*(self.gamma-1.))
			eta=self.vw_extra/self.sigma_inf
			omega=self.M_enc_interp(self.rs[0])/self.params['M']
			dens_slope=np.abs(self.dens_pow_slope_rs)

			return 1./(dens_slope*eta**2)*((0.5)*(2*A-dens_slope)*(1+omega)-(2.-self.params['gamma'])/4.)

	@property 
	def rs_analytic_approx(self):
		return gal_properties.rs_analytic_approx(self.params['M'],self.vw_extra)

	@property 
	def rs_residual(self):
		'''Residual of the stagnation radius from the analytic result'''
		if self.stag_unique:
			return (self.rs_analytic-(self.rs[0]/self.rinf))/self.rs_analytic

	@property
	def vw_rs(self):
		if self.stag_unique:
			return (self.sigma_interp(self.rs[0])**2+self.vw_extra**2)**0.5

	@property
	def vw_rs_analytic(self):
		'''Approximation for the effective wind velocity at rs'''
		return self.vw_extra*(1.+(0.4**2*self.sigma_200**2/self.vw_extra_500**2))**0.5

	@property
	def rho_rs(self):
		'''Density at rs'''
		if self.stag_unique:
			return self.rho_profile(self.rs[0])

	@property
	def temp_rs(self):
		'''Temperature at the rs'''
		if self.stag_unique:
			return self.temp_profile(self.rs[0])

	@property
	def q_rs(self):
		'''Value of the source function at the stagnation radius'''
		if self.stag_unique:
			return self.q_interp(self.rs[0])

	@property
	def menc_rs(self):
		if self.stag_unique:
			return self.M_enc_interp(self.rs[0])

	@property
	def temp_rs_analytic(self):
		'''Analytic expression for the temperature at the stagnation radius'''
		if self.stag_unique:
			return ((self.gamma-1.)/self.gamma)*0.5*self.vw_rs**2*self.mu*mp/kb

	@property
	def temp_rs_residual(self):
		'''Residual of the temperature at the stagnation radius compared to the analytic result'''
		if self.stag_unique:
			return (self.temp_rs_analytic-self.temp_interp(self.rs))/self.temp_rs_analytic

	@property
	def rho_rs_analytic(self):
		'''Analytic estimate for the density at the stagnation radius'''
		if self.params['gamma']<0.2:
			return 2.5E-24*self.eta/(self.M_bh_8)**0.13/(self.vw_extra_500)
		else:
			return 5.5E-24*self.eta*self.vw_extra_500/(self.M_bh_8)**0.57

	@property
	def heating_pos_rs_analytic(self):
		if self.params['gamma']<0.2:
			return 3.2E-21*self.vw_extra_500**4.*self.eta/self.M_bh_8**1.14
		else:
			return 3.4E-21*self.vw_extra_500**6*self.eta/self.M_bh_8**1.57

	@property
	def menc_rs_analytic(self):
		if self.params['gamma']<0.2:
			return gal_properties.menc_rs_analytic_core(self.params['M'],self.vw_extra)
		else:
			return gal_properties.menc_rs_analytic(self.params['M'],self.vw_extra)

	@property 
	def eddr_analytic(self):
		if self.params['gamma']<0.2:
			return gal_properties.eddr_analytic_core(self.params['M'], self.vw_extra)
		else:
			return gal_properties.eddr_analytic(self.params['M'], self.vw_extra)

	@property
	def heating_pos_rs(self):
		return self.heating_pos_profile(self.rs[0])

	@property 
	def cooling_rs(self):
		return self.cooling_profile(self.rs[0])

	@property
	def hc_rs(self):
		'''Ratio of heating to cooling at the stagnation radius'''
		if self.stag_unique:
			return self.heating_pos_rs/self.cooling_rs

	@property
	def hc_rs_analytic(self):
		'''Analytic estimate for the ratio of the heating and cooling rates at rs'''
		if self.params['gamma']<0.2:
			return 20.*(self.vw_rs_analytic/5.E7)**7.4*self.mu**2.7/self.M_bh_8**0.86
		else: 
			return 4.8*(self.vw_rs_analytic/5.E7)**5.4*self.mu**2.7/self.M_bh_8**0.43
			
	@property
	def tde_table(self):
		'''Get crossing radius for jet'''
		m6=self.params['M']/(1.E6*M_sun)
		jet=tde_jet.Jet()
		jet.m6=m6

		r=integrate.ode(jet.vj)

		r.set_integrator('vode')
		r.set_initial_value(0., t=jet.delta).set_f_params(self.rho_profile)
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

		f=jet.rho(rc)/self.rho_profile(rc)
		gamma=jet.gamma_j*(1.+2.*jet.gamma_j*f**(-0.5))**(-0.5)   

		rc_col=Column([rc/pc], name=r'$r_c$', format=latex_exp.latex_exp)
		n_rc_col=Column([self.rho_profile(rc)/(self.mu*mp)], name=r'$n_{rc}$',format=latex_exp.latex_exp)
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

	@property 
	def vsig(self):
		vsig=ascii.read('vsig.csv', delimiter=',',names=['gal', 'vsig', 'eps','log_re'])
		vsig_idx=np.where(vsig['gal']==self.name)[0]
		if vsig['vsig'][vsig_idx] and vsig['eps'][vsig_idx] and vsig['log_re'][vsig_idx]:
			eps=vsig['eps'][vsig_idx[0]]
			re=10.**vsig['log_re'][vsig_idx[0]]*pc
			return vsig['vsig'][vsig_idx[0]]*(eps/(1.-eps))**0.5*(self.rs[0]/(0.5*re))**0.6
		else:
			return None

	@property
	def rcirc(self):
		'''Compute the circularization radius at the stagnation radius'''
		if self.stag_unique and self.vsig:
			return (self.vsig**2.*self.rs[0]**2/self.rinf)

	@property
	def rc_rss_ratio(self):
		'''Ratio of circularizion radius to inner sonic point'''
		if self.stag_unique and self.vsig:
			return self.rcirc/self.r_ss






























