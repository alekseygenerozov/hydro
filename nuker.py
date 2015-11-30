#!/usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.optimize import fsolve
from astropy.io import ascii
import astropy.constants as const

import warnings
from scipy.special import gamma
import scipy.optimize as opt
from scipy.interpolate import interp1d

import cgs_const as cgs

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value
#Hubble time
th=4.35*10**17
#params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7.)

def nuker(r, Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., **kwargs):
	return Ib*2.**((beta-gamma)/alpha)*(rb/r)**gamma*(1.+(r/rb)**alpha)**((gamma-beta)/alpha)

##Derivative of nuker parameterized surface brightness profile
def nuker_prime(r, Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., **kwargs):
	return -((2**((beta - gamma)/alpha)*Ib*(gamma + beta*(r/rb)**alpha)*(rb/r)**gamma)/ (r*(1 + (r/rb)**alpha)**((alpha + beta - gamma)/alpha)))
    
##Return the inverse abel transform of a function 
def inverse_abel(i_prime, r, **kwargs):
	f=lambda y: i_prime(y, **kwargs)/np.sqrt(y**2-r**2)
	return -(1/np.pi)*integrate.quad(f, r, np.inf)[0]

##Convert from magnitudes per arcsec^2 to luminosities per parsec^2
def mub_to_Ib(mub):
	Msun=4.83
	return 10**(-(mub-Msun-21.572)/2.5)

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

##Reads Wang & Merritt parameter table into python dictionary
def nuker_params():
    table=ascii.read('wm')
    galaxies=dict()
    for i in range(len(table)):
		d=dict()
		d['Uv']=table[i]['$\\Upsilon_V $']
		d['alpha']=table[i]['$\\alpha$']
		d['beta']=table[i]['$\\beta$']
		d['gamma']=table[i]['$\\Gamma$']
		d['rb']=10.**table[i]['$\log_{10}(r_{b})$']
		d['Ib']=mub_to_Ib(table[i]['$\mu_b$'])
		d['mub']=table[i]['$\mu_b$']
		d['d']=table[i]['Distance']
		d['M']=M_sun*10.**table[i]['$\\log_{10}(M_{\\bullet}/M_{\\odot})$\\tablenotemark{e}']
		galaxies[table[i]['Name']]=d

	return galaxies

class Galaxy(object):
	"""Class to store info about Nuker galaxies"""
	def __init__(self, gname, gdata):
		try:
			self.params=gdata[gname]
		except KeyError:
			print 'Error! '+gname+' is not in catalog!'
			raise
		self.name=gname
		self.sigma_0=(3.)**0.5*gal_properties.sigma_200(self.params['M'])*2.0E7

		self.rmin=1.0e-3
		self.rmax=1.0e5

		def rho_stars(self,r):
		'''Stellar density
		:param r: radius 
		'''
		rpc=r/pc
		if rpc<self.rmin_star or rpc>self.rmax_star:
			return 0.
		else:
			return M_sun*self.params['Uv']*inverse_abel(nuker_prime, rpc, **self.params)/pc**3

		def _get_rho_stars_interp(self):
			_rho_stars_rad=np.logspace(np.log10(self.rmin_star), np.log10(self.rmax_star),1000)*pc
			_rho_stars_grid=[self.rho_stars(r) for r in _rho_stars_rad]
			return interp1d(np.log(_rho_stars_rad),np.log(_rho_stars_grid))

		def _rho_stars_interp(self,r):
		interp=self._get_rho_stars_interp()
		try:
			if np.isnan(interp(np.log(r))):
				return 0.
			else:
		    	return np.exp(interp(np.log(r)))
		except ValueError:
			return 0.

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

		def phi_s(self, r):
			rpc=r/pc
			return (-G*self.M_enc(r)/r)-4.*np.pi*G*integrate.quad(lambda r1:self._rho_stars_interp(r1*pc)*r1*pc**3, rpc, self.rmax_star)[0]/pc

		def phi_bh(self, r):
			'''Black hole potential'''
			return cgs.G*self.params['M']/r

		def q(self, r):
			'''Source term representing mass loss from stellar winds'''
			return self.rho_stars(r)/th

		def sigma(self, r):
			((3./(self.params['gamma']+2.))*G*(self.params['M'])/self.radii+self.eps_stellar_heating*self.sigma_0**2)**0.5


