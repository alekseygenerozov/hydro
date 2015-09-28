import astropy.constants as const
import numpy as np
from scipy import integrate

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value

def vj(r,rho_interp):
	f=rho(r)/rho_interp(r)
	beta_sh=(1.-(1./jet.gamma_j)**2.-(2./gamma_j)*(f**-0.5))**0.5
	return jet.beta_j/beta_sh

class Jet:
	def __init__(self):
		self.ms=1.
		self.m6=1.
		self.eta=0.1
		self.theta=0.1
		self.gamma_j=10.
		self.beta_j=(1.-1./self.gamma_j**2)**0.5

	@property
	def tfb(self):
		return  3.5E6*np.sqrt(self.m6)*np.sqrt(self.ms)

	@property
	def delta(self):
		return c*self.tfb

	@property
	def mdot_peak(self):
		return 1.85E26*np.sqrt(self.ms)/np.sqrt(self.m6)

	@property
	def lj(self):
		return 1.67E47*self.eta*np.sqrt(self.ms)/np.sqrt(self.m6)

	def rho(self,r):
		return self.lj/(np.pi*self.theta**2*c**3*self.gamma_j**2*r**2)

	def vj(self,r,rho_profile):
		f=self.rho(r)/rho_profile(r)
		beta_sh=(1.-(1./self.gamma_sh(f)**2.))**0.5
		return self.beta_j/beta_sh

	def gamma_sh(self, f):
		return self.gamma_j*(1.+2.*self.gamma_j*f**(-0.5))**(-0.5) 

	def final_gamma(self, rho_profile):
		'''Calculate the final Lorentz factor of a jet at the time of reverse shock crossing'''
		r=integrate.ode(self.vj)
		r.set_integrator('vode')
		r.set_initial_value(0., t=self.delta).set_f_params(rho_profile)
		time=0.

		try:
			while r.y<r.t:
				old=r.y[0]
				r.integrate(r.t+0.01*pc)
				new=r.y[0]
				time+=(new-old)/(self.beta_j*c)
			rc=r.y[0]
		except Exception as inst:
			print inst
			rc=np.nan

		f=self.rho(rc)/rho_profile(rc)
		gamma=self.gamma_sh(f) 

		return gamma

		




