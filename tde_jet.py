import astropy.constants as const
import numpy as np

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value

def vj(r,jet,rho_interp):
    f=jet.rho(r*pc)/self.rho_interp(r)
    beta_sh=(1.-(1./jet.gamma_j)**2.-(2./gamma_j)*(f**-0.5))**0.5
    return jet.beta_j/beta_sh

class Jet:
	def __init__(self):
		self.ms=1.
		self.m6=1.
		self.eta=0.1
		self.theta=0.1
		self.gamma_j=10.
		self.beta_j=1.-1./self.gamma_j**2

	@property
	def tfb(self):
		return  3.5E6*np.sqrt(m6)*np.sqrt(ms)
	@property
	def delta(self):
		self.delta=c*self.tfb
	@property
	def mdot_peak(self):
		self.mdot_peak=1.85E26*np.sqrt(ms)/np.sqrt(m6)
	@property
	def lf(self):
		self.lj=1.67E47*eta*np.sqrt(ms)/np.sqrt(m6)

	def rho(self,r):
		return self.lj/(np.pi*self.theta**2*c**3*self.gamma_j**2*r**2)






