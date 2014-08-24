
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
th=4.35*10**17
year=3.15569E7


def sigma_200(M):
	return ((M)/(1.48E8*M_sun))**(1./4.65)

def r_Ia(M):
	return 4.*(sigma_200(M))**-0.5*pc

def r_s(M, vw):
	'''Simplified analytic expression for the stagnation radius--given a particular bh mass and particular vw (not including sigma)'''
	return(7./4.)*G*M/((vw)**2./2.)

def l_edd(M):
	return 4.*np.pi*G*M*c/(0.4)

def mdot_edd(M, efficiency=0.1):
	return l_edd(M)/(efficiency*c**2)

def rb(M, cs):
	'''Bondi radius from the mass M and sound speed cs'''
	return G*M/cs**2

def mdot_bondi(M, cs, rho):
	return 4.*np.pi*0.25*(G*M)**2.*cs**-3*rho

def cs(temp, mu=1, gamma=5./3.):
	return (gamma*kb*temp/(mu*mp))**0.5


