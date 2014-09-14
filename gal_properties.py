
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
	'''M-sigma relationship taken from Wang and Merritt 2004'''
	return ((M)/(1.48E8*M_sun))**(1./4.65)

def r_Ia(M):
	return 4.*(sigma_200(M))**-0.5*pc

def rs_approx(M, vw):
	'''Simplified analytic expression for the stagnation radius--given a particular bh mass and particular vw (not including sigma)'''
	return(7./4.)*G*M/((vw)**2./2.)

def rs_approx_nond(eta):
	return 7./2.*eta**-2.

def vw_from_rs(M, rs):
	'''Inverse of rs_approx'''
	return ((7./2.)*G*M/(rs))**0.5

# def rho_rs(M, vw, nuk_gamma, eta=1):
# 	'''Analytic approximation to the density at the stagnation radius'''
# 	vw_extra_500=vw/5.E7
# 	M_bh_8=M/1.E8/M_sun
# 	if nuk_gamma<0.2:
# 		return 2.5E-24*eta/(M_bh_8)**0.13/(vw_extra_500)
# 	else:
# 		return 5.5E-24*eta*vw_extra_500/(M_bh_8)**0.57

def l_edd(M):
	return 4.*np.pi*G*M*c/(0.4)

def mdot_edd(M, efficiency=0.1):
	return l_edd(M)/(efficiency*c**2)

def rb(M, cs):
	'''Bondi radius from the mass M and sound speed cs'''
	return G*M/cs**2

def mdot_bondi(M, cs, rho):
	'''Bondi accretion rate'''
	return 4.*np.pi*0.25*(G*M)**2.*cs**-3*rho

def cs(temp, mu=1, gamma=5./3.):
	'''Sound speed'''
	return (gamma*kb*temp/(mu*mp))**0.5

def eta(t):
	'''Normalization of the mass source function as a function of stellar age'''
	return 0.11*(t/th)**-1.26

def rate_Ia(t):
	'''Rate of SNe Ia using delay time distribution from Maoz et al. '12'''
	return 0.03*(t/(1.E8*year))**(-1.12)*(1/year)*(1./(10.**10*M_sun))
	
def vw_eff_Ia(t):
	return ((2.*th*rate_Ia(t)*(1.E51))/eta(t))**0.5

def vw_eff_stars(t):
	if t<10.**7.5*year:
		return 1.E8
	else:
		return 1.E7

def vw_eff(t, M):
	r_Ia_0=r_Ia(M)
	vw_eff=np.array([vw_eff_stars(t), (vw_eff_Ia(t)**2+vw_eff_stars(t)**2)**0.5])
	rs=np.array([rs_approx(M, vw) for vw in vw_eff])
	if rs[0]<r_Ia_0:
		return vw_eff[0]
	elif rs[1]>r_Ia_0:
		return vw_eff[1]
	else:
		return vw_from_rs(M, r_Ia_0)

def be(r, rs=1.E18, M_bh=1.E8*M_sun, vw0=1.E8, M_enc0=0., rho0=0., beta=1.8, sigma=True, shell=True):
	'''Analytic expression for the Bernoulli parameter note the non-trivial gauge condition here.'''
	x=r/rs
	#from_s=4.*np.pi*G*rs**2*rho0*(3.-beta)*(x**(4.-2.*beta)-1.)/((1.-beta)*(4.-2.*beta)*(x**(3.-beta)-1.))
	if shell:
		from_s=-4.*np.pi*G*rs**2*rho0*(3.-beta)/(2.-beta)*(1./(x**(3.-beta)-1.))*((x**(3.-beta)-1.)/(3.-beta)-(x**(5.-2.*beta)-1)/(5.-2.*beta))
	else:
		from_s=0.

	if sigma:
		return (vw0**2/2.)-(G*M_bh/(2.*rs))*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/(2.*rs))*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)+from_s
	else:
		return (vw0**2/2.)-(G*M_bh/rs)*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/rs)*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)+from_s





