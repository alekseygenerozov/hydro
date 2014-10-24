
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


def rinf(M):
	return 14.*(M/(1.E8*M_sun))**0.6*pc

def sigma_200(M):
	'''M-sigma relationship taken from Wang and Merritt 2004; should be updated to a more modern version (e.g. the one from Gulletkin)'''
	return ((M)/(2.E8*M_sun))**(1./5.1)

def M_sigma(sigma_200):
	'''Inverse of the above M-sigma relationship'''
	return (2.E8*M_sun)*(sigma_200)**5.1

def r_Ia(t,M):
	return (G/(sigma_200(M)*2.E7*rate_Ia(t)))**0.5

def rs_approx(M, vw, correction=False):
	'''Simplified analytic expression for the stagnation radius--given a particular bh mass and particular vw (not including sigma)'''
	if correction:
		rs=7./4.*G*M/(xi(M,vw)**2.*(vw)**2./2.)
	else:
		rs=7./4.*G*M/((vw)**2./2.)

	return rs

def rs_approx_t(t,M,correction=False):
	return rs_approx(M, vw_eff(t, M, correction), correction)

def rs_r_Ia(t,M, correction=False):
	return rs_approx_t(t,M,correction)/r_Ia(t,M)

# def rs_approx_2(M, vw):
# 	return (7./4.)*G*M/(xi(M,vw)*vw**2./2.-2.*(sigma_200(M))**2.*(2.E7)**2)

# def rs_approx_nond(eta):
# 	return 7./4.*eta**-2.

def vw_from_rs(M, rs):
	'''Inverse of rs_approx'''
	return ((7./2.)*G*M/(rs))**0.5

def l_edd(M):
	return 4.*np.pi*G*M*c/(0.4)

def mdot_edd(M, efficiency=0.1):
	return l_edd(M)/(efficiency*c**2)

def xi(M, vw):
	'''Correction to wind velocity to account for the contribution of the stellar velocity dispersion'''
	M8=(M/(1.E8*M_sun))
	vw500=vw/5.E7
	return (1.+(0.14*M8**0.43/vw500**2.))**0.5

def menc_rs_analytic(M, vw, correction=True):
	Menc=9.41E40*(M/(1.E8*M_sun))**1.43*(vw/5.E7)**-2.
	if correction:
		Menc=Menc*xi(M,vw)**-2.
	return Menc

def menc_rs_analytic_core(M, vw, correction=True):
	Menc=4.45E40*(M/(1.E8*M_sun))**1.86*(vw/5.E7)**-4.
	if correction:
		Menc=Menc*xi(M,vw)**-4.
	return Menc

def eddr_analytic(M, vw, eta=1., correction=True):
	'''Analytic expression for the Eddington ratio--for cuspy galaxies'''
	eddr=1.56E-3*(M/(1.E8*M_sun))**0.43*(vw/5.E7)**-2.
	if correction:
		eddr=eddr*xi(M,vw)**-2.
	return eddr

def eddr_analytic_core(M, vw, eta=1., correction=True):
	eddr=7.4E-4*(M/(1.E8*M_sun))**0.86*(vw/5.E7)**-4.
	if correction:
		eddr=eddr*xi(M,vw)**-2.
	return eddr

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

def vw_eff_stars(t):
	if t<10.**7.5*year:
		return 1.E8
	else:
		return 1.E7

def tcool_tff_rs(t, M,correction=False):
	return 10.*(xi(M,vw_eff(t,M,correction)))**5.4*(vw_eff(t, M,correction)/(5.E7))**5.4*(M/(1.E8*M_sun))**(-0.4)
	#return 10.*(vw_eff(t, M)/(5.E7))**5.4*(M/(1.E8*M_sun))**(-0.43)
	
def rho_rs(M, vw):
	return 3.95E-24*(vw/5.E7)*(M/(1.E8*M_sun))**-0.56

def vw_eff_Ia(t):
	eps1a=0.4
	return ((2.*th*rate_Ia(t)*(eps1a*1.E51))/eta(t))**0.5

def vw_eff(t, M, correction=False):
	r_Ia_0=r_Ia(t,M)
	vw_eff=np.array([vw_eff_stars(t), (vw_eff_Ia(t)**2+vw_eff_stars(t)**2)**0.5])
	rs=np.array([rs_approx(M, vw, correction) for vw in vw_eff])
	if rs[0]<r_Ia_0:
		return vw_eff[0]
	elif rs[1]>r_Ia_0:
		return vw_eff[1]
	else:
		return vw_from_rs(M, r_Ia_0)

def t_sigma_Ia(M, eps1a=0.4):
	return 4.5E12*(M/1.E8/M_sun)**2.8/eps1a**7.1

def t_rs_Ia(M, eps1a=0.4):
	return 4.73E16*(M/1.E8/M_sun)**-1.57*eps1a**-1.43

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





