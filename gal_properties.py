 
import astropy.constants as const
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import warnings

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
L_sun=const.L_sun.cgs.value
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

def l_edd(M):
	'''Eddington luminosity'''
	return 4.*np.pi*G*M*c/(0.4)

def mdot_edd(M, efficiency=0.1):
	'''Eddington accretion rate'''
	return l_edd(M)/(efficiency*c**2)

def Lv(Mv):
	'''Luminosity from mag'''
	Mv_sun=4.83
	return L_sun*10.**((Mv_sun-Mv)/2.5)

##Convert from magnitudes per arcsec^2 to luminosities per parsec^2
def mub_to_Ib(mub):
	'''Convert from magnitudes per arcsec^2 to luminosities per pc^2'''
	Msun=4.83
	return 10**(-(mub-Msun-21.572)/2.5)

def M_mvbulge(Mv):
	'''Mbh from the bulge luminosity'''
	return 10**9.23*(Lv(Mv)/(10.**11*L_sun))**1.11*M_sun

def uppsilon_vband(sigma, reff, Mv):
	'''mass-to-light ratio in the vband'''
	return 2.*sigma**2.*reff/(G*Lv(Mv)/2.)*L_sun/M_sun

def uppsilon_vband_mag(Mv):
	'''scaling relation for mass-to-light ratio in V band'''
	return 4.9*(Lv(Mv)/(10.**10*L_sun))**0.18

def epsilon_adaf_delta_5(eddr, alpha=0.1):
	'''efficiciency for ADAF as a function of eddington ratio, eddr,
	from Table 1 and equation 11 Xie & Yuan 2012. delta=0.5 (see Xie & Yuan).

	:param alpha: Shakura-Sunyaev alpha parameter
	'''
	if eddr<2.9E-5:
		eps0=1.58
		a=0.65
	elif eddr<3.3E-3:
		eps0=0.055
		a=0.076
	elif eddr<5.3E-3:
		eps0=0.1
		a=0.12
	else:
		eps0=0.08
		a=0.
	# print a, eddr
	return eps0*(alpha/0.1)**0.5*(eddr/0.01)**a

def epsilon_adaf_delta_1(eddr, alpha=0.1):
	'''efficiciency for ADAF as a function of eddington ratio, eddr,
	from Table 1 and equation 11 Xie & Yuan 2012. delta=0.1 (see Xie & Yuan).

	:param alpha: Shakura-Sunyaev alpha parameter
	'''
	if eddr<9.4E-5:
		eps0=0.12
		a=0.59
	elif eddr<5.E-3:
		eps0=0.026
		a=0.27
	elif eddr<5.9E-3:
		eps0=0.50
		a=4.53
	else:
		eps0=0.08
		a=0.

	return eps0*(alpha/0.1)**0.5*(eddr/0.01)**a

def epsilon_sharma(eddr):
	if eddr<1.E-4:
	    return 7.1E-6*(eddr/(1.E-8))**0.9
	elif eddr<1.E-2:
	    return 2.6E-2
	else:
	    return 0.1
	
def lambda_c(temp):
	'''Power law approximation to cooling function for solar abundance plasma (taken from figure 34.1 of Draine)'''
	if temp>2.E7:
		return 2.3E-24*(temp/1.E6)**0.5
	else:
		return 1.1E-22*(temp/1.E6)**-0.7

def rinf(M):
	return 14.*(M/(1.E8*M_sun))**0.6*pc

def rbreakCore(M):
	return 106.*(M/(1.E8*M_sun))**0.39*pc

def sigma_200(M):
	'''M-sigma from McConnell et al. 2011'''
	return ((M)/(2.E8*M_sun))**(1./5.1)

def sigma(M):
	return 2.E7*sigma_200(M)

def M_sigma(sigma_200):
	'''Inverse of the above M-sigma relationship'''
	return (2.E8*M_sun)*(sigma_200)**5.1

def vff(M, r):
	return (G*M/r)**0.5

def tff(M, r):
	return r/vff(M, r)

def r_Ia(M):
	return 38.*pc*(M/(10.**8*M_sun))**-0.1

def zeta(M, vw):
	return (vw**2.+3.*sigma(M)**2.)/(3.**0.5*sigma(M))

def zeta_c_fit(gamma, rbrinf):
	return (rbrinf)**(0.5*(1.-gamma))

def zeta_anal(x,  gamma=1., nu=None):
	'''This function explicitly calculates zeta given for x and power law density slope, nu.'''
	if not nu:
		nu=dens_slope(gamma)
	return ((1.0/(3.0*x*nu))*(x**(2.-gamma)*4.0+(13.+8.*gamma)/(4.+2.*gamma)-nu*3./(2.+gamma)))**0.5

def zeta_min(gamma=1, nu=None):
	if not nu:
		nu=dens_slope(gamma)
	return minimize(lambda x: zeta_anal(x, gamma=gamma, nu=nu), 1.)['fun']

def dens_slope(gamma):
	'''Approximate expression for the density slope at the stagnation radius'''
	return -(1./6.*(1.-4.*(1+gamma)))

def temp_rs_tilde(M, vw,  mu=0.62):
	return 0.4*mu*mp*vw**2/kb/2.

def temp_rs(M, vw, gamma=1., mu=0.62):
	return 0.4*mu*mp*vw**2/kb/2.*(13.+8.*gamma)/(13.+8.*gamma-6.*dens_slope(gamma))

def rs_approx(M, vw, gamma=1., nu=None):
	'''Simplified analytic expression for the stagnation radius--given a particular bh mass and particular vw.
	This neglects the stellar mass term. Note that we have incorporate the effect of the constant stellar velocity dispersion.'''
	sigma_0=(3.0)**0.5*sigma(M)
	if not nu:
		nu=dens_slope(gamma)
	return G*M/(nu*(vw**2.+sigma_0**2))*((13.+8.*gamma)/(4.+2.*gamma)-nu*(3./(2.+gamma)))

def rs_rinf_approx(z, gamma=1., nu=None):
	if not nu:
		nu=dens_slope(gamma)
	return (1./(3*nu*z**2.))*((13.+8.*gamma)/(4.+2.*gamma)-nu*(3./(2.+gamma)))

def rs_rinf_exact(z, gamma=1., nu=None):
	if not nu:
		nu=dens_slope(gamma)
	if z<zeta_min(gamma, nu):
		print z, gamma, nu,zeta_min(gamma, nu)
		print 'no solution for specified zeta'
		return np.nan

	return fsolve(lambda x:zeta_anal(x, gamma=gamma, nu=nu)-z, rs_rinf_approx(z, gamma=gamma, nu=nu))

def M_enc_rs_analytic(M, vw, gamma=1.):
	'''Stellar mass enclosed inside of the stagnation radius'''
	return M*(rs_approx(M, vw, gamma=gamma)/rinf(M))**(2.-gamma)

def	M_enc_rs_analytic_exact(M, vw, gamma=1.):
	'''Stellar mass enclosed inside of the stagnation radius--more exact'''
	sigma_0=(3.)**0.5*sigma(M)
	z=(vw**2.+sigma_0**2.)**0.5/sigma_0

	return M*(rs_rinf_exact(z, gamma=gamma))**(2.-gamma)

def mdot_analytic(M, vw, gamma=1.,eta=0.1):
	return eta*M_enc_rs_analytic(M, vw, gamma=gamma)/th

def mdot_analytic_exact(M, vw, gamma=1.,eta=0.1):
	return eta*M_enc_rs_analytic_exact(M, vw, gamma=gamma)/th

def eddr_analytic(M, vw, gamma=1., eta=0.1):
	'''Analytic expression for the Eddington ratio--for cuspy galaxies'''
	return mdot_analytic(M, vw, gamma=gamma, eta=eta)/mdot_edd(M)

def eddr_analytic_exact(M, vw, gamma=1., eta=0.1):
	'''Analytic expression for the Eddington ratio--for cuspy galaxies'''
	return mdot_analytic_exact(M, vw, gamma=gamma, eta=eta)/mdot_edd(M)

def rho_rs_analytic(M, vw, gamma=1,eta=0.1):
	'''Analytic expression for gas density at the stagnation radius rs'''
	rs=rs_approx(M, vw, gamma=gamma)
	return mdot_analytic(M, vw, gamma=gamma, eta=eta)/(4./3.*np.pi*rs_approx(M,vw,gamma=gamma)**2*vff(M,rs))

def rho_stars_rs_analytic(M, vw, gamma=1.):
	'''Analytic expression for the stellar density at the stagnation radius rs'''
	return M*(2.-gamma)/(4.*np.pi*rinf(M)**3.)*(rs_approx(M,vw,gamma=gamma)/rinf(M))**(-1.-gamma)

def q_rs_analytic(M, vw, gamma=1., eta=0.1):
	'''Analytic estimate for the mass source function and the stagnation radius rs'''
	return eta*rho_stars_rs_analytic(M, vw, gamma=gamma)/th

def tcool_rs(M, vw, gamma=1., mu=0.62, eta=0.1):
	'''Analytic estimate  for the ratio of the cooling time at the stagnation radius rs''' 
	temp_rs=temp_rs(M, vw, gamma=gamma, mu=mu)
	return 1.5*kb*temp_rs/(rho_rs_analytic(gamma,eta,vw,rbrinf)/(mu*mp)*lambda_c(temp_rs))

def tcool_tff_rs(M, vw, mu=0.62, gamma=1., eta=0.1):
	'''Analytic estimate for the ratio of the cooling to the free-fall time at the stagnation radius.'''
	rs1=rs_approx(M,vw,gamma=gamma)
	return tcool_rs(M, vw, mu, gamma, eta, rbrinf)/tff(M,rs1)

def rbondi(M, cs):
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
	'''vw from Ia's in impulsive limit'''
	eps1a=0.4
	return ((2.*th*rate_Ia(t)*(eps1a*1.E51))/eta(t))**0.5





	

