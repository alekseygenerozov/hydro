 
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

def epsilon_sharma(eddr):
	if eddr<1.E-4:
		return 2.6E-2*(eddr/(1.E-4))**0.9
	elif eddr<1.E-2:
		return 2.6E-2
	elif eddr<0.1:
		return 0.1*(eddr/0.1)**0.58
	else:
		return 0.1

def epsilon_sharma_2(eddr):
	if eddr>0.01:
		return 0.1
	else:	
		eff=np.genfromtxt('/Users/aleksey/Documents/papers/Sharma/middle.csv', delimiter=',')
		extrap=extrap1d_pow(interp1d(10.**eff[:,0], 10.**eff[:,1]))
		return extrap(eddr)
	
def lambda_c(temp):
	'''Power law approximation to cooling function for solar abundance plasma (taken from figure 34.1 of Draine)'''
	if temp>2.E7:
		return 2.3E-24*(temp/1.E6)**0.5
	else:
		return 1.1E-22*(temp/1.E6)**-0.7

def rinf(M):
	return 14.*(M/(1.E8*M_sun))**0.6*pc

def rb_core(M):
	'''scaling relationship for break radius from Lauer et al. 2007 sample'''
	return 106.*(M/(1.E8*M_sun))**0.39*pc

def rb_rinf_core(M):
	return rb_core(M)/rinf(M)

def gamma_fit(M):
	'''Average Nuker gamma based on Lauer et al. 2007 sample'''
	return 0.3*(M/(1.E8*M_sun))**(-0.24)

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
	'''radius for which time between Ias is equal to dynamical time'''
	return 35.*pc*(M/(10.**8*M_sun))**-0.1

def zeta(M, vw):
	'''normalized heating rate'''
	return (vw**2.+3.*sigma(M)**2.)**0.5/(3.**0.5*sigma(M))


def zeta_norm(M, vw, gamma=None, rb_rinf=None):
	'''normalized heating rate

	:param gamma: nuker gamma if not provided use 0.3*M8^-0.24
	:param rb_rinf: break radius divided by the influence radius. If not provided 
	use scaling relationship (rb_rinf_core) for cores (gamma<0.3) and 100 pc/rinf for cusps

	'''
	if not gamma:
		gamma=gamma_fit(M)
	if not rb_rinf:
		if gamma<0.3:
			rb_rinf=rb_rinf_core(M)
		else:
			rb_rinf=100.*pc/rinf(M)

	z=zeta(M, vw)
	zc=zeta_c_fit(gamma, rb_rinf)

	return z/zc

def zeta_c_fit(gamma, rb_rinf):
	'''fit to critical heating rate for thermal instability'''
	return (rb_rinf)**(0.5*(1.-gamma))

def vw_crit(M, gamma=None, rb_rinf=None, sig=None):
	'''critical heating rate below which rs should run away. Corresponds to zeta_c_fit'''
	if not gamma:
		gamma=gamma_fit(M)
	#Scaling relations for rb if this is not specified...
	if not rb_rinf:
		if gamma<0.3:
			rb_rinf=rb_rinf_core(M)
		else:
			rb_rinf=100.*pc/rinf(M)
	#Critical zeta		
	zc=zeta_c_fit(gamma, rb_rinf)
	#If sigma is not specified use the M-sigma relation
	if not sig:
		sig=sigma(M)
	sigma_0=(3.)**0.5*sigma(M)

	return (zc**2.-1.)**0.5*sigma_0

def zeta_anal(x,  gamma=1., nu=None):
	'''This function explicitly calculates zeta given x (rs/rinf) and power law density slope, nu.'''
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

def vw_ti_core(M, eta):
	'''thermal instability criterion for cusp galaxies--assumes gamma=0.8'''
	M8=M/10.**8/M_sun
	return 2.814679E7*M8**0.1055555556*(eta/0.02)**0.1388889

def vw_ti_cusp(M, eta):
	'''thermal instability criterion for cusp galaxies--assumes gamma=0.8'''
	M8=M/10.**8/M_sun
	return 2.425311E7*M8**0.08275862*(eta/0.02)**0.17241

def vw_ti_Ia_cusp(M, eta):
	'''thermal instability criterion for core galaxies w/ Ia heating--assumes gamma=0.8'''
	M8=M/10.**8/M_sun
	return 3.58487E7*(eta/0.02)**0.2941/(M8**0.246)

def vw_ti_Ia_core(M, eta):
	'''thermal instability criterion for core galaxies w/ Ia heating--assumes gamma=0.1'''
	M8=M/10.**8/M_sun
	return (3.645E7*(eta/0.02)**0.2941)/(M8**0.390)

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





	

