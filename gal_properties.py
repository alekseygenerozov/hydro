
import astropy.constants as const
import numpy as np
from scipy.optimize import fsolve

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


def lambda_c(temp):
	'''Power law approximation to cooling function for solar abundance plasma (taken from figure 34.1 of Draine)'''
	if temp>2.E7:
		return 2.3E-24*(temp/1.E6)**0.5
	else:
		return 1.1E-22*(temp/1.E6)**-0.7

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

def rs_implicit(zeta, dens_slope=1., gamma=1., full_output=False):
	'''solve implcit equation for rs, for a given stellar density profile and density slope'''
	f=lambda x:x-(0.5/(dens_slope*zeta**2.))*(x**(2.-gamma)*(4.5-0.5*gamma-dens_slope)+(3.5-dens_slope))
	return fsolve(f, 7./(4.*zeta**2*dens_slope),full_output=full_output)

def zeta(x, dens_slope=1., gamma=1.):
	'''Above solve implicit equation for rs/rinf given a zeta. This function explicitly calculates zeta given an x'''
	return ((x**(2.-gamma)*(4.5-0.5*gamma-0.5*dens_slope)+(3.5-dens_slope))/(2.*dens_slope*x))**0.5

def dens_slope(gamma):
	'''Approximate expression for the density slope at the stagnation radius'''
	return -(1./6.*(1.-4.*(1+gamma)))

def rs_approx(M, vw, gamma=1.):
	'''Simplified analytic expression for the stagnation radius--given a particular bh mass and particular vw.
	This assumes sigma is dominated by black holem and hence one gets the funny looking 5/2 out front'''
	return 5./2.*G*M/((vw)**2.*dens_slope(gamma))

def rs_approx_t(t, M, gamma=1.):
	return rs_approx(M, vw_eff(t, M), gamma=gamma)

# def rs_approx_rinf(zeta):
# 	'''Analytic approximation for the ratio of rs to rinf'''
# 	return 5./(4.*zeta**2)

def rs_r_Ia(t, M, gamma=1.):
	return rs_approx_t(t, M, gamma=gamma)/r_Ia(t,M)

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
	return (1.+(0.12*M8**0.4/vw500**2.))**0.5

def M_enc_rs_analytic(M, vw, gamma=1.):
	return M*(rs_approx(M, vw, gamma=gamma)/rinf(M))**(2.-gamma)

def mdot_analytic(M, vw, gamma=1.,eta=0.1):
	return eta*M_enc_rs_analytic(M, vw, gamma=gamma)/th

def eddr_analytic(M, vw, gamma=1., eta=0.1):
	'''Analytic expression for the Eddington ratio--for cuspy galaxies'''
	return mdot_analytic(M, vw, gamma=gamma, eta=eta)/mdot_edd(M)

def vff(M, r):
	return (2*G*M/r)**0.5

def tff(M, r):
	return r/vff(M, r)

# def vff_rs(M,vw):
# 	return vff(M, rs_approx(M,vw))

# def tff_rs(M,vw):
# 	return rs_approx(M,vw)/vff_rs(M,vw)

def temp_rs(M, vw, mu=0.62):
	gamma=5./3.
	return (gamma-1.)/gamma*mu*mp*vw**2*xi(M,vw)**2

def temp_approx_0(M, vw, r, mu=0.62, gamma=1.):
	rs=rs_approx(M, vw, gamma)

	x=r/rs
	gammaf=(2.-gamma)/(1.-gamma)
	f=1.-gammaf*(x**(1.-gamma)-1)/(x**(1.-gamma)-1./x)
	v0=vff(M,r)/2.
	cs2_approx=vw**2./2.+v0**2*(1+f)

	return 0.4*cs2_approx*(mu*mp)/kb

def temp_approx(M, vw, r, mu=0.62, gamma=1.):
	rs=rs_approx(M,vw,gamma=gamma)

	x=r/rs
	gammaf=(2.-gamma)/(1.-gamma)
	f=1.-gammaf*(x**(1.-gamma)-1)/(x**(1.-gamma)-1./x)
	v0=vff(M,r)/2.

	cs2_approx=vw**2./2.+v0**2*(1+f-f**2)

	return 0.4*cs2_approx*(mu*mp)/kb

def vel_approx(M, vw, r, gamma=1., rs=None):
	rs=rs_approx(M, vw, gamma=gamma)

	x=r/rs
	v0=vff(M,r)/2.
	gammaf=(2.-gamma)/(1.-gamma)
	f=(1.-gammaf*(x**(1.-gamma)-1)/(x**(1.-gamma)-1./x))
	v0=vff(M,r)/(2.)**0.5

	return v0*(f**2)**0.5

def rho_approx(M, vw, r, gamma=1., rs=None, eta=0.1):
	rs=rs_approx(M,vw,gamma=gamma)

	x=r/rs
	beta=1+gamma
	##Note that the same approximation for the stagnation radius, rs, should be used everywhere (this is currently not the case!)
	q0=q_rs_analytic(M,vw, gamma=gamma, eta=eta)
	#q0=M*(2.-gamma)/(4.*np.pi*rinf(M)**3.)*(rs/rinf(M))**(-1.-gamma)

	return q0*tff(M, rs)/(3.-beta)*(1./x**1.5)*((x**(3.-beta)-1.)**2*(2.-beta))/((2.-beta)*(x**(3.-beta)-1.)-(3.-beta)*x*(x**(2.-beta)-1.))


def rho_rs_analytic(M, vw, gamma=1,eta=0.1):
	rs=rs_approx(M, vw, gamma=gamma)
	return mdot_analytic(M, vw, gamma=gamma, eta=eta)/(4./3.*np.pi*rs_approx(M,vw,gamma=gamma)**2*vff(M,rs))

def rho_stars_rs_analytic(M, vw, gamma=1):
	return M*(2.-gamma)/(4.*np.pi*rinf(M)**3.)*(rs_approx(M,vw,gamma=gamma)/rinf(M))**(-1.-gamma)

def q_rs_analytic(M, vw, gamma=1., eta=0.1):
	return eta*rho_stars_rs_analytic(M, vw, gamma)/th

def tcool_rs(M, vw, mu=0.62, gamma=1., eta=0.1):
	temp_rs=temp_rs(M, vw, mu)
	return 1.5*kb*temp_rs/(rho_rs_analytic/(mu*mp)*lambda_c(temp_rs))

def tcool_tff_rs(M, vw, mu=0.62, gamma=1., eta=0.1):
	return tcool_rs(M, vw, mu, gamma, eta)/tff_rs(M, vw)

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
	
def rho_rs(M, vw):
	return 3.95E-24*(vw/5.E7)*(M/(1.E8*M_sun))**-0.56
 
def vw_eff_Ia(t):
	eps1a=0.4
	return ((2.*th*rate_Ia(t)*(eps1a*1.E51))/eta(t))**0.5

def vw_eff_imp(t, M):
	r_Ia_0=r_Ia(t,M)
	vw_eff=np.array([vw_eff_stars(t), (vw_eff_Ia(t)**2+vw_eff_stars(t)**2)**0.5])
	rs=np.array([rs_approx(M, vw, gamma=gamma) for vw in vw_eff])
	if rs[0]<r_Ia_0:
		return vw_eff[0]
	elif rs[1]>r_Ia_0:
		return vw_eff[1]
	else:
		return vw_eff[0]
		# return vw_from_rs(M, r_Ia_0)

# def be(r, rs=1.E18, M_bh=1.E8*M_sun, vw0=1.E8, M_enc0=0., rho0=0., beta=1.8, sigma=True, shell=True):
# 	'''Analytic expression for the Bernoulli parameter note the non-trivial gauge condition here.'''
# 	x=r/rs
# 	if shell:
# 		from_s=-4.*np.pi*G*rs**2*rho0*(3.-beta)/(2.-beta)*(1./(x**(3.-beta)-1.))*((x**(3.-beta)-1.)/(3.-beta)-(x**(5.-2.*beta)-1)/(5.-2.*beta))
# 	else:
# 		from_s=0.

# 	if sigma:
# 		return (vw0**2/2.)-(G*M_bh/(2.*rs))*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/(2.*rs))*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)+from_s
# 	else:
# 		return (vw0**2/2.)-(G*M_bh/rs)*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/rs)*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)+from_s





