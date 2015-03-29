
import astropy.constants as const
import numpy as np


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


def a(gamma=1):
	return 3./(gamma+2.)

def h(x, gamma=1.):
	return  (2.*gamma+1)/(a(gamma)*(gamma+2.))*(1.-(2.-gamma)/(1.-gamma)*(x**(1.-gamma)-1.)/(x**(1.-gamma)-1./x))

def temp_approx_0(M, vw, r, mu=0.62, gamma=1.):
	rs=rs_approx(M, vw, gamma)
	sigma_0=3.0**0.5*sigma_200(M)*2.0E7
	x=r/rs
	cs2_approx=(vw**2.+sigma_0**2.)/2.+a(gamma)*0.5*vff(M,r)**2*((3./(gamma+2.)/a(gamma))+h(x,gamma))

	return 0.4*cs2_approx*(mu*mp)/kb

def temp_approx(M, vw, r, mu=0.62, gamma=1.):
	rs=rs_approx(M, vw, gamma)
	sigma_0=3.0**0.5*sigma_200(M)*2.0E7
	x=r/rs
	cs2_approx=(vw**2.+sigma_0**2.)/2.+a(gamma)*0.5*vff(M,r)**2*((3./(gamma+2.)/a(gamma))+h(x,gamma)-h(x,gamma)**2.)

	return 0.4*cs2_approx*(mu*mp)/kb

def vel_approx(M, vw, r, gamma=1.):
	rs=rs_approx(M, vw, gamma=gamma)
	sigma_0=3.0**0.5*sigma_200(M)*2.0E7
	x=r/rs
	
	return (a(gamma))**0.5*vff(M,r)*abs(h(x,gamma))

def rho_approx(M, vw, r, gamma=1., eta=0.1):
	rs=rs_approx(M,vw,gamma=gamma)
	x=r/rs
	delta=1+gamma
	##Note that the same approximation for the stagnation radius, rs, should be used everywhere (this is currently not the case!)
	q0=q_rs_analytic(M,vw, gamma=gamma, eta=eta)
	return  -q0*tff(M, rs)/(2.-gamma)*((x**(2.-gamma)-1.)/x**1.5)*((2.+gamma)/3.)**0.5/h(x,gamma=gamma)
	def enth_analytic_nd(x, zeta, gamma, w):
		delta=1.+gamma

		return 1./x + w/x - ((3. - delta)*w*x**(2. - delta))/(2. - delta) - \
		(((2. - delta)*(3. - delta) - (3. - delta)**2)*w*(-1. + x**(5. - 2.*delta)))/\
		((5. - 2*delta)*(2. - delta)*(-1. + x**(3. - delta))) + \
		((1. - 2*delta)*(3. - delta)*(-1. + x**(2. - delta)))/\
		(2.*(2. - delta)*(1 + delta)*(-1. + x**(3. - delta))) + 1.5*w**(1./(3. - delta))*zeta**2

def enth_analytic(phi_rs, x, zeta, gamma,  w):
	'''Analytic expression for v^2/2+(1/(gamma-1))*(kb T/(mu mp)). Derived from Bernoulli conservation.'''
	return phi_rs*en_analytic_nd(x, zeta, gamma, w)

def be_analytic_nd(x, zeta, gamma, w, rb_rinf):
	delta=1.+gamma

	return -(((3. - delta)*rb_rinf**(2. - delta)*w**(1 + (-2 + delta)/(3 - delta)))/(2. - delta)) - \
	(((2. - delta)*(3. - delta) - (3. - delta)**2)*w*(-1. + x**(5. - 2.*delta)))/\
	((5. - 2*delta)*(2. - delta)*(-1. + x**(3. - delta))) + \
	((1. - 2*delta)*(3. - delta)*(-1. + x**(2. - delta)))/\
	(2.*(2. - delta)*(1 + delta)*(-1. + x**(3. - delta))) + 1.5*w**(1./(3. - delta))*zeta**2

def be_analytic(phi_rs, x, zeta, gamma, w, rb_rinf):
	'''Analytic expression for the Bernulli parameter

	:param phi_rs: potential at stagnation radius rs
	:param x: r/rs
	:param zeta: sqrt(vw^2+sigma_0^2)/sigma_0--where sigma_0 is the constant velocity dispersion
	:param gamma: nuker gamma
	:param w: (rs/rsoi)^(3-delta)
	:param rb_rinf: rb/rsoi
	'''
	return phi_rs*be_analytic_nd(x, zeta, gamma, w, rb_rinf)

def w_solve(rb_rinf, zeta, gamma):
	'''minimum rs/rinf, for a given rb/rinf, zeta, and gamma'''
	warnings.filterwarnings('error')

	delta=gamma+1.
	try:
		return fsolve(lambda w: be_analytic_nd(rb_rinf/w**(1./(2.-gamma)), zeta, gamma, w, rb_rinf),100.)[0]
	except Warning:
		return np.nan

def zeta_c(rb_rinf,  gamma):
	'''minimum zeta for an outflow given rb/rinf and gamma'''
	warnings.filterwarnings('error')
	delta=gamma+1.
	rs_rinf=0.001
	w=rs_rinf**(2.-gamma)

	try:
		return fsolve(lambda zeta: be_analytic_nd(rb_rinf/w**(1./(2.-gamma)), zeta, gamma, w, rb_rinf),100.)[0]
	except Warning:
		return np.nan
