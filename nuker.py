#!/usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.optimize import fsolve
from astropy.io import ascii
import astropy.constants as const
from astropy.table import Table

import warnings
from scipy.special import gamma
import scipy.optimize as opt
from scipy.interpolate import interp1d

import tde_jet

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

##Run a command from the bash shell
def bash_command(cmd):
    process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process.communicate()[0]

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

##Reads Wang & Merritt parameter table into python dictionary; skip skips over galaxies discarded by WM
def nuker_params(skip=False):
    table=ascii.read('wm')
    galaxies=dict()
    for i in range(len(table)):
        d=dict()
        if table['$\\log_{10}\\dot{N}$\\tablenotemark{f}'][i]=='-' and skip:
            continue
        d['Uv']=table[i]['$\\Upsilon_V $']
        d['alpha']=table[i]['$\\alpha$']
        d['beta']=table[i]['$\\beta$']
        d['gamma']=table[i]['$\\Gamma$']
        d['rb']=10.**table[i]['$\log_{10}(r_{b})$']
        d['Ib']=mub_to_Ib(table[i]['$\mu_b$'])
        d['mub']=table[i]['$\mu_b$']
        d['d']=table[i]['Distance']
        d['M']=M_sun*10.**table[i]['$\\log_{10}(M_{\\bullet}/M_{\\odot})$\\tablenotemark{e}']
        d['type']=table[i][r'Profile\tablenotemark{b}']
        if d['type']=='$\\cap$':
            d['type']='core'
        else:
            d['type']='cusp'
        galaxies[table[i]['Name']]=d

    return galaxies

class Galaxy:
    """Class to store info about Nuker galaxies
    """
    def __init__(self, gname, gdata):
        try:
            self.params=gdata[gname]
        except KeyError:
            print 'Error! '+gname+' is not in catalog!'
            raise
        self.name=gname
        print self.name
        self.eta=1.
        self.rmin_star=1.E-3
        self.rmax_star=1.E5
        self.menc=menc


    def rho_stars(self,r):
        return M_sun*self.params['Uv']*inverse_abel(nuker_prime, r, **self.params)

    def M_enc(self,r):
        if r<self.rmin:
            return 0.
        # elif self.menc=='circ':
        #     return integrate.quad(lambda r1:2.*np.pi*r1*M_sun*self.params['Uv']*nuker(r1, **self.params),self.rmin)
        else:
            return integrate.quad(lambda r1:4.*np.pi*r1**2*self.rho_stars(r1), self.rmin, r)[0]

    def sigma(self, r):
        '''Velocity dispersion of galaxy

        Parameters
        ===========
        r : float
            radius 
        '''
        rg=G*(self.params['M'])/c**2./pc
        return (c**2*self.rg/r*(self.M_enc(r)+self.params['M'])/self.params['M'])**0.5

    def phi_s(self,r):
        '''Potential from the stars

        Parameters
        ===========
        r : float
            radius 
        '''
        return (-G*self.M_enc(r)/r)-4.*np.pi*G*integrate.quad(lambda r1:self.rho_stars(r1)*r1, r, self.rmax)[0]

    def phi_bh(self,r):
        '''Potential from the black hole

        Parameters
        ===========
        r : float
            radius 
        '''
        return -G*self.params['M']/r

    def phi(self,r):
        '''Total potential

        Parameters
        ===========
        r : float
            radius 
        '''
        return self.phi_bh(r)+self.phi_s(r)

    def q(self, r):
        '''Source term representing mass loss from stellar winds'''
        return self.eta*self.rho(r)/th


    ##Getting the radius of influence: where the enclosed mass begins to equal the mass of the central BH. 
    def rinf(self):
        '''Get radius of influence for galaxy'''
        def mdiff(r):
            return self.params['M']-self.M_enc(r)

        return fsolve(mdiff, 1)[0]

    #Get accretion rate  for galaxy by integrating source from stagnation radius
    def get_mdot(self, vw=1.E8):
        bf=0.01
        #Get the stagnation radius in parsecs
        rs=G*self.params['M']/((vw**2/2.))/pc
        #Get mdot by integrating the source function (assuming that the break radius is well outside of our region of interest)
        if rs<bf*self.params['rb']:
            mdot=4.*np.pi*(self.q(rs)*rs**3)/(2.-self.params['gamma'])
        else:
            with warnings.catch_warnings(record=True) as w:
                mdot=4.*np.pi*integrate.quad(lambda r: r**2*self.q(r), bf*self.params['rb'], rs)[0]
                mdot=mdot+4.*np.pi*(self.q(bf*self.params['rb'])*(bf*self.params['rb'])**3)/(2.-self.params['gamma'])

        return mdot

    def rs_gen(self, eta=3.):
        f=lambda x: (1+x**(2.-self.params['gamma']))/x-eta
        fprime=lambda x: -(1. + x**(2. - self.params['gamma']))/x**2. + x**(-self.params['gamma'])*(2.-self.params['gamma'])
        if f(opt.fmin(f, 1.)[0])>0:
            print 'root not found!'
            return
        return opt.fsolve(f, 0.5, fprime=fprime)

    def get_eddr(self, vw=1.E8):
        l_edd=4.*np.pi*G*self.params['M']*c/(0.4)
        mdot_edd=l_edd/(0.1*c**2)

        mdot=self.get_mdot()
        return mdot/mdot_edd

    def set_gas_dens(self, r, rho_g):
        self.rho_g=interp1d(r, rho_g)

    def vj(r,jet):
        f=jet.rho(r*pc)/self.rho_g(r)
        beta_sh=(1.-(1./jet.gamma_j)**2.-(2./gamma_j)*(f**-0.5))**0.5
        return jet.beta_j/beta_sh

    def get_rc(self, ms=1., eta=0.1, theta=0.1, gamma_j=10.):
        m6=self.params['M']/(1.E6*M_sun)
        jet=tde_jet.Jet(ms=ms, m6=m6, eta=eta, theta=theta, gamma_j=gamma_j)

        r=integrate.ode(vj, )
        r.set_integrator('vode')
        r.set_initial_value(0., t=jet.delta/pc)
        try:
            while r.y<r.t:
                r.integrate(r.t+0.01)
            rc=r.y[0]
        except Exception as inst:
            print inst
            rc=np.nan

        f=jet.rho(rc*pc)/self.rho_g(rc)
        gamma=gamma_j*(1.+2.*gamma_j*f**(-0.5))**(-0.5)   
        # return np.array([self.name, rc, self.rho_g(rc)/mp, gamma],dtype=[('Name',str), ('b',float), ('c', float),('d',float)])
        return [rc, self.rho_g(rc)/mp, gamma]

    def summary(self, vw=1.E8):
        return Table(data=[self.params])




def main():    
    galaxies=nuker_params()

    rad=np.logspace(0,3,100)   
    g1=Galaxy('NGC4551', galaxies, cgs=False)
    g2=Galaxy('NGC4168', galaxies, cgs=False)
    rho=np.array(map(g1.rho,rad))/M_sun
    rho2=np.array(map(g2.rho, rad))/M_sun

    rho_nick=np.genfromtxt('NGC4551_nick')
    rho2_nick=np.genfromtxt('NGC4168_nick')

    plt.loglog()
    plt.plot(rad,rho)
    plt.plot(rho_nick[:,0], rho_nick[:,1])

    plt.plot(rad, rho2)
    plt.plot(rho2_nick[:,0], rho2_nick[:,1])
    plt.savefig('inv_abel_test.png')

if __name__ == '__main__':
    main()