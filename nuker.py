#!/usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.optimize import fsolve
from astropy.io import ascii
import astropy.constants as const

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

##Reads Wang & Merritt parameter table into python dictionary
def nuker_params():
    table=ascii.read('wm')
    galaxies=dict()
    for i in range(len(table)):
        d=dict()
        d['Uv']=table[i]['$\\Upsilon_V $']
        d['alpha']=table[i]['$\\alpha$']
        d['beta']=table[i]['$\\beta$']
        d['gamma']=table[i]['$\\Gamma$']
        d['rb']=10.**table[i]['$\log_{10}(r_{b})$']
        d['Ib']=mub_to_Ib(table[i]['$\mu_b$'])
        d['mub']=table[i]['$\mu_b$']
        d['d']=table[i]['Distance']
        d['M']=M_sun*10.**table[i]['$\\log_{10}(M_{\\bullet}/M_{\\odot})$\\tablenotemark{e}']
        galaxies[table[i]['Name']]=d

    return galaxies

class Galaxy:
    """Class to store info about Nuker galaxies"""
    def __init__(self, gname, gdata, eta=0.1, cgs=False, menc='default'):
        try:
            self.params=gdata[gname]
        except KeyError:
            print 'Error! '+gname+' is not in catalog!'
            raise
        self.name=gname
        self.eta=eta
        # self.grams=grams

        self.rho=self.get_rho()

        if menc=='circ':
            self.M_enc=self.get_M_enc_circ()
        elif menc=='full':
            self.M_enc=self.get_M_enc()
        else:
            self.M_enc=self.get_M_enc_simp()
        self.q=self.get_q()
        self.phi=self.get_phi()
        self.sigma=self.get_sigma()
        # self.rinf=self.get_rinf()


    ##Construct stellar density profile based on surface brightness profile
    def get_rho(self):
        def rho(r):
            rho1=M_sun*self.params['Uv']*inverse_abel(nuker_prime, r, **self.params)
            return rho1
        return rho

    ##Construct the mass enclosed profile based on 
    def get_M_enc(self):
        def M_enc(r):
            f=lambda r1: 4.*np.pi*r1**2.*self.rho(r1)
            return integrate.quad(f, 0, r)[0]
        return M_enc

    ##Get simplified expression for the mass encloed in a particular radius 
    def get_M_enc_simp(self):
        M_enc0=(self.get_M_enc())(1.)
        def M_enc(r):
            return M_enc0*r**(2-self.params['gamma'])
        return M_enc

    def get_M_enc_circ(self):
        def M_enc(r):
            f=lambda r1:2.*np.pi*r1*M_sun*self.params['Uv']*nuker(r1, **self.params)
            return integrate.quad(f, 0, r)[0]
        return M_enc

    ##Construct the mass source term assuming some efficiency
    def get_q(self):
        def q(r):
            return self.eta*self.rho(r)/th
        return q

    ##Getting the radius of influence: where the enclosed mass begins to equal the mass of the central BH. 
    def get_rinf(self):
        def mdiff(r):
            return self.params['M']-self.M_enc(r)

        return fsolve(mdiff, 1)[0]

    ##Get potential. Needs more thorough checking. Integral truncated at 10 times the break radius instead of at infinity.
    def get_phi(self):
        def phi(r):
            phi_bh=-G*self.params['M']/r
            phi_1=-G*self.M_enc(r)/r
            f=lambda r: self.rho(r)*r
            phi_2=-4.*np.pi*G*integrate.quad(f, r, 10.*self.params['rb'])[0]
            return phi_bh+phi_1+phi_2
        return phi

    ##Get the velocity dispersion as a function of r
    def get_sigma(self):
        def sigma(r):
            rg=G*self.params['M']/c**2/pc
            return (c**2*rg/r*(self.M_enc(r)+self.params['M'])/self.params['M'])**0.5
        return sigma


def main():    
    galaxies=nuker_params()

    rad=np.logspace(0,3,100)   
    g1=Galaxy('NGC4551', galaxies, grams=False)
    g2=Galaxy('NGC4168', galaxies, grams=False)
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