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
    def __init__(self, gname, gdata, eta=0.1, cgs=True):
        self.params=gdata[gname]
        self.eta=eta
        self.cgs=cgs

        self.rho=self.get_rho()
        self.M_enc_circ=self.get_M_enc_circ()
        self.M_enc=self.get_M_enc()
        self.q=self.get_q()
        self.rinf=self.get_rinf()


    ##Construct stellar density profile based on surface brightness profile
    def get_rho(self):
        def rho(r):
            rho1=M_sun*self.params['Uv']*inverse_abel(nuker_prime, r, **self.params)
            # if self.cgs:
            #     rho1=M_sun*rho1/pc**3
            return rho1

        return rho

    ##Construct the mass enclosed profile based on 
    def get_M_enc(self):
        def M_enc(r):
            f=lambda r1: 4.*np.pi*r1**2.*self.rho(r1)
            return integrate.quad(f, 0, r)[0]
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
            return self.params['M']-self.M_enc_circ(r)

        return fsolve(mdiff, 1)[0]

# ##Getting the potential from the Nuker params
# def get_M_enc(params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., M=1.e6)):
#     rho=get_rho(params)
#     def M_enc(r):
#         f=lambda r1: 4.*np.pi*r1**2.*rho(r1)
#         return integrate.quad(f, 0, r)[0]

#     return M_enc

# ##Getting the potential from the Nuker params
# def get_grad_phi(params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., M=1.e6)):
#     rho=get_rho(params)
#     def grad_phi(r):
#         grad_phi_bh=-G*params['M']/r**2
#         f=lambda r1: 4.*np.pi*r1**2.*rho(r1)
#         menc=integrate.quad(f, 0, r)[0]
#         grad_phi_s=-G*menc/r**2

#         return grad_phi_s+grad_phi_bh

#     return grad_phi

# ##Getting mass source term from the Nuker params
# def get_q(eta, params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., M=1.e6)):
#     def q(r):
#         rho=get_rho(params)
#         return eta*rho/th


def main():    
    galaxies=nuker_params()

    rad=np.logspace(0,3,100)   
    g1=Galaxy('NGC4551', galaxies, cgs=False)
    g2=Galaxy('NGC4168', galaxies, cgs=False)
    rho=map(g1.rho,rad)
    rho2=map(g2.rho, rad)
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