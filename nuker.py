import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from astropy.io import ascii
import astropy.constants as const

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
#params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7.)

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
        d['M']=table[i]['$\\log_{10}(M_{\\bullet}/M_{\\odot})$\\tablenotemark{e}']
        galaxies[table[i]['Name']]=d

    return galaxies

##Construct number density profile based on surface brightness profile
def get_rho(params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7.)):
    def rho(r):
        return params['Uv']*inverse_abel(nuker_prime, r, **params)
    return rho

##Getting the potential from the Nuker params
def get_grad_phi(params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., M=1.e6)):
    rho=get_rho(params)
    def grad_phi(r):
        grad_phi_bh=-G*params['M']/r**2
        f=lambda r1: 4.*np.pi*r1**2.*rho(r1)
        menc=integrate.quad(f, 0, r)[0]
        grad_phi_s=-G*menc/r**2

        return grad_phi_s+grad_phi_bh

    return grad_phi

##Getting mass source term from the Nuker params
def get_q(eta, params=dict(Ib=17.16, alpha=1.26, beta=1.75, rb=343.3, gamma=0, Uv=7., M=1.e6)):
    def q(r):
        rho=get_rho(params)



def main():    
    galaxies=nuker_params()

    rad=np.logspace(0,3,100)   
    rho=map(get_rho(params=galaxies['NGC4551']),rad)
    rho2=map(get_rho(params=galaxies['NGC4168']), rad)
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