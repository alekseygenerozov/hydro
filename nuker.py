#!/usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.optimize import fsolve
from astropy.io import ascii
import astropy.constants as const

import warnings
from scipy.special import gamma
import scipy.optimize as opt
from scipy.interpolate import interp1d

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
    def __init__(self, gname, gdata, eta=0.1, cgs=False, menc='default', rmin=1.E-3, rmax=1.E5, points=100):
        try:
            self.params=gdata[gname]
        except KeyError:
            print 'Error! '+gname+' is not in catalog!'
            raise
        self.name=gname
        print self.name
        self.eta=eta
        self.menc=menc

        self.rmin=rmin
        self.rmax=rmax

        self.points=points
        self.r=np.logspace(np.log10(rmin), np.log10(rmax), self.points)
        self.rho_grid=self.get_rho_grid()
        self.M_enc_grid=self.get_M_enc_grid()

        self.phi_grid,self.phi_s_grid=self.get_phi_grid()

        # # self.sigma=self.get_sigma()
        self.rinf=self.get_rinf()

    def rho(self,r):
        return M_sun*self.params['Uv']*inverse_abel(nuker_prime, r, **self.params)

    def M_enc(self,r):
        if r<self.rmin:
            return 0.
        # elif self.menc=='circ':
        #     return integrate.quad(lambda r1:2.*np.pi*r1*M_sun*self.params['Uv']*nuker(r1, **self.params),self.rmin)
        else:
            return integrate.quad(lambda r1:4.*np.pi*r1**2*self.rho(r1), self.rmin, r, epsabs=1.E-3, epsrel=1.E-3)[0]

    def M_enc2(self,r):
        if r<self.rmin:
            return 0.
        # elif self.menc=='circ':
        #     return integrate.quad(lambda r1:2.*np.pi*r1*M_sun*self.params['Uv']*nuker(r1, **self.params),self.rmin)
        else:
            return integrate.quad(lambda r1:4.*np.pi*r1**2*self.rho_grid(r1), self.rmin, r, epsabs=1.E-3, epsrel=1.E-3)[0]

    def sigma(self, r):
        return (c**2*rg/r*(self.M_enc(r)+self.params['M'])/self.params['M'])**0.5

    def phi_s(self,r):
        return (-G*self.M_enc(r)/r)-4.*np.pi*G*integrate.quad(lambda r1:self.rho(r1)*r1, r, self.rmax, epsabs=1.E-3, epsrel=1.E-3)[0]

    def phi_s2(self, r):
        return (-G*self.M_enc_grid(r)/r)-4.*np.pi*G*integrate.quad(lambda r1:self.rho_grid(r1)*r1, r, self.rmax, epsabs=1.E-3, epsrel=1.E-3)[0]

    def phi_bh(self,r):
        return -G*self.params['M']/r

    def phi(self,r):
        return self.phi_bh(r)+self.phi_s(r)

    def q(self, r):
        return self.eta*self.rho(r)/th

    def get_rho_grid(self):
        rho_dat=map(self.rho, self.r)
        interp=interp1d(np.log10(self.r), np.log10(rho_dat))
        def rho_grid(r):
            return 10.**interp(np.log10(r))
        return rho_grid

    def get_M_enc_grid(self):
        m_dat=map(self.M_enc2, self.r)
        interp=interp1d(np.log10(self.r), np.log10(m_dat))
        def M_enc_grid(r):
            return 10.**interp(np.log10(r))
        return M_enc_grid

    def get_phi_grid(self):
        phi_s_dat=np.array(map(self.phi_s2, self.r))
        interp=interp1d(np.log10(self.r), np.log10(-phi_s_dat))
        def phi_s_grid(r):
            return -10.**interp(np.log10(r))
        def phi_grid(r):
            return phi_s_grid(r)+self.phi_bh(r)
        return [phi_grid, phi_s_grid]


    ##Getting the radius of influence: where the enclosed mass begins to equal the mass of the central BH. 
    def get_rinf(self):
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