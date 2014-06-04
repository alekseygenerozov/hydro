import sys
import hydro_experiment as hydro
import nuker
import parker
import astropy.constants as const
import numpy as np
import ipyani
import pickle
import matplotlib.pylab as plt

from scipy.interpolate import interp1d
from scipy.optimize import fsolve



saved=np.load('/home/aleksey/Second_Year_Project/hydro/ngc4551_2/vw_crit/smooth_bdry5/save.npz')['a']
# grid2=pickle.load( open( "grid_backup.p", "rb" ) )
start=hydro.prepare_start(saved[-1])

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value

x=10.**7.11/(3.6*10.**6)

galaxies=nuker.nuker_params()

rmin=3.8E16*x/pc
rmax=7.E19/pc
temp=1.e7
rho_0=1.E-23
d=dict(rho_0=rho_0, temp=temp, log=True)
tcross=parker.tcross(rmin*pc, rmax*pc, temp)

g1_2=nuker.Galaxy('NGC4551', galaxies, cgs=False, eta=1.) 
def M_enc_simp(r):
    return 4.76E39*(r/1.04)**(2-g1_2.params['gamma'])

#Find the stagnation radius for a particular solution. Take array of radii and velocities
def find_stag(r, v, guess=1.E18):
    interp=interp1d(r, v)
    return fsolve(interp, guess)

#Check solution as code is running
def sol_check(loc, index=2, size=70):
    saved=np.genfromtxt(loc+'/save')
    fig,ax=plt.subplots()
    ax.set_xscale('log')
    if index!=2:
        ax.set_yscale('log')
    plt.plot(saved[0:size,0], saved[0:size,index])
    plt.plot(saved[-size:,0], saved[-size:,index])
    plt.show()
    
#Check solution as code is running
def cons_check(loc, index=2, logy=False):
    saved=np.genfromtxt(loc+'/cons')
    fig,ax=plt.subplots()
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    plt.plot(saved[-69:,0], saved[-69:,index])
    plt.plot(saved[-69:,0], saved[-69:,index+3])
    plt.show()    

#Analytic expression for Bernoulli parameter.
def be(r, M_bh=10.**7.11*M_sun, M_enc0=7.95E6*M_sun, r_s=5.E17, vw=1.E8, beta=1.8, sigma=False):
    x=r/r_s
    
    if not sigma:
        return (vw**2/2.)-(G*M_bh/r_s)*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/r_s)*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)
    else:
        return (vw**2/2.)-(G*M_bh/(2.*r_s))*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/(2.*r_s))*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)

def be_check(loc, menc=True, index=-1, vw=1.E8, beta=1.8, sigma=False):
    #Load saved data
    saved2=np.load(loc+'/save.npz')['a']
    #Calculate the stagnation radius for the numerical solution.
    r_s=find_stag(saved2[index,:,0], saved2[index,:,2], guess=5.E17)

    #If flag to include the enclosed stellar mass is set then calculate Menc0 (the mass enclosed at the stagnation radius)
    if menc:
        M_enc0=M_enc_simp(r_s/pc)
    else:
        M_enc0=0.

    #Calculate the analytic Bernoulli parameter.
    bes=np.empty_like(saved2[index,:,0])
    for i in range(len(saved2[index,:,0])):
        bes[i]=be(saved2[index,i,0], vw=vw, r_s=r_s, M_enc0=M_enc0, beta=beta, sigma=sigma)

    fig,ax=plt.subplots()
    plt.loglog()
    ax.plot(saved2[index,:,0], saved2[index,:,-3])
    ax.plot(saved2[index,:,0], bes)
    fig.savefig(loc+'/be_cons_'+loc+'.png')
    fig.clf()

    fig,ax=plt.subplots()
    plt.loglog()
    ax.plot(saved2[index,:,0], abs((saved2[index,:,-3]-bes)/bes))
    fig.savefig(loc+'/be_cons2_'+loc+'.png')



