import sys
import hydro
import nuker
import parker
import astropy.constants as const
import numpy as np
from ipyani import *

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

params=dict(n=70, safety=0.6, Re=90.,  params=d,  floor=0.,
    logr=True, symbol='r', isot=True,  movies=False, vw=10.**8, qpc=True, params_delta=(), Re_s=10.**20.,
    veff=False, scale_heating=1.,outdir='.', eps=0., visc2=False, tinterval=0.05*tcross)
grid2=hydro.Grid(g1_2.params['M'], M_enc_simp, g1_2.q, init=[rmin*pc, rmax*pc, parker.background], **params)

grid2.solve(2.*tcross)
grid2.isot_off()
grid2.solve(2.*tcross)
grid2.backup()


