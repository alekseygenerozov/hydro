#!/usr/bin/env python

import sys
import hydro
import nuker
import parker
import astropy.constants as const
import numpy as np
import ipyani
import pickle

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


saved=np.load('/home/aleksey/Second_Year_Project/hydro/ngc4551_2/vw_crit/smooth_bdry5/save.npz')['a']
# grid2=pickle.load( open( "grid_backup.p", "rb" ) )
start=hydro.prepare_start(saved[-1])


outdir='.'
params=dict(n=70, safety=0.6, Re=180.,  params=d,  floor=0.,
    logr=True, symbol='r', isot=False,  movies=False, qpc=True, params_delta=(), Re_s=500., vw=0.,
    veff=False, scale_heating=1.,outdir=outdir, eps=1., visc_scheme='cap_visc', tinterval=0.05*tcross)
grid2=hydro.Grid(g1_2.params['M'], M_enc_simp, g1_2.q, init_array=start, **params)
vw_base=c*((grid2.rg/grid2.radii)*(grid2.M_tot/grid2.M_bh))**0.5

for x in range(5,0,-1):
    outdir='vw_'+str(x*100)
    params['outdir']=outdir
    params['vw']=(vw_base**2+(x*1.E7)**2)**0.5

    grid2=hydro.Grid(g1_2.params['M'], M_enc_simp, g1_2.q, init_array=start, **params)
    grid2.solve(8.*tcross)

    ipyani.movie_save(outdir, interval=1, 
        ymin=[10.**18,-10.**13,10.**4, 10.**-24, -1., 10.**6], ymax=[10**22,10.**13,2*10.**7, 10.**-20, 2., 10.**8], logy=[True, False, True, True, False, True])




# grid2.solve(6.*tcross)
# grid2.backup()

# ipyani.movie_save(outdir, interval=1, ymin=[10.**18,-10.**13,10.**4, 10.**-24, -1., 10.**6], ymax=[10**22,10.**13,2*10.**7, 10.**-20, 2., 10.**8], logy=[True, False, True, True, False, True])




