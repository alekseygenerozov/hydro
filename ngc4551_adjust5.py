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


saved=np.load('/home/aleksey/Second_Year_Project/hydro/ngc4551_2/vw_crit/smooth_bdry2/save.npz')['a']
# grid2=pickle.load( open( "grid_backup.p", "rb" ) )
start=hydro.prepare_start(saved[-1])

vw=np.array([  1.15125402e+08,   1.10303121e+08,   1.05706427e+08,
         1.01325837e+08,   9.71522974e+07,   9.31771672e+07,
         8.93921952e+07,   8.57895024e+07,   8.23615640e+07,
         7.91011914e+07,   7.60015151e+07,   7.30559680e+07,
         7.02582690e+07,   6.76024074e+07,   6.50826270e+07,
         6.26934119e+07,   6.04294709e+07,   5.82857241e+07,
         5.62572889e+07,   5.43394670e+07,   5.25277313e+07,
         5.08177144e+07,   4.92051972e+07,   4.76860978e+07,
         4.62564622e+07,   4.49124552e+07,   4.36503524e+07,
         4.24665332e+07,   4.13574749e+07,   4.03197480e+07,
         3.93500119e+07,   3.84450126e+07,   3.76015806e+07,
         3.68166299e+07,   3.60871585e+07,   3.54102483e+07,
         3.47830665e+07,   3.42028675e+07,   3.36669944e+07,
         3.31728813e+07,   3.27180549e+07,   3.23001368e+07,
         3.19168449e+07,   3.15659946e+07,   3.12454998e+07,
         3.09533731e+07,   3.06877261e+07,   3.04467682e+07,
         3.02288061e+07,   3.00322420e+07,   2.98555712e+07,
         2.96973805e+07,   2.95563447e+07,   2.94312241e+07,
         2.93208610e+07,   2.92241762e+07,   2.91401654e+07,
         2.90678958e+07,   2.90065021e+07,   2.89551829e+07,
         2.89131971e+07,   2.88798602e+07,   2.88545406e+07,
         2.88366566e+07,   2.88256729e+07,   2.88210974e+07,
         2.88224780e+07,   2.88294004e+07,   2.88414847e+07,
         2.88583831e+07])


outdir='/home/aleksey/Second_Year_Project/hydro/ngc4551_2/vw_crit/smooth_bdry3'
params=dict(n=70, safety=0.6, Re=180.,  params=d,  floor=0.,
    logr=True, symbol='r', isot=False,  movies=False, qpc=True, params_delta=(), Re_s=500., vw=vw,
    veff=False, scale_heating=1.,outdir=outdir, eps=1., visc2=False, tinterval=0.05*tcross)

grid2=hydro.Grid(g1_2.params['M'], M_enc_simp, g1_2.q, init_array=start, **params)
grid2.solve_adjust(2.*tcross, 'Re', 360.)
grid2.backup()

ipyani.movie_save(outdir)

# grid2=pickle.load( open( "grid_backup.p", "rb" ) )
# start=hydro.prepare_start(grid2.saved[-1])
# grid2=hydro.Grid(g1_2.params['M'], M_enc_simp, g1_2.q, init_array=start, **params)

# grid2.solve(0.14*tcross)

# vw=2.5E7
# grid2.set_param('Re_s', 500.)
# grid2.solve_adjust(6.*tcross, 'vw', c*((grid2.rg/grid2.radii)*(grid2.M_tot/grid2.M_bh)+(vw/c)**2)**0.5)
# grid2.backup()
# grid2.solve_adjust(6.*tcross, 'eps', 1.)


