#!/usr/bin/env python

from .. import galaxy
import matplotlib.pyplot as plt
import numpy as np

import astropy.constants as const

from scipy.misc import derivative
from scipy.interpolate import interp1d

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value
th=4.35*10**17
year=3.15569E7



def test_dens_extend():
	gal_params={'NGC3165':{'beta':0.86, 'gamma':1.78, 'Uv':4.1, 'M':3.6E7*galaxy.M_sun, 'alpha':4.29, 'Ib':2093., 'rb':89.2}}
	gal=galaxy.NukerGalaxyExtend('NGC3165',gal_params,init={'rmin':5.0E16, 'rmax':1.0E19})
	gal2=galaxy.NukerGalaxy('NGC3165',gal_params,init={'rmin':5.0E16, 'rmax':1.0E19})
	
	gal.rmin_star=(1.55E17/(galaxy.pc))
	rho_cut=gal.rho_stars_grid[gal.radii/galaxy.pc<gal.rmin_star]
	r_cut=gal.radii[gal.radii/galaxy.pc<gal.rmin_star]

	assert np.allclose(rho_cut, rho_cut[0]*(r_cut/r_cut[0])**0.5)



	

