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


def test_phi():
	gal_dict=galaxy.nuker_params(skip=True)
	max_diff=0.

	for name in ['NGC4551', 'NGC4168']:
		gal=galaxy.NukerGalaxy(name,init={'r1':0.01*pc, 'r2':1.E4*pc, 'length':100})

		explicit_diff=gal.get_spatial_deriv('phi_s_grid')[gal.start:gal.end+1]
		grad_phi=(G*gal.M_enc_grid/gal.radii**2 )[gal.start:gal.end+1]

		max_diff_gal=max(abs((grad_phi-explicit_diff)/grad_phi)*100.)
		max_diff=max(max_diff, max_diff_gal)

	assert max_diff<1.

	

