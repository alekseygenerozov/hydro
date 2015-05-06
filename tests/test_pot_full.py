#!/usr/bin/env python

from nuker_catalog import galaxy
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
	f=open('gals_compare','w')

	for name in gal_dict:
		max_diff=0.
		gal=galaxy.NukerGalaxy(name,init={'rmin':0.01*pc, 'rmax':1.E4*pc, 'length':100})
		explicit_diff=gal.get_spatial_deriv('phi_grid')[gal.start:gal.end+1]
		grad_phi=(G*(gal.M_enc_grid+gal.params['M'])/gal.radii**2 )[gal.start:gal.end+1]

		max_diff_gal=max(abs((grad_phi-explicit_diff)/grad_phi)*100.)
		max_diff=max(max_diff, max_diff_gal)

		f.write(name+' '+str(max_diff)+'\n')
		print name,max_diff

	f.close()
if __name__ == '__main__':
	test_phi()

