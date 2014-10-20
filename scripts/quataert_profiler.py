#!/usr/bin/env python

import cProfile
import pstats

# import pyximport
import numpy as np
# pyximport.install(setup_args={'include_dirs': np.get_include()})

from nuker_catalog import galaxy,gal_properties

import numpy as np
import dill

import astropy.constants as const
#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value


def main():
	rmin=3.8E16
	rmax=3.E18

	init={'rmin':rmin, 'rmax':rmax, 'func_params':dict(rho_0=1.E-23, temp=1.E7, n=0.)}

	gal=galaxy.Galaxy(init=init)
	gal.set_param('bdry', 'bp')
	gal.set_param('mu', 0.5)
	# gal.set_param('tinterval', None)
	gal.set_param('outdir', 'quataert')

	gal.set_param('isot', True)
	gal.solve(gal.tcross)
	gal.set_param('isot', False)
	gal.solve(gal.tcross)

main()

# cProfile.run('main()', 'restats')
# p = pstats.Stats('restats')
# p.strip_dirs().sort_stats(1).print_stats()

