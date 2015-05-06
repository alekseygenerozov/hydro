#!/usr/bin/env python

from .. import galaxy
from .. import gal_properties as gp
import sys
import numpy as np

import astropy.constants as const


G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value
# -----------------------------------------------------------------
# This block of code tells Python to drop into the debugger
# if there is an uncaught exception when run from the command line.
def info(type, value, tb):
	if hasattr(sys, 'ps1') or not sys.stderr.isatty():
	# we are in interactive mode or we don't have a tty-like
	# device, so we call the default hook
		sys.__excepthook__(type, value, tb)
	else:
		import traceback, pdb
		# we are NOT in interactive mode, print the exception...
		traceback.print_exception(type, value, tb)
		print
		# ...then start the debugger in post-mortem mode.
		pdb.pm()
sys.excepthook = info
# -----------------------------------------------------------------


gal=galaxy.Galaxy()

def test_quataert_vw():
	gal.sigma_grid
	assert np.allclose(gal.sigma_0/(2.0E7),(3.0)**0.5*gp.sigma_200(gal.params['M']))
	gal.set_param('sigma_0', 1.E7)
	assert np.allclose(gal.sigma_0, 1.E7)
	assert np.allclose(gal.vw**2.-gal.sigma_grid**2., gal.vw_extra**2.)
	k=3./(2.+gal.params['gamma'])
	assert np.allclose(gal.sigma_grid**2., k*G*gal.params['M']/gal.radii+gal.sigma_0**2.)
	gal.set_param('eps_stellar_heating', 0.)
	assert np.allclose(gal.sigma_grid**2., k*G*gal.params['M']/gal.radii)








