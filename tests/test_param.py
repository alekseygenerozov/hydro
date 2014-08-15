#!/usr/bin/env python

import galaxy
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

gal_dict=galaxy.nuker_params()
gal=galaxy.NukerGalaxy('NGC4168', gal_dict)

def test_isot():
	cs_old=np.copy(gal.cs)
	gal.set_param('isot', True)
	assert gal.fields==['log_rho', 'vel']
	assert np.allclose(cs_old/gal.cs,[np.sqrt(gal.gamma)]*gal.length)
	assert gal.isot==True

	gal.set_param('isot', False)
	assert gal.fields==['log_rho', 'vel', 's']
	assert np.allclose(cs_old, gal.cs)
	assert gal.isot==False

def test_vw():
	gal.set_param('sigma_heating', False)
	assert gal.vw.shape==(gal.length,)
	assert np.allclose(gal.vw, np.array([gal.vw_extra]*gal.length))

def test_eps():
	gal.set_param('eps', 0.)
	assert gal.eps==0.
	assert np.allclose(gal.phi_grid, -G*gal.params['M']/gal.radii)
	assert np.allclose(gal.grad_phi_grid, G*gal.params['M']/gal.radii**2)

	gal.set_param('eps', 1.)
	assert gal.eps==1.
	assert np.allclose(gal.grad_phi_grid, G*(gal.M_enc_grid+gal.params['M'])/gal.radii**2)








