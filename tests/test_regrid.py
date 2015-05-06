#!/usr/bin/env python

from .. import galaxy
import sys
import numpy as np

from math import e

import pytest

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

first_deriv_weights=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.
second_deriv_weights=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.)

def test_regrid_rmin():
	gal=galaxy.NukerGalaxy('NGC4551')

	rmin=0.5*gal.radii[0]
	rmax=gal.radii[-1]
	gal.re_grid(rmin, rmax, gal.length)
	assert np.allclose(gal.radii,np.logspace(np.log(rmin), np.log(rmax), gal.length, base=e))

def test_regrid_rmax():
	gal=galaxy.NukerGalaxy('NGC4551')

	rmin=gal.radii[0]
	rmax=2.*gal.radii[-1]
	gal.re_grid(rmin, rmax, gal.length)
	assert np.allclose(gal.radii,np.logspace(np.log(rmin), np.log(rmax), gal.length, base=e))

def test_regrid_length():
	gal=galaxy.NukerGalaxy('NGC4551')
	assert gal.length==70

	rmin=gal.radii[0]
	rmax=gal.radii[-1]
	gal.re_grid(rmin, rmax, 50)
	assert np.allclose(gal.radii,np.logspace(np.log(rmin), np.log(rmax), 50, base=e))
	assert gal.end==46
	assert gal.start==3
	assert gal.tcross==(gal.radii[-1]-gal.radii[0])/(galaxy.kb*1.E7/galaxy.mp)**0.5
	assert np.all(gal.first_deriv_coeffs==np.array([first_deriv_weights/(r*gal.delta_log[0]) for r in gal.radii]))
	assert np.all(gal.second_deriv_coeffs==np.array([(1./r**2)*(second_deriv_weights/(gal.delta_log[0]**2)-(first_deriv_weights)/(gal.delta_log[0]))\
		for r in gal.radii]))


		







	



	




