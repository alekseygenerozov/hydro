#!/usr/bin/env python

from .. import galaxy
import sys
import numpy as np

from math import e

import pytest

# # -----------------------------------------------------------------
# # This block of code tells Python to drop into the debugger
# # if there is an uncaught exception when run from the command line.
# def info(type, value, tb):
# 	if hasattr(sys, 'ps1') or not sys.stderr.isatty():
# 	# we are in interactive mode or we don't have a tty-like
# 	# device, so we call the default hook
# 		sys.__excepthook__(type, value, tb)
# 	else:
# 		import traceback, pdb
# 		# we are NOT in interactive mode, print the exception...
# 		traceback.print_exception(type, value, tb)
# 		print
# 		# ...then start the debugger in post-mortem mode.
# 		pdb.pm()
# sys.excepthook = info
# # -----------------------------------------------------------------



def test_init_array_lin():
	gal=galaxy.Galaxy.from_dir(loc='data/lin_grid')

	assert gal._logr==False
	assert np.allclose(gal.radii,np.linspace(1.,100.,gal.length))
	assert gal.length==100
	assert gal.end==96
	assert gal.start==3

	assert np.allclose(gal.rho,np.ones(gal.length))
	assert np.allclose(gal.log_rho,np.zeros(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))

def test_init_array_length():
	gal=galaxy.Galaxy.from_dir(loc='data/lin_grid', length=50)
	print gal.length

	assert gal._logr==False
	assert np.allclose(gal.radii,np.linspace(1.,100.,gal.length))

	assert np.allclose(gal.rho,np.ones(gal.length))
	assert np.allclose(gal.log_rho,np.zeros(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))

def test_init_array_rescale():
	gal=galaxy.Galaxy.from_dir(loc='data/lin_grid', rescale=2.)
	print gal.length

	assert gal._logr==False
	assert np.allclose(gal.radii,np.linspace(2.,200.,gal.length))

	assert np.allclose(gal.rho,np.ones(gal.length))
	assert np.allclose(gal.log_rho,np.zeros(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))

def test_init_array_log():
	gal=galaxy.Galaxy.from_dir(loc='data/log_grid')

	assert gal._logr==True
	assert np.allclose(gal.radii,np.logspace(1.,100.,gal.length, base=e))
	assert gal.length==100

	assert np.allclose(gal.rho,np.ones(gal.length))
	assert np.allclose(gal.log_rho,np.zeros(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))

def test_init_array_log2():
	gal=galaxy.NukerGalaxy.from_dir(args=['NGC4551'],loc='data/log_grid')
	gal_dict=galaxy.nuker_params()

	assert gal.params==gal_dict['NGC4551']

	assert gal._logr==True
	assert np.allclose(gal.radii,np.logspace(1.,100.,gal.length, base=e))
	assert gal.length==100

	assert np.allclose(gal.rho,np.ones(gal.length))
	assert np.allclose(gal.log_rho,np.zeros(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))

def test_init_array_rescale_auto():
	gal=galaxy.NukerGalaxy.from_dir(args=['NGC4551'],loc='data/log_grid', rescale='auto')
	gal_dict=galaxy.nuker_params()

	assert gal.params==gal_dict['NGC4551']

	assert gal._logr==True
	assert np.allclose(abs(gal.rinf/gal.radii[0]-96.5)/96.5, 0.)
	#assert np.allclose(gal.radii,np.logspace(1.,100.,gal.length, base=e))
	assert gal.length==100

	assert np.allclose(gal.rho,np.ones(gal.length))
	assert np.allclose(gal.log_rho,np.zeros(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))





	



	




