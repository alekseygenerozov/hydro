#!/usr/bin/env python

import galaxy
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



def test_init_array_lin():
	start=np.load('data/lin_grid.npz')['arr_0']
	gal=galaxy.Galaxy(init_array=start)

	assert gal._logr==False
	assert np.allclose(gal.radii,np.linspace(1.,100.,gal.length))
	assert gal.length==100

	assert np.allclose(gal.log_rho,np.ones(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))

def test_init_array_log():
	start=np.load('data/log_grid.npz')['arr_0']
	gal=galaxy.Galaxy(init_array=start)

	assert gal._logr==True
	assert np.allclose(gal.radii,np.logspace(1.,100.,gal.length, base=e))
	assert gal.length==100

	assert np.allclose(gal.log_rho,np.ones(gal.length))
	assert np.allclose(gal.vel, np.ones(gal.length))
	assert np.allclose(gal.temp, np.ones(gal.length))

def test_init_array_bad1():
	start=np.load('data/bad_grid_1.npz')['arr_0']
	with pytest.raises(Exception):
		gal=galaxy.Galaxy(init_array=start)

def test_init_array_bad2():
	start=np.load('data/bad_grid_2.npz')['arr_0']
	with pytest.raises(Exception):
		gal=galaxy.Galaxy(init_array=start)
	# assert exc.args[0]=="Radii must be evenly spaced in linear or log space!"

def test_init_array_bad3():
	start=np.load('data/bad_grid_3.npz')['arr_0']
	with pytest.raises(Exception):
		gal=galaxy.Galaxy(init_array=start)



	



	




