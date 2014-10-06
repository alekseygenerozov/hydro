#!/usr/bin/env python

from .. import galaxy
from .. import nuker_driver_gen as nd
import sys
import numpy as np
import dill

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



def test_driver_vanilla():
	init_file='test_driver_vanilla'
	gal=dill.load(open('NGC4551/vw_500.0/grid.p','rb'))
	driver=nd.Driver(init_file)
	
	assert driver.gal.name==gal.name
	assert np.all(driver.gal.radii==gal.radii)
	assert driver.gal.non_standard==gal.non_standard

def test_driver_param():
	init_file='test_driver_params'
	gal=dill.load(open('NGC4551/vw_500.0/grid.p','rb'))
	driver=nd.Driver(init_file)
	
	assert driver.gal.vw_extra==1.E8
	assert driver.gal.cond_scheme=='shcherba'

def test_driver_grid():
	init_file='test_driver_grid'
	gal=dill.load(open('NGC4551/vw_500.0/grid.p','rb'))
	driver=nd.Driver(init_file)

	assert np.allclose(driver.gal.radii, 2.*gal.radii)


def test_driver_adjust():
	init_file='test_driver_adjust'
	driver=nd.Driver(init_file)

	assert driver.adjust_params_dict=={'Re':180}




