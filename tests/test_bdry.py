#!/usr/bin/env python

from .. import galaxy
import parker 
import sys
import numpy as np

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



def test_bdry_def():
	start=np.load('data/power_laws/save.npz')['a'][-1]
	start_copy=np.copy(start)
	gal=galaxy.Galaxy.from_dir('data/power_laws')


	gal._update_ghosts()

	assert np.allclose(gal.rho, start_copy[:,1])
	assert np.allclose(gal.vel, start_copy[:,2])

def test_bdry_bp():
	start=np.load('data/bp/save.npz')['a'][-1]
	start_copy=np.copy(start)

	gal=galaxy.Galaxy.from_dir('data/bp/')
	gal.set_param('bdry', 'bp')
	assert gal.bdry=='bp'

	gal._update_ghosts()
	assert np.allclose(gal.rho, start_copy[:,1])
	assert np.allclose(gal.vel, start_copy[:,2])
	




