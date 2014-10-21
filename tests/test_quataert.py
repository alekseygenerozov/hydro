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

def quataert_test():
	rmin=3.8E16
	rmax=3.E18

	init={'rmin':rmin, 'rmax':rmax, 'func_params':dict(rho_0=1.E-23, temp=1.E7, n=0.)}

	gal=galaxy.Galaxy(init=init)
	gal.set_param('bdry', 'bp')
	gal.set_param('mu', 0.5)
	gal.set_param('tinterval', None)
	gal.set_param('outdir', 'quataert')
	gal.set_param('sigma_heating', False)

	gal.set_param('isot', True)
	gal.solve(gal.tcross)
	gal.set_param('isot', False)
	gal.solve(gal.tcross)

def test_quataert():
	quataert_test()
	saved1=np.load('quataert/save.npz')['a']
	saved2=np.load('quataert_reproduce/save.npz')['a']

	assert np.allclose(saved1[-1,:,:8], saved2[-1,:,:8])




