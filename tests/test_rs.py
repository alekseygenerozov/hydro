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

def test_all_pos():
	gal=galaxy.Galaxy.from_dir(loc='data/all_pos')
	assert gal.rs==[]

def test_all_neg():
	gal=galaxy.Galaxy.from_dir(loc='data/all_neg')
	assert gal.rs==[]

def test_one_zero():
	gal=galaxy.Galaxy.from_dir(loc='data/one_zero')
	assert np.allclose(gal.rs, [49.5])

def test_two_zeros():
	gal=galaxy.Galaxy.from_dir(loc='data/two_zeros')
	assert np.allclose(gal.rs, [49.5, 99.5])

def test_bdry():
	gal=galaxy.Galaxy.from_dir(loc='data/bdry')
	assert np.allclose(gal.rs, [0.5])






	



	




