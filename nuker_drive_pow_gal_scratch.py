#!/usr/bin/env python

import galaxy
import gal_properties

import astropy.constants as const
import numpy as np

import argparse
from math import e

from astropy.io import ascii
from astropy.table import Column



#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B .cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value


def main():
	rinf=gal_properties.rinf(1.E7*M_sun)
	gal=galaxy.PowGalaxy(init={'rmin':0.01*rinf,'rmax':100.*rinf,'f_initial':galaxy.background,'func_params':{'temp': 1.E7, 'rho_0': 1e-23}, 'length':70})

	gal.isot_on()
	gal.set_param('bdry', 'bp')
	gal.set_param('eps', 0.)
	gal.set_param('mu', 0.615)
	gal.set_param('params[gamma]', 0.8)
	gal.set_param('vw_extra', 6.E7)
	gal.set_param('sigma_heating', False)
	gal.set_param('outdir', gal.name+'/vw_'+str(gal.vw_extra/1.E5))

	gal.solve(2.*gal.tcross)
	gal.isot_off()
	gal.solve(2.*gal.tcross)
	#Resetting boundary conditions
	gal.set_param('Re_s', 500.)
	gal.set_param('bdry', 'default')
	gal.solve(gal.tcross)
	#Turning on the stellar potential
	gal.set_param('eps',1.)
	gal.solve(3.*gal.tcross)
	gal.set_param('sigma_heating', True)	
	gal.solve(8.*gal.tcross)

	gal.backup()







if __name__ == '__main__':
	main()
