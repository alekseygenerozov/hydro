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
	gal=galaxy.PowGalaxy.from_dir(loc='batch_collected/NGC4551/vw_500.0/')
	gal.set_param('params[gamma]', 0.8)
	# gal.set_param('vw_extra', 6.E7)
	gal.solve()
	gal.set_param('outdir', gal.name+'/vw_'+str(gal.vw_extra/1.E5))
	# gal.set_param('mu',0.62)
	# gal.solve()
	gal.re_grid(0.005*rinf, 100*rinf)
	gal.solve(5.*gal.tcross)

if __name__ == '__main__':
	main()
