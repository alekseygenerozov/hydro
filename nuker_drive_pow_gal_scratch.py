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
	gal=galaxy.PowGalaxy.from_dir(loc='pow_gal_M1.0e+07_gamma8.0e-01/vw_600.0', init={'rmax':7.E19})
	gal.solve_adjust('params[gamma]',0.1)
	gal.solve()
	gal.re_grid(gal.radii[0], 100*rinf)
	gal.solve(5.*gal.tcross)
	gal.solve()

if __name__ == '__main__':
	main()
