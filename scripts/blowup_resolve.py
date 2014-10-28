#!/usr/bin/env python

import galaxy
import numpy as np

import astropy.constants as const
from astropy.io import ascii

import argparse
import dill


#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B .cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value

##Should add flag to compute all of the nuker galaxies
def main():
		parser=argparse.ArgumentParser(
				description='Code for gradually adjusting parameter to a particular value')
		parser.add_argument('d', nargs=1,
				help='Directory containing the solution of interest.')
		

		args=parser.parse_args()
		d=args.d[0]

		gal=dill.load(open(d+'/grid.p'))
		gal.refine()

if __name__ == '__main__':
	main()