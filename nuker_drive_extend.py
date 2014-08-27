#!/usr/bin/env python

import galaxy

import astropy.constants as const
import numpy as np

import argparse

from astropy.io import ascii
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
	parser.add_argument('init', nargs=1,
		help='File containing list of galaxies and config files')

	args=parser.parse_args()
	init=ascii.read(args.init[0])

	for i in range(len(init)):
		try:
			print init[i]['pickle']
			gal=dill.load(open(init[i]['pickle'], 'rb'))
		except:
			print 'Could not open pickle!'
			continue

		gal.set_param('outdir', gal.name+'/vw_{0}'.format(gal.vw_extra/1.E5)+'_extended/')
		gal.solve(100.*gal.tcross)
		galaxy.bash_command('cp '+init[i]['pickle']+' '+gal.outdir+'/init.p')


if __name__ == '__main__':
	main()
