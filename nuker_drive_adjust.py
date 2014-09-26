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
	parser.add_argument('init', nargs=1,
		help='File containing list of pickled grids')
	parser.add_argument('-s', '--sudden', 
		action='store_true', help='help to suddenly change paramters w/o going through a gradual adjustment.')

	args=parser.parse_args()
	init=ascii.read(args.init[0])
	sudden=args.sudden

	for i in range(len(init)):
		#Opening the specified pickled galaxy object. Will use this parameters of the pickled grid.
		try:
			print init[i]['pickle']
			gal=dill.load(open(init[i]['pickle'], 'rb'))
		except:
			print 'Could not open pickle!'
			continue
		print init[i]['target']

		#Useful name for the output dir.
		try:
			gal.set_param('outdir', init[i]['outdir'])
		except IndexError:
			gal.set_param('outdir', gal.name+'/vw_{0}_{1}_{2}'.format(gal.vw_extra/1.E5, init[i]['param'], init[i]['target']))

		#Behavior of code depends on the type of param that is passed, it is a float which may be gradually adjusted then 
		#solve while gradually adjusting parameter, using the solve_adjust method.
		if type(init[i]['target'])==float and not sudden:
			gal.solve_adjust(5.*gal.tcross, init[i]['param'], init[i]['target'])
		else:
			gal.set_param(init[i]['param'], init[i]['target'])

		#If time is specified then run for that amount of time otherwise, run until conservation checks are specified 
		try:
			gal.solve(init[i]['time']*gal.tcross)
		except IndexError:
			gal.solve()

		galaxy.bash_command('cp '+init[i]['pickle']+' '+gal.outdir+'/init.p')


if __name__ == '__main__':
	main()
