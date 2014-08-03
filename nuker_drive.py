#!/usr/bin/env python

import galaxy
import parker
import sol_check as sc

import astropy.constants as const
import numpy as np

import argparse
from params_parse import params_parse

import ipyani
from math import e

from astropy.io import ascii
from astropy.table import Column

from ConfigParser import SafeConfigParser
import ast

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
		description='Code for generating nuker galaxies')
	parser.add_argument('init', nargs=1,
		help='File containing list of galaxies and config filed')

	args=parser.parse_args()
	cols=['vw_extra','rescale','index','time','length','config']
	default=[1.E8, 1., -1, None,None,'']

	init=ascii.read(args.init[0])
	for i in range(len(cols)):
		if np.in1d(cols[i], init.colnames):
			continue
		else:
			cole=[default[i]]*len(init)
			col=Column(name=cols[i], data=cole)
			init.add_column(col) 

	gal_dict=galaxy.nuker_params()
	for i in range(len(init)):
		gal_name=init[i]['gal']
		print init[i]['rescale'],init[i]['index'],init[i]['length']

		try:
			gal=galaxy.NukerGalaxy.from_dir(init[i]['gal'],init[i]['save'], rescale=init[i]['rescale'],\
				index=init[i]['index'], length=init[i]['length'])
			gal.set_param('vw_extra', init[i]['vw_extra'])
		except:
			print 'Unable to initialize galaxy'
			continue
		print gal.vel

		params_dict=params_parse(init[i]['config'])	
		print params_dict	
		for param in params_dict.keys():
			gal.set_param(param, params_dict[param])

		if gal.outdir=='.':
			gal.set_param('outdir', gal.name+'/vw_'+str(gal.vw_extra/1.E5))
		try:
			time=init[i]['time']*gal.tcross
		except:
			time=None
		gal.solve(time)
		gal.backup()




if __name__ == '__main__':
	main()
