#!/usr/bin/env python

import galaxy
import parker
import sol_check as sc

import astropy.constants as const
import numpy as np

import argparse

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

class CaseConfigParser(SafeConfigParser):
	def __init__(self):
		SafeConfigParser.__init__(self)
		self.optionxform = str

def params_parse(conf_file):
	config = CaseConfigParser()
	params_dict={}
	param_names=['Re','Re_s','visc_scheme', 'bdry']

	try:
		f=open(conf_file,'r')
		config.readfp(f)
	except: 
		return {}
	for name in param_names:
		try:
			params_dict[name]=ast.literal_eval(config.get('params',name))
		except:
			pass

	return params_dict

#Run hydro solver for a particular galaxy instance. 
def run_hydro(gal, time=5.):
	if gal.outdir=='.':
		gal.set_param('outdir', gal_name+'/vw_'+str(vw/1.E5))

	gal.solve(time*gal.tcross)
	gal.backup()


##Should add flag to compute all of the nuker galaxies
def main():
	parser=argparse.ArgumentParser(
		description='Code for generating nuker galaxies')
	parser.add_argument('init', nargs=1,
		help='File containing list of galaxies and config filed')

	args=parser.parse_args()
	cols=['vw_extra','rescale','index','time','config']
	default=[1.E8, 1., -1, 5.,'']

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
		try:
			start=galaxy.prepare_start(np.load(init[i]['save']+'/save.npz')['a'][init[i]['index']], rescale=init[i]['rescale'])

			gal=galaxy.NukerGalaxy(gal_name,gal_dict,init_array=start)
			gal.set_param('vw_extra', init[i]['vw_extra'])
		except:
			continue

		params_dict=params_parse(init[i]['config'])	
		print params_dict	
		for param in params_dict.keys():
			gal.set_param(param, params_dict[param])

		run_hydro(gal, init[i]['time'])




if __name__ == '__main__':
	main()
