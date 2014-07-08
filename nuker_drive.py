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

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B .cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value


#Run hydro solver for a particular galaxy instance. 
def run_hydro(gal_name, vw=5.E7, save='', rescale=1., index=-1, time=5., outdir='', ss=False):
	#Prepare initixal data
	saved=np.load(save+'/save.npz')['a']
	saved=saved[index][:,[0,1,2,3,-1]]

	#Set up galaxy for run
	start=galaxy.prepare_start(saved, rescale=rescale)
	gal_dict=galaxy.nuker_params()
	gal=galaxy.NukerGalaxy(gal_name, gal_dict, init_array=start)
	if outdir:
		gal.set_param('outdir', outdir)
	else:
		gal.set_param('outdir', gal_name+'/vw_'+str(vw/1.E5))

	gal.set_param('vw_extra',vw)
	gal.solve(time*gal.tcross)




#Compute density profile from scratch
def run_hydro_scratch(gal_name, vw=5.E7, rmin=1.36E17, rmax=7.E19, outdir=''):
	gal_dict=galaxy.nuker_params()
	#Solving from the isothermal evolution
	gal=galaxy.NukerGalaxy(gal_name, gal_dict, init=[rmin, rmax, parker.background,{'log': True, 'temp': 1.E7, 'rho_0': 1e-23}])
	if outdir:
		gal.set_param('outdir', outdir)
	else:
		gal.set_param('outdir', gal.name+'/vw_'+str(vw/1.E5))
	gal.isot_on()
	gal.set_param('bdry', 'bp')
	gal.set_param('eps', 0.)
	gal.set_param('vw_extra', vw)
	gal.solve(2.*gal.tcross)
	gal.isot_off()
	gal.solve(2.*gal.tcross)
	#Resetting boundary conditions
	gal.set_param('bdry', 'default')
	gal.solve(gal.tcross)
	#Turning on the stellar potential
	gal.set_param('eps',1.)
	gal.solve(8.*tcross)

	#Save movies
	ipyani.movie_save(outdir, interval=1, ymin=[None, None, None, 10.**-25, -1., 10.**6], ymax=[None, None, None, 10.**-20, 2., 10.**8], logy=[True, True, True, True, False, True])




##Should add flag to compute all of the nuker galaxies
def main():
	parser=argparse.ArgumentParser(
        description='Code for generating nuker galaxies')
	parser.add_argument('init', nargs=1,
		help='Name of file containing galaxy names, vws.')
	#Read input file
	params=['gal','vw','save','rescale','rmin','rmax','index','time','outdir','ss']
	default=['NGC4551',1.E8,None,1.,None,None,-1,5.,'',False]
	args=parser.parse_args()
	init=ascii.read(args.init[0])
	print init
	#Filling up gaps in the parameter file
	for i in range(len(params)):
		if np.in1d(params[i], init.colnames):
			continue
		else:
			cole=[default[i]]*len(init)
			col=Column(name=params[i], data=cole)
			init.add_column(col) 

    #Iterate over all of the galaxies of interest
	for i in range(len(init)):
		if init['save'][i]:
			run_hydro(init['gal'][i], vw=init['vw'][i], save=init['save'][i], rescale=init['rescale'][i], index=int(init['index'][i]), time=init['time'][i],\
				outdir=init['outdir'][i], ss=init['ss'][i])
		else:
			run_hydro_scratch(init['gal'][i], vw=init['vw'][i], rmin=init['rmin'][i], rmax=init['rmax'][i], outdir=init['outdir'][i])



if __name__ == '__main__':
    main()
