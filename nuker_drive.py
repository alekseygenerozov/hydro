#!/usr/bin/env python

import hydro
import nuker
import parker
import astropy.constants as const
import numpy as np

import argparse
from astropy.io import ascii

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B .cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value

#for initializing background profile...
temp=1.e7
rho_0=1.E-23
d=dict(rho_0=rho_0, temp=temp, log=True)

#default params for solver
params=dict(n=70, safety=0.6, Re=90.,  params=d,  floor=0.,
    logr=True, symbol='r', isot=False,  movies=False, qpc=True, params_delta=(), Re_s=500., vw=0,
    veff=False, scale_heating=1.,outdir='.', bdry='default', visc_scheme='default')

#Run hydro solver for a particular galaxy instance. 
def run_hydro(galaxy, vw=5.E7, save=''): 
	if save:
		#Prepare initial data
		saved=np.load(save+'/save.npz')['a']
		rescale=galaxy.params['M']/(10.**7.11*M_sun)
		start=hydro.prepare_start(saved[-1], rescale=rescale)
		#Output directory should reflect galaxy name, and value of additional vw parameter
		params['outdir']=galaxy.name+'/vw_'+str(vw)
		#Set up the wind velocity parameter
		params['vw']=np.array(map(lambda r:np.sqrt(galaxy.sigma(r/pc)**2+(vw)**2), start[:,0]))
		print start[:,0]
		tcross=parker.tcross(start[0,0],start[-1,0], 1.E7)
		print tcross
		# params['tinterval']=0.05*tcross
		# grid2=hydro.Grid(galaxy, init_array=start, **params)
		# grid2.solve(0.06*tcross)
		# #Save movies
		# ipyani.movie_save(params['outdir'], interval=1, ymin=[None, None, None, 10.**-25, -1., 10.**6], ymax=[None, None, None, 10.**-20, 2., 10.**8], logy=[True, True, True, True, False, True])
	#Compute density profile from scratch; to be implemented later...
	else:
		pass


##Should add flag to compute all of the nuker galaxies
def main():
	parser=argparse.ArgumentParser(
        description='Code for generating nuker galaxies')
	parser.add_argument('-i', '--init',
		help='Name of file containing galaxy names, vws.')
	#Read input file
	args=parser.parse_args()
	init=ascii.read(args.init)
	gals=init['gal']
	vws=init['vw']
	saves=init['save']

	#Generate dictionary of nuker parameters for all galaxies
	gdict=nuker.nuker_params()
    #Iterate over all of the galaxies of interest
	for i in range(len(gals)):
		try: 
			galaxy=nuker.Galaxy(gals[i], gdict, eta=1.) 
		except:
			continue

		run_hydro(galaxy, vw=vws[i], save=saves[i])



if __name__ == '__main__':
    main()