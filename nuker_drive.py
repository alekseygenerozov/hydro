#!/usr/bin/env python

import hydro
import nuker
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

#For initializing background profile...
temp=1.e7
rho_0=1.E-23
d=dict(rho_0=rho_0, temp=temp, log=True)


#Default params for solver
params=dict(n=70, safety=0.6, Re=90.,  params=d,  floor=0.,
    logr=True, symbol='r', isot=False,  movies=False, qpc=True, params_delta=(), Re_s=500., vw=0,
    veff=False, scale_heating=1.,outdir='.', bdry='default', visc_scheme='default', eps=1.)

#Run hydro solver for a particular galaxy instance. 
def run_hydro(galaxy, vw=5.E7, save='', rescale=1., index=-1, time=5., outdir='', ss=False):
	#Output directory to write to 
	if outdir:
		params['outdir']=outdir
		print outdir
	else:
		params['outdir']=galaxy.name+'/vw_'+str(vw/1.E5)
	#Prepare initial data
	saved=np.load(save+'/save.npz')['a']
	saved=saved[index][:,[0,1,2,3,-1]]
	if ss: 
		saved=hydro.extend_to_ss(saved)
	start=hydro.prepare_start(saved, rescale=rescale)
	#Set up the wind velocity parameter
	params['vw']=np.array(map(lambda r:np.sqrt(galaxy.sigma(r/pc)**2+(vw)**2), start[:,0]))
	tcross=parker.tcross(start[0,0],start[-1,0], 1.E7)
	params['tinterval']=0.05*tcross
	grid2=hydro.Grid(galaxy, init_array=start, **params)
	grid2.solve(time*tcross)

	# try:
	# 	for i in range(1,4):
	# 		fig=sc.cons_check(params['outdir'], logy=True, index=i)
	# 		fig.savefig(params['outdir']+'/cons'+str(i)+'_vw'+str(vw/1.E5)+'_'+galaxy.name+'.png')
	# except:
	# 	pass

	grid2.backup()
	# try:
	# 	ipyani.movie_save(params['outdir'], interval=1, ymin=[None, None, None, 10.**-25, -1., 10.**6], ymax=[None, None, None, 10.**-20, 2., 10.**8], logy=[True, True, True, True, False, True])
	# except:
	# 	pass


#Compute density profile from scratch
def run_hydro_scratch(galaxy, vw=5.E7, rmin=1.36E17, rmax=7.E19):
	#Get the radial grid
	params['outdir']=galaxy.name+'/vw_'+str(vw/1.E5)
	radii=np.logspace(np.log(rmin), np.log(rmax), params['n'], base=e)
	#Initialize isothermal evolution
	params['isot']=True
	params['bdry']='bp'
	#Note setting eps to 0 should turn off the gravitational effect of the stars 
	params['eps']=0.
	tcross=parker.tcross(rmin,rmax,1.E7)
	params['tinterval']=0.05*tcross
	params['vw']=np.array(map(lambda r:np.sqrt(galaxy.sigma(r/pc)**2+(vw)**2), radii))
	#Solving from the isothermal evolution
	grid2=hydro.Grid(galaxy, init=[rmin, rmax, parker.background], **params)
	grid2.solve(2.*tcross)
	grid2.isot_off()
	grid2.solve(2.*tcross)
	#Resetting boundary conditions
	grid2.set_param('bdry', 'default')
	grid2.solve(tcross)
	#Turning on the stellar potential
	grid2.set_param('eps',1.)
	grid2.solve(8.*tcross)



	#Save movies
	ipyani.movie_save(params['outdir'], interval=1, ymin=[None, None, None, 10.**-25, -1., 10.**6], ymax=[None, None, None, 10.**-20, 2., 10.**8], logy=[True, True, True, True, False, True])




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
	init=ascii.read(args.init)
	#Filling up gaps in the parameter file
	for i in range(len(params)):
		if np.in1d(params[i], init.colnames):
			continue
		else:
			cole=[default[i]]*len(init)
			col=Column(name=params[i], data=cole)
			init.add_column(col) 

	#Generate dictionary of nuker parameters for all galaxies
	gdict=nuker.nuker_params()
    #Iterate over all of the galaxies of interest
	for i in range(len(init)):
		try: 
			galaxy=nuker.Galaxy(init['gal'][i], gdict, eta=1.) 
		except:
			continue

		if init['save'][i]:
			run_hydro(galaxy, vw=init['vw'][i], save=init['save'][i], rescale=init['rescale'][i], index=int(init['index'][i]), time=init['time'][i],\
				outdir=init['outdir'][i], ss=init['ss'][i])
		else:
			run_hydro_scratch(galaxy, vw=init['vws'][i], rmin=init['rmin'][i], rmax=init['rmax'][i])



if __name__ == '__main__':
    main()
