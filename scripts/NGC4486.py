#!/usr/bin/env python

import galaxy 
import numpy as np

gal_name='NGC4486_1'
save='/vega/astro/users/ag3293/NGC4486_1/vw_500.0'
saved=np.load(save+'/save.npz')['a']
saved=saved[-1]

outdir=''
vw=5.E7
#Set up galaxy for run
start=galaxy.prepare_start(saved)
gal_dict=galaxy.nuker_params()
gal=galaxy.NukerGalaxy('NGC4486', gal_dict, init_array=start)

if outdir:
	gal.set_param('outdir', outdir)
else:
	gal.set_param('outdir', gal_name+'/vw_'+str(vw/1.E5))

gal.set_param('vw_extra',vw)
gal.set_param('Re_s', 500.)
gal.set_param('visc_scheme', 'cap_visc')

gal.solve(5.*gal.tcross)
gal.backup()