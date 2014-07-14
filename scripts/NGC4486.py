#!/usr/bin/env python

import galaxy 
import numpy as np

gal_name='NGC4486'
save='/Users/aleksey/Second_Year_Project/hydro/batch_A2052/'+gal_name+'/vw_1000.0'
saved=np.load(save+'/save.npz')['a']
saved=saved[-1]

vw=5.E7
outdir=''
#Set up galaxy for run
start=galaxy.prepare_start(saved)
gal_dict=galaxy.nuker_params()
gal=galaxy.NukerGalaxy(gal_name, gal_dict, init_array=start)

if outdir:
	gal.set_param('outdir', outdir)
else:
	gal.set_param('outdir', gal_name+'/vw_'+str(vw/1.E5))

gal.set_param('vw_extra',vw)
# gal.set_param('Re_s', 500.)
# gal.set_param('visc_scheme', 'cap_visc')

gal.solve(5.*gal.tcross)