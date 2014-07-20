#!/usr/bin/env python

import galaxy 
import numpy as np

gal_name='NGC3605'
save='/vega/astro/users/ag3293/'+gal_name+'_1/vw_200.0'
saved=np.load(save+'/save.npz')['a']
saved=saved[-1]

outdir=''
vw=2.E7
#Set up galaxy for run
start=galaxy.prepare_start(saved)
gal_dict=galaxy.nuker_params()
gal=galaxy.NukerGalaxy(gal_name, gal_dict, init_array=start)

if outdir:
	gal.set_param('outdir', outdir)
else:
	gal.set_param('outdir', gal_name+'/vw_'+str(vw/1.E5))

gal.set_param('vw_extra',vw)
gal.set_param('Re_s', 1.E20)
gal.solve(3.*gal.tcross)
gal.set_param('visc_scheme', 'cap_visc')
gal.solve(3.*gal.tcross)

gal.backup()