#!/usr/bin/env python

import pickle
import parker
import astropy.constants as const

c=const.c.cgs.value

x=10.**7.11/(3.6*10.**6)
rmin=3.8E16*x
rmax=7.E19
tcross=parker.tcross(rmin, rmax, 1.E7)

grid2=pickle.load( open( "grid_backup.p", "rb" ) )
vw=2.5E7

grid2.set_param('Re_s', 1000.)
grid2.solve_adjust(2.*tcross, 'vw', c*((grid2.rg/grid2.radii)*(grid2.M_tot/grid2.M_bh)+(vw/c)**2)**0.5)
grid2.solve(4.*tcross)