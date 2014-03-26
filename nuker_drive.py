import hydro
import nuker
import parker
import astropy.constants as const
import numpy as np

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B .cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value

##Driver for each Nuker 
def nuker_driver(galaxy, eta=0.1, rho_0=1.E-23, temp=1.E7):
	galaxies=nuker.nuker_params()
	gparams=galaxies[galaxy]

	M_enc=nuker.get_M_enc(gparams)
	q=nuker.get_q(eta, params=gparams)
	rmin,rmax=grid_bdry(gparams['M'])


	d=dict(rho_0=rho_0, temp=temp, log=True, n=0.)
	params=dict(n=70, safety=0.6, Re=150.,  params=d,  floor=0.,
		logr=True, symbol='r', isot=False,  movies=True)

	grid=hydro.Grid(rmin, rmax, parker.background, gparams['M'], M_enc, q, **params)
	grid.solve(parker.tcross(rmin, rmax, 1.E7))


#Setting up grid boundary--scaled to mass
def grid_bdry(M, temp=1.E7):
	ct=np.sqrt(kb*temp/mp)
	#Nominal Parker radius.
	rp=G*M/(2*ct**2)
	#For now grid boundaries will simply be hard-coded multiple of rp
	rmin=0.1*rp
	rmax=10.*rp
	return [rmin, rmax]

