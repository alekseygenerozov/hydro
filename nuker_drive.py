import hydro
import nuker

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value

##Driver for each Nuker 
def nuker_driver(galaxy, eta):
	galaxies=nuker.nuker_params()
	params=galaxies[galaxy]

	grad_phi=nuker.get_grad_phi(params)
	q=nuker.get_q(eta, params=params)
	rmin,rmax=grid_bdry(params['M'])

#Setting up grid boundary--scaled to mass
def grid_bdry(M, temp=1.E7):
	ct=np.sqrt(kb*temp/mp)
	#Nominal Parker radius.
	rp=G*M/(2*ct**2)
	#For now grid boundaries will simply be hard-coded multiple of rp
	rmin=0.1*rp
	rmax=10.*rp
	return [rmin, rmax]


