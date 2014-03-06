import hydro
import nuker

def nuker_driver(galaxy, eta):
	galaxies=nuker.nuker_params()
	params=galaxies[galaxy]

	grad_phi=nuker.get_grad_phi(params)
	q=nuker.get_q(params)

	temp=1.E7
	ct=np.sqrt(kb*temp/mp)
	rp=G*params['M']/(2*ct**2)
	rmin=0.1*rp
	rmax=10.*rp