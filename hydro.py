import numpy as np
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import e

import subprocess
import astropy.constants as const


#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value


##Run a command from the bash shell
def bash_command(cmd):
    process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process.communicate()[0]
    # process.wait()
    # return process



class Zone:
	"""Class to store zones of (static) grid, to be used to solve
	Euler equations numerically. Contains radius (rad), primitive 
	variables and pressures. Also contains mass enclosed inside 
	cell.

	"""
	def __init__(self, rad=1.E16, prims=(0,0,0), M=1.E6):
		#Radius of grid zone and mass enclosed
		self.rad=rad
		self.M=M

		#Primitive variables
		self.log_rho=prims[0]
		self.vel=prims[1]
		self.temp=prims[2]
		#Temporary storage for all of the primitive variables
		# self.rho_tmp=self.rho
		# self.vel_tmp=self.vel
		# self.temp_tmp=self.temp

		#Updating non-primitive variables within zone
		self.update_aux()

	#Equation of state. Note this could be default in the future there could be functionality to override this.
	def eos(self):
		#gamma=5./3.
		#gamma=1.01
		mu=1.
		self.pres=self.rho*kb*self.temp/(mu*mp)
		self.cs=np.sqrt(kb*self.temp/(mu*mp))

	# #Method which updates all of the primitive variables in the grid after each time step
	# def update(self):
	# 	for field in ['rho', 'vel', 'temp']:
	# 		val=getattr(self, field+'_tmp')
	# 		setattr(self, field, val)

	#Method which will be used to update non-primitive vars. 
	def update_aux(self):
		self.rho=np.exp(self.log_rho)
		self.eos()
		self.r2vel=self.rad**2*self.vel
		self.frho=self.rad**2*self.vel*self.rho
		# self.visc=self.cs*self.vel


	#Finding the maximum transport speed across the zone
	def alpha_max(self):
		return max([np.abs(self.vel+self.cs), np.abs(self.vel-self.cs)])

	def bernoulli(self):
		return 0.5*self.vel**2+self.cs**2*self.log_rho-G*self.M/self.rad











class Grid:
	"""Class stores (static) grid for solving Euler equations numerically"""

	def __init__(self, r1, r2, f_initial, n=100, M=1.E6*M_sun, Mdot=1., num_ghosts=3, periodic=False, safety=0.6, Re=100., q=None, params=dict(),
		floor=1.e-30, symbol='rs', logr=True):
		assert r2>r1
		assert n>1

		self.fields=['log_rho', 'vel', 'temp']
		self.out_fields=['t', 'rad' ,'rho', 'vel', 'temp', 'frho']

		self.M=M
		if q:
			self.q=q

		self.Mdot=Mdot
		#Grid will be stored as list of zones
		self.grid=[]
		delta=float(r2-r1)/n
		self.safety=safety
		self.Re=Re

		self.floor=floor

		#Setting up the grid (either logarithmic or linear in r)
		self.logr=logr
		if logr:
			self.radii=np.logspace(np.log(r1), np.log(r2), n, base=e)
		else:
			self.radii=np.linspace(r1+(delta/2.), r2-(delta/2.), n)
		#Attributes to store length of the list as well as start and end indices (useful for ghost zones)
		self.length=n
		self.start=0
		self.end=n-1

		self.time_derivs=np.zeros(n, dtype={'names':[self.fields[0], self.fields[1], self.fields[2]], 'formats':['float64',
			'float64', 'float64']})


		#Initializing the grid using the initial value function f_initial
		for rad in self.radii:
			prims=f_initial(rad, **params)
			self.grid.append(Zone(rad=rad, prims=prims, M=M))
		if periodic:
			self.periodic=True
		else:
			self.periodic=False
			self._add_ghosts(num_ghosts=num_ghosts)


		#Computing differences between all of the grid elements 
		delta=np.diff(self.radii)
		#print delta
		self.delta=np.insert(delta, 0, delta[0])
		#print self.delta

		delta_log=np.diff(np.log(self.radii))
		self.delta_log=np.insert(delta_log, 0, delta_log[0])


		self.delta_t=0
		self.time_cur=0
		self.total_time=0
		self.time_target=0

		self.saved=np.empty([0, self.length, 6])
		self.time_stamps=[]

		self.symbol=symbol



	def q(self, rad):
		return 0.

	def _bernoulli_diff(self):
		return self.grid[self.end].bernoulli()-self.grid[self.start].bernoulli()

	def _bernoulli_check(self):
		#Integating the momentum source term
		integral=0.
		for i in range(self.start, self.end):
			integral+=-self.q(self.radii[i])*self.grid[i].vel*self.delta[i]/self.grid[i].rho
			
		return [self.grid[self.start].bernoulli(),self.grid[self.end].bernoulli(), self._bernoulli_diff(),integral]


	#Adding ghost zones onto the edges of the grid (moving the start of the grid)
	def _add_ghosts(self, num_ghosts=3):
		self.start=num_ghosts
		self.end=self.end-num_ghosts
	

	#Interpolating field (using zones wiht indices i1 and i2) to radius rad 
	def _interp_zones(self, rad, i1, i2, field):
		rad1=self.grid[i1].rad
		rad2=self.grid[i2].rad

		val1=getattr(self.grid[i1],field)
		val2=getattr(self.grid[i2],field)
		return np.interp(np.log(rad), [np.log(rad1), np.log(rad2)], [val1, val2])


	#Method to update the ghost cells
	def _update_ghosts(self):
		start_cell=copy.deepcopy(self.grid[self.start])
		r_start=start_cell.rad
		log_rho_start=start_cell.log_rho
		#Updating the end ghost zones, extrapolating using a power law density
		for i in range(0, self.start):
			log_rho=-1.5*np.log(self.grid[i].rad/r_start)+log_rho_start
			self.grid[i].log_rho=log_rho
			self.grid[i].rho=np.exp(log_rho)

		mdot=start_cell.rho*start_cell.vel*start_cell.rad**2

		#Updating velocities in the ghost cells
		for i in range(0, self.start):
			vel=mdot/self.grid[i].rho/self.grid[i].rad**2
			setattr(self.grid[i], 'vel', vel)
			self.grid[i].update_aux()

		end_cell=copy.deepcopy(self.grid[self.end])
		r_end=end_cell.rad
		log_rho_end=end_cell.log_rho

		#Updating the end ghost zones, extrapolating using a power law density
		for i in range(self.end+1, self.length):
			log_rho=-2.*np.log(self.grid[i].rad/r_end)+log_rho_end
			self.grid[i].log_rho=log_rho
			self.grid[i].rho=np.exp(log_rho)

		#Updating the velocities at the end of the grid assuming a constant mdot.
		mdot=end_cell.rho*end_cell.vel*end_cell.rad**2
		for i in range(self.end+1, self.length):
			vel=mdot/self.grid[i].rho/self.grid[i].rad**2
			self.grid[i].vel=vel
			self.grid[i].update_aux()





	#Get stencil for a particular zone
	def _get_stencil(self, i, left=3, right=3):
		#Check that we will not fall off the edge of our grid
		if self.periodic:
			stencil=[]
			for j in range(-left, right+1):
				stencil.append(self.grid[(i+j)%self.length])
			return stencil
		else:
			assert i-left>=0
			assert i+right<self.length

			return self.grid[i-left:i+right+1]


	#Getting derivatives for a given field (density, velocity, etc.). If second is set to be true then the discretized 2nd
	#deriv is evaluated instead of the first
	def get_spatial_deriv(self, i, field, second=False):
	# def get_spatial_deriv(self, i, func=getattr, second=False, args=()):
		left=3
		right=3
		num_zones=left+right+1
		#Getting stencil for current grid point
		stencil=self._get_stencil(i, left=left, right=right)
		field_list=np.zeros(num_zones)
		for i in range(num_zones):
			field_list[i]=getattr(stencil[i], field)

		#Coefficients we will use.
		coeffs=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.

		# if second:
		# 	coeffs2=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.*self.delta[i])
		# 	return np.sum(field_list*coeffs2)/self.delta[i]
		# else:
		# 	return np.sum(field_list*coeffs)/self.delta[i]
		if second:
			if self.logr:
				coeffs2=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.*self.delta_log[i])
				return (1./self.grid[i].rad**2/self.delta_log[i])*(np.sum(field_list*coeffs2)
					-np.sum(field_list*coeffs))
			else:
				coeffs2=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.*self.delta[i])
				return np.sum(field_list*coeffs2)/self.delta[i]
				
		else:
			if self.logr:
				return np.sum(field_list*coeffs)/(self.delta_log[i]*self.radii[i])
			else:
				return np.sum(field_list*coeffs)/self.delta[i]


	def get_laplacian(self, i, field):
		return self.get_spatial_deriv(i, field, second=True)+(2./self.radii[i])*(self.get_spatial_deriv(i, field))


	#Evaluate Courant condition for the entire grid. This gives us an upper bound on the time step we may take 
	def _cfl(self):
		alpha_max=0.
		delta_t=np.zeros(self.length)
		#Finding the maximum transport speed across the grid
		for i in range(0, self.length):
			alpha_max=self.grid[i].alpha_max()
			delta_t[i]=self.safety*self.delta[i]/alpha_max

			# if zone_alpha_max>alpha_max:
			# 	alpha_max=zone_alpha_max
		#Setting the time step
		cfl_delta_t=np.min(delta_t)
		target_delta_t=self.time_target-self.time_cur
		self.delta_t=min([target_delta_t, cfl_delta_t])

	#Wrapper for evaluating the time derivative of all fields
	def dfield_dt(self, i, field):
		if field=='rho':
			return self.drho_dt(i)
		elif field=='vel':
			return self.dvel_dt(i)
		elif field=='temp':
			return self.dtemp_dt(i)
		elif field=='log_rho':
			return self.dlog_rho_dt(i)
		else:
			return 0.

	#Partial derivative of density with respect to time
	def dlog_rho_dt(self, i):
		rad=self.grid[i].rad
		rho=self.grid[i].rho
		vel=self.grid[i].vel

		#return -cs*drho_dr+art_visc*drho_dr_second
		return -vel*self.get_spatial_deriv(i, 'log_rho')-(1/rad**2)*self.get_spatial_deriv(i, 'r2vel')+self.q(rad)/rho

	#Partial derivative of density with respect to time
	def drho_dt(self, i):
		rad=self.grid[i].rad

		#return -cs*drho_dr+art_visc*drho_dr_second
		return -(1./rad)**2*self.get_spatial_deriv(i, 'frho')+self.q(rad)

	#Evaluating the partial derivative of velocity with respect to time
	def dvel_dt(self, i):
		#return 0.
		rad=self.grid[i].rad
		vel=self.grid[i].vel
		rho=self.grid[i].rho
		temp=self.grid[i].temp
		# assert rad>0
		# #If the density zero of goes negative return zero to avoid numerical issues
		if rho<=self.floor:
		 	return 0

		# dpres_dr=self.get_spatial_deriv(i, 'pres')
		dlog_rho_dr=self.get_spatial_deriv(i, 'log_rho')
		dtemp_dr=self.get_spatial_deriv(i, 'temp')
		dv_dr=self.get_spatial_deriv(i, 'vel')
		lap_vel=self.get_laplacian(i, 'vel')
		#lap_vel=self.get_spatial_deriv(i, 'vel', 'second')
		art_visc=min(self.grid[i].cs,  np.abs(self.grid[i].vel))*(self.radii[self.end]-self.radii[0])/self.Re

		#Need to be able to handle for general potential in the future
		return -vel*dv_dr-dlog_rho_dr*(kb*temp/mp)+(kb/mp)*dtemp_dr-(G*self.M)/rad**2+art_visc*lap_vel-(self.q(rad)*vel/rho)

	#Evaluating the partial derivative of temperature with respect to time.
	def dtemp_dt(self, i):
		return 0.

	#Evolve the system forward for time, time. If field is specified then we create a movie showing the solution for 
	#the field as a function of time
	def evolve(self, time, max_steps=None,  analytic_func=[None, None, None, None, None, None]):
		#Initialize the current and target times
		self.time_cur=0
		self.time_target=time
		num_steps=0

		params_name='params'
		fparams=file(params_name, 'w')
		fparams.write('Re={0} rin={1:8.7e} rout={2:8.7e} floor={3:8.7e} n={4}'.format(self.Re, self.radii[0], 
			self.radii[-1], self.floor, self.length))

		out_name='tmp'
		bash_command('rm '+out_name)
		fout=file(out_name, 'a')
		#While we have not yet reached the target time
		while self.time_cur<time:
			# print self.time_cur
			# self.save()
			# np.savetxt(out, self.saved[-1])
			if num_steps%5==0:
				self.save()
				np.savetxt(fout, self.saved[-1])
				print self.total_time/self.time_target
			#self.save()

			self._step()
			# #Taking a single time-step
			# try:
			# 	self._step()
			# except:
			# 	break	
			
			self.time_cur+=self.delta_t
			self.total_time+=self.delta_t
			num_steps+=1

			if max_steps:
				if num_steps>max_steps:
					break

		for i in range(1, len(self.out_fields)):
			self.animate(index=i, analytic_func=analytic_func[i])

		plt.clf()


	#Create movie of solution
	def animate(self,  analytic_func=None, index=1):
		if analytic_func:
			vec_analytic_func=np.vectorize(analytic_funcs)
		def update_img(n):
			time=self.time_stamps[n]
			sol.set_ydata(self.saved[n,:,index])

			if analytic_func:
				analytic_sol.set_ydata(vec_analytic_func(self.radii))
			label.set_text(str(time))
		ymin=np.min(self.saved[:,:,index])
		ymax=np.max(self.saved[:,:,index])

		fig,ax=plt.subplots()
		label=ax.text(0.02, 0.95, '', transform=ax.transAxes)	

		if index==2:
			ax.set_yscale('log')
			ax.set_ylim(self.floor, 10.**3*self.floor)
		sol,=ax.plot(self.radii, self.saved[0,:,index], self.symbol)

		if index==3:
			ax.set_ylim(-3, 3)

		if analytic_func:
			analytic_sol,=ax.plot(self.radii, vec_analytic_func(self.radii))
		
		#Exporting animation
		sol_ani=animation.FuncAnimation(fig,update_img,len(self.saved),interval=50)
		sol_ani.save('sol_'+self.out_fields[index]+'.mp4', dpi=200)

		# plt.close()

	#Save the state of the grid
	def save(self):
		#fields=['rho', 'vel', 'temp', 'frho']
		grid_prims=np.zeros((len(self.out_fields), self.length))
		for i in range(len(self.out_fields)):
			if i==0:
				grid_prims[i].fill(self.total_time)
			else:
				grid_prims[i]=self.get_field(self.out_fields[i])[1]
				if self.out_fields[i]=='vel':
					grid_prims[i]=grid_prims[i]/self.get_field('cs')[1]

		#Saving the state of the grid within list
		#self.saved.append((self.total_time, np.transpose(grid_prims)))
		self.saved=np.append(self.saved,[np.transpose(grid_prims)],0)
		self.time_stamps.append(self.total_time)

	#Clear all of the info in the saved list
	def clear_saved(self):
		self.saved=np.empty([self.length, 4])
		self.time_stamps=[]

	#Take a single step in time
	def _step(self):
		gamma=[8./15., 5./12., 3./4.]
		zeta=[-17./60.,-5./12.,0.]

		self._cfl()

		for substep in range(3):
			self._sub_step(gamma[substep], zeta[substep])
			for i in range(0, self.length):
				#self.grid[i].u()
				self.grid[i].update_aux()
			self._update_ghosts()

	#Substeps
	def _sub_step(self, gamma, zeta):
		#Calculating the derivatives of the field
		for field in self.fields:
			for i in range(self.start,self.end+1):
				self.time_derivs[i][field]=self.dfield_dt(i, field)

		#Updating the values in the grid: have to calculate the time derivatives for all relevant fields before this step
		for field in self.fields:
			for i in range(self.start, self.end+1):
				f=getattr(self.grid[i],field)+gamma*self.time_derivs[i][field]*self.delta_t
				g=f+zeta*self.time_derivs[i][field]*self.delta_t
				setattr(self.grid[i], field, g)


	#Extracting the array corresponding to a particular field from the grid as well as an array of radii
	def get_field(self, field):
		field_list=np.zeros(self.length)
		rad_list=np.zeros(self.length)
		for i in range(0, self.length):
			field_list[i]=getattr(self.grid[i], field)
			rad_list[i]=getattr(self.grid[i],'rad')

		return [rad_list, field_list]









		
































