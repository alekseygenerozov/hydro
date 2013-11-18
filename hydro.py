import numpy as np
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#Need better way of dealing with constants...
G=6.67E-8
kb=1.38E-16
mp=1.67E-24
M_sun=2.E33





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
		self.rho=prims[0]
		self.vel=prims[1]
		self.temp=prims[2]
		#Temporary storage for all of the primitive variables
		self.rho_tmp=self.rho
		self.vel_tmp=self.vel
		self.temp_tmp=self.temp

		#Updating non-primitive variables within zone
		self.update_aux()

	#Equation of state. Note this could be default in the future there could be functionality to override this.
	def eos(self):
		gamma=5./3.
		mu=1.
		self.pres=self.rho*kb*self.temp/(mu*mp)
		self.cs=np.sqrt(gamma*kb*self.temp/(mu*mp))

	#Method which updates all of the primitive variables in the grid after each time step
	def update(self):
		for field in ['rho', 'vel', 'temp']:
			val=getattr(self, field+'_tmp')
			setattr(self, field, val)

	#Method which will be used to update non-primitive vars. 
	def update_aux(self):
		self.eos()
		self.frho=self.rad**2*self.vel*self.rho
		# self.visc=self.cs*self.vel


	#Finding the maximum transport speed across the zone
	def alpha_max(self):
		return max([np.abs(self.vel+self.cs), np.abs(self.vel-self.cs)])










class Grid:
	"""Class stores (static) grid for solving Euler equations numerically"""

	def __init__(self, r1, r2, f_initial, n=100, M=1.E6*M_sun, Mdot=1., num_ghosts=3, periodic=False, safety=0.6, Re=100., params=dict()):
		assert r2>r1
		assert n>1



		self.M=M
		self.Mdot=Mdot
		#Grid will be stored as list of zones
		self.grid=[]
		delta=float(r2-r1)/n
		self.safety=safety

		self.Re=Re
		#Getting list of radii

		self.radii=np.linspace(r1+(delta/2.), r2-(delta/2.), n)
		#Spacing of grid, which we for now assume to be uniform
		self.delta=self.radii[1]-self.radii[0]
		#Attributes to store length of the list as well as start and end indices (useful for ghost zones)
		self.length=n
		self.start=0
		self.end=n-1
		#Initializing the grid using the initial value function f_initial
		for rad in self.radii:
			prims=f_initial(rad, **params)
			self.grid.append(Zone(rad=rad, prims=prims, M=M))
		if periodic:
			self.periodic=True
		else:
			self.periodic=False
			self._add_ghosts(num_ghosts=num_ghosts)
		self.delta_t=0
		self.time_cur=0
		self.total_time=0
		self.time_target=0

		self.saved=[]
		self.bdry_rho=True






	#Adding ghost zones onto the edges of the grid (moving the start of the grid)
	def _add_ghosts(self, num_ghosts=3):
		self.start=num_ghosts
		self.end=self.end-num_ghosts
		#self.length=self.length+2*num_ghosts
		#Append and prepend ghost1 and ghost2 n times
		#print num_ghosts
		# for i in range(num_ghosts):
		# 	print self.grid[0]
		# 	ghost1=copy.deepcopy(self.grid[0])
		# 	ghost2=copy.deepcopy(self.grid[-1])
		# 	ghost1.rad=self.grid[0].rad-self.delta
		# 	ghost2.rad=self.grid[-1].rad+self.delta

		# 	self.grid.insert(0,ghost1)
		# 	self.grid.append(ghost2)
		#Start keeps track of where the actual grid starts	
	

	#Interpolating field (using zones wiht indices i1 and i2) to radius rad 
	def _interp_zones(self, rad, i1, i2, field):
		rad1=self.grid[i1].rad
		rad2=self.grid[i2].rad

		val1=getattr(self.grid[i1],field)
		val2=getattr(self.grid[i2],field)
		return np.interp(rad, [rad1, rad2], [val1, val2])


	#Method to update the ghost cells
	def _update_ghosts(self):
		# first_cell=copy.deepcopy(self.grid[self.start])
		# mdot=first_cell.rho*first_cell.vel*first_cell.rad**2
		
		# for i in range (0, self.start):
		# 	if self.bdry_rho:
		# 		vel=mdot/self.grid[i].rho/self.grid[i].rad**2
		# 		setattr(self.grid[i], 'vel', vel)
		# 		self.grid[i].update_aux()
		# 		self.bdry_rho=False
		# 	else:
		# 		rho=mdot/self.grid[i].vel/self.grid[i].rad**2
		# 		setattr(self.grid[i], 'rho', rho)
		# 		self.grid[i].update_aux()
		# 		self.bdry_rho=True
		# #Interpolating the density for all of the ghost zones
		for i in range(1, self.start):
			rho=self._interp_zones(self.grid[i].rad, 0, self.start, 'rho')
			setattr(self.grid[i], 'rho', rho)
		#Calculating the mdot in the first real zone
		first_cell=copy.deepcopy(self.grid[self.start])
		mdot=first_cell.rho*first_cell.vel*first_cell.rad**2

		#Updating velocities in the ghost cells
		for i in range(0, self.start):
			vel=mdot/self.grid[i].rho/self.grid[i].rad**2
			setattr(self.grid[i], 'vel', vel)
			self.grid[i].update_aux()

		#Updating the end ghost zones, copy everything except for the radius
		for i in range(self.end+1, self.length):
			tmp=copy.deepcopy(self.grid[self.end])
			rad=self.grid[i].rad
			self.grid[i]=tmp	
			self.grid[i].rad=rad



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
		left=3
		right=3
		num_zones=left+right+1
		#Getting stencil for current grid point
		stencil=self._get_stencil(i, left=left, right=right)
		field_list=np.zeros(num_zones)
		for i in range(num_zones):
			field_list[i]=getattr(stencil[i], field)

		#Coefficients we will use.
		if second:
			coeffs=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.*self.delta)
		else:	
			coeffs=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.
		return np.sum(field_list*coeffs)/self.delta


	#Evaluate Courant condition for the entire grid. This gives us an upper bound on the time step we may take 
	def _cfl(self):
		alpha_max=0.
		#Finding the maximum transport speed across the grid
		for zone in self.grid:
			zone_alpha_max=zone.alpha_max()
			if zone_alpha_max>alpha_max:
				alpha_max=zone_alpha_max
		#Setting the time step
		cfl_delta_t=self.safety*(self.delta/alpha_max)
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
		else:
			return 0.

	#Partial derivative of density with respect to time
	def drho_dt(self, i):
		rad=self.grid[i].rad

		#return -cs*drho_dr+art_visc*drho_dr_second
		return -(1./rad)**2*self.get_spatial_deriv(i, 'frho')

	#Evaluating the partial derivative of velocity with respect to time
	def dvel_dt(self, i):
		#return 0.
		rad=self.grid[i].rad
		vel=self.grid[i].vel
		rho=self.grid[i].rho
		# assert rad>0
		# #If the density zero of goes negative return zero to avoid numerical issues
		# if rho<=0:
		# 	return 0


		dpres_dr=self.get_spatial_deriv(i, 'pres')
		dv_dr=self.get_spatial_deriv(i, 'vel')
		dv_dr_second=self.get_spatial_deriv(i, 'vel', second=True)
		art_visc=min(self.grid[i].cs, self.grid[i].vel)*(self.radii[self.end]-self.radii[0])/self.Re

		return -vel*dv_dr-(1./rho)*dpres_dr-(G*self.M)/rad**2+art_visc*dv_dr_second

	#Evaluating the partial derivative of temperature with respect to time.
	def dtemp_dt(self, i):
		return 0.

	#Evolve the system forward for time, time. If field is specified then we create a movie showing the solution for 
	#the field as a function of time
	def evolve(self, time, max_steps=None,  analytic_func=[None, None, None, None]):
		#Initialize the current and target times
		self.time_cur=0
		self.time_target=time
		num_steps=0
		#ims=[]
		#fig,ax=plt.subplots()
		#While we have not yet reached the target time
		while self.time_cur<time:
			# print self.time_cur
			if num_steps%5==0:
				self.save()
				# field_sol=self.get_field(field_animate)
				# radii=field_sol[0]
				# #If analytic solution has been passed to the method
				# if analytic_func:
				# 	vec_analytic_func=np.vectorize(analytic_func)
				# 	field_analytic=vec_analytic_func(radii, self.time_cur)
				# 	plt.ylim([ 1.5*np.min(field_analytic), 1.5*np.max(field_analytic)])
				# 	ax.text(0.1, 1, str(self.time_cur))
				# 	ims.append(ax.plot( radii, field_analytic, 'b', radii, field_sol[1], 'rs'))


			#Updating each of the primitive variable fields.	
			for field in ['rho', 'vel', 'temp']:
				self._step(field)
			for i in range(0, self.length):
				self.grid[i].update()
				self.grid[i].update_aux()
			self._update_ghosts()
			#Updating the time
			self.time_cur+=self.delta_t
			self.total_time+=self.delta_t
			num_steps+=1

			if max_steps:
				if num_steps>max_steps:
					break

		for i in range(4):
			self.animate(index=i, analytic_func=analytic_func[i])
		# if field_animate:
		# 	self.animate(analytic_func=analytic_func)
			# field_ani=animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,blit=True)
			# field_ani.save('sol_'+field_animate+'.mp4', dpi=200)
		plt.clf()


	#Create movie of solution
	def animate(self,  analytic_func=None, index=1):
		if analytic_func:
			vec_analytic_func=np.vectorize(analytic_func)
		def update_img(n):
			time=self.saved[n][0]
			# ymin=0.9*min(self.saved[n][1][:,index])
			sol.set_ydata(self.saved[n][1][:,index])
			#ax.set_ylim(ymin, ymin+yrange)
			# for i in range(len(fields)):
			# 	sol[i].set_ydata(self.saved[n][1][:,1])
			if analytic_func:
				analytic_sol.set_ydata(vec_analytic_func(self.radii))
			#label.set_text(str(time))

		fig,ax=plt.subplots()
		sol,=ax.plot(self.radii, self.saved[0][1][:,index], 'rs')
		ax.set_ylim(0.9*ax.get_ylim()[0])

		# sol=[]
		# fig,ax=plt.subplots(3, figsize=(8, 24))
		# for i in range(len(fields)):
		# 	tmp,=ax[i].plot(self.radii, self.saved[0][1][:,1], 'rs')
		# 	sol.append(tmp)
		if analytic_func:
			analytic_sol,=ax.plot(self.radii, vec_analytic_func(self.radii))
		#label=ax[0].text(0.02, 0.95, '', transform=ax.transAxes)	

		sol_ani=animation.FuncAnimation(fig,update_img,len(self.saved),interval=50)
		sol_ani.save('sol'+str(index)+'.mp4', dpi=200)

	#Save the state of the grid
	def save(self):
		fields=['rho', 'vel', 'temp', 'frho']
		grid_prims=np.zeros((len(fields), self.length))
		for i in range(len(fields)):
			grid_prims[i]=self.get_field(fields[i])[1]
		#Saving the state of the grid within list
		self.saved.append((self.total_time, np.transpose(grid_prims)))

	#Clear all of the info in the saved list
	def clear_saved(self):
		self.saved=[]
			

	# # #Take a single step in time
	# def _step(self):
	# 	#Evaluating the Courant condition
	# 	self._cfl()
	# 	#Getting array of all the densities
	# 	rho0=self.get_field('rho')[1]
	# 	rho1=np.zeros_like(rho0)
	# 	#Updating all of the grid zones
	# 	for i in range(self.start, self.end+1):
	# 		rho1[i]=rho0[i]+self.delta_t*self.drho_dt(i)
	# 		self.grid[i].rho=rho1[i]
	# 	rho2=np.zeros_like(rho1)
	# 	for i in range(self.start, self.end+1):
	# 		rho2[i]=(3./4.)*rho0[i]+(1./4.)*rho1[i]+(1./4.)*self.drho_dt(i)*self.delta_t
	# 		self.grid[i].rho=rho2[i]
	# 	for i in range(self.start, self.end+1):
	# 		self.grid[i].rho=(1./3.)*rho0[i]+(2./3.)*rho2[i]+(2./3.)*self.drho_dt(i)*self.delta_t
	# 		self.grid[i].update()
	# 	#self._update_ghosts()

	#Take a single step in time
	def _step(self, field):
		gamma=[8./15., 5./12., 3./4.]
		zeta=[-17./60.,-5./12.,0.]

		#Evaluating the Courant condition
		self._cfl()
		#Getting array of all values of the field in grid.
		f=self.get_field(field)[1]
		g=f[:]

		#Updating all of the grid zones using predictor corrector scheme as discussd in...
		for j in range(3):
			for i in range(self.start, self.end+1):
				fprime=self.dfield_dt(i, field)
				f[i]=g[i]+gamma[j]*self.delta_t*fprime
				if j!=2:
					g[i]=f[i]+zeta[j]*self.delta_t*fprime
			#Updating the the grid with the results of a single time step
			for i in range(self.start, self.end+1):
				setattr(self.grid[i], field+'_tmp', f[i])

		# if self.periodic:
		# 	end=getattr(self.grid[-1], field)
		# 	setattr(self.grid[0], field, end)
		# else:


	#Extracting the array corresponding to a particular field from the grid as well as an array of radii
	def get_field(self, field):
		field_list=np.zeros(self.length)
		rad_list=np.zeros(self.length)
		for i in range(0, self.length):
			field_list[i]=getattr(self.grid[i], field)
			rad_list[i]=getattr(self.grid[i],'rad')

		return [rad_list, field_list]









		
































