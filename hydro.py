import numpy as np
import copy
import warnings
import pickle
from scipy import integrate

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import e

import subprocess
import astropy.constants as const
import progressbar as progress
import ipyani
import sys

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value



##Run a command from the bash shell
def bash_command(cmd):
    process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process.communicate()[0]

#Compute logarithmic slope given values and radii
def get_slope(r1, r2, f1, f2):
    return np.log(f2/f1)/np.log(r2/r1)

#Extend inner grid boundary to be supersonic
def extend_to_ss(dat):
    #Getting the slopes of all profile
    slope=np.empty(5)
    field=np.copy(dat[0])
    for i in range(len(field)):
        slope[i]=get_slope(dat[0,0], dat[3,0], dat[0,i], dat[3,i])
    #Getting target radius
    r_end=dat[0,0]*np.exp((1./slope[2])*np.log(-2./dat[0,2]))

    #Computing the grid spacing
    delta_log=np.log(dat[1,0]/dat[0,0])
    r=dat[0,0]
    logr=np.log(dat[0,0])
    dat_extend=np.copy(dat)
    while r>r_end:
        logr=logr-delta_log
        r=np.exp(logr)
        for i in range(len(field)):
            field[i]=field[i]*(r/dat_extend[0,0])**slope[i]
        #Stacking to the end of the grid
        dat_extend=np.vstack([field, dat_extend])
    return dat_extend
        
        


##Preparing array with initialization file (idea is to go back from save file)
def prepare_start(end_state, rescale=1):
	# end_state=dat[-70:]
	end_state[:,2]=end_state[:,2]*end_state[:,-1]
	end_state[:,1]=np.log(end_state[:,1])
	start=end_state[:,:4]
	#rescaling radial grid. This will be useful for going from one mass black hole to another. 
	start[:,0]=start[:,0]*rescale

	return start


class Zone:
	"""Class to store zones of (static) grid, to be used to solve
	Euler equations numerically. Contains radius (rad), primitive 
	variables and pressures. Also contains mass enclosed inside 
	cell.

	"""
	def __init__(self, vw=np.array(0), phi=np.array(0), q=np.array(0), rad=1.E16, prims=(0,0,0), isot=True, gamma=5./3., mu=1.):
		#Radius of grid zone 
		self.rad=rad
		self.mu=mu

		#Primitive variables
		self.log_rho=prims[0]
		self.vel=prims[1]
		self.temp=prims[2]
		self.q=q
		self.isot=isot
		self.gamma=gamma
		self.vw=vw
		self.phi=phi
		#Entropy
		self.entropy()
		#Updating non-primitive variables within zone
		self.update_aux()

	#Equation of state. Note this could be default in the future there could be functionality to override this.
	def eos(self):
		self.pres=self.rho*kb*self.temp/(self.mu*mp)
		if not self.isot:
			self.temperature()
			self.cs=np.sqrt(self.gamma*kb*self.temp/(self.mu*mp))
		else:                                                                                                     
			self.cs=np.sqrt(kb*self.temp/(self.mu*mp))

	#Temperature->Entropy
	def entropy(self):
		self.s=(kb/(self.mu*mp))*np.log(1./np.exp(self.log_rho)*(self.temp)**(3./2.))

	#Entropy->Temperature
	def temperature(self):
		self.temp=(np.exp(self.log_rho)*np.exp(self.mu*mp*self.s/kb))**(2./3.)

	#Calculate hearting in cell
	def get_sp_heating(self):
		return (0.5*self.vel**2+0.5*self.vw[0]**2-(self.gamma)/(self.gamma-1)*(self.pres/self.rho))

	#Method which will be used to update non-primitive vars. 
	def update_aux(self):
		self.rho=np.exp(self.log_rho)
		self.eos()
		self.r2vel=self.rad**2*self.vel
		self.frho=self.rad**2*self.vel*self.rho
		self.be=self.bernoulli()
		self.fen=self.rho*self.rad**2*self.vel*self.be
		self.sp_heating=self.get_sp_heating()
		self.src_rho=self.q[0]*self.rad**2
		self.src_en=self.rad**2.*self.q[0]*(self.vw**2/2.+self.phi)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.src_v=-(self.q[0]*self.vel/self.rho)+(self.q[0]*self.sp_heating/(self.rho*self.vel))
		if self.isot:
			self.src_s=0.
		else:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				self.src_s=self.q[0]*self.sp_heating/(self.rho*self.vel*self.temp)


	#Finding the maximum transport speed across the zone
	def alpha_max(self):
		return max([np.abs(self.vel+self.cs), np.abs(self.vel-self.cs)])

	def bernoulli(self):
		u=self.pres/(self.rho*(self.gamma-1.))
		return 0.5*self.vel**2+(self.pres/self.rho)+u+self.phi[0]

class Grid:
	"""Class stores (static) grid for solving Euler equations numerically"""

	def __init__(self, galaxy, init=None, init_array=None,  n=100, num_ghosts=3, safety=0.6, Re=100., Re_s=100., params=dict(), params_delta=dict(),
		floor=1.e-30, symbol='rs', logr=True, gamma=5./3., isot=False, tol=1.E-3,  movies=True, mu=1., vw=0., qpc=True, veff=False, outdir='./', scale_heating=1.,
		s_interval=100, eps=1.,  tinterval=-1, visc_scheme='default', bdry='default', bdry_fixed=False):
		assert n>2*num_ghosts

		#Initializing the radial grid
		if init!=None:
			assert len(init)==3
			r1,r2,f_initial=init[0],init[1],init[2]
			assert r2>r1
			assert hasattr(f_initial, '__call__')

			#Setting up the grid (either logarithmic or linear in r)
			self.logr=logr
			if logr:
				self.radii=np.logspace(np.log(r1), np.log(r2), n, base=e)
			else:
				self.radii=np.linspace(r1, r2, n)
			prims=np.zeros([100,3])
			for i in range(len(self.radii)):
				prims[i]=f_initial(self.radii[i], **params)
			self.length=n
		elif type(init_array)==np.ndarray:
			self.radii=init_array[:,0]
			prims=init_array[:,1:]
			self.length=len(self.radii)
		else:
			raise Exception("Not enough initialization info entered!")

		#Attributes to store length of the list as well as start and end indices (useful for ghost zones)
		self.start=0
		self.end=self.length-1
		self.tinterval=tinterval

		#self.const_visc=False
		self.isot=isot
		#self.fields=['log_rho', 'vel']
		if self.isot:
			self.fields=['log_rho', 'vel']
		else:
			self.fields=['log_rho', 'vel', 's']
		self.movies=movies
		self.outdir=outdir
		self.out_fields=['rad', 'rho', 'vel', 'temp', 'frho', 'be', 's', 'cs']
		self.cons_fields=['frho', 'be', 's', 'fen']
		self.src_fields=['src_rho', 'src_v', 'src_s', 'src_en']
		self.s_interval=s_interval

		self.mu=mu
		self.params=params
		self.params_delta=params_delta	


		self.gamma=gamma
		#Grid will be stored as list of zones
		self.grid=[]
		#delta=float(r2-r1)/n
		self.safety=safety

		self.visc_scheme=visc_scheme
		self.Re=Re
		self.Re_s=Re_s

		self.floor=floor

		#Setting up the grid (either logarithmic or linear in r)
		self.logr=logr
		#Setting up source terms and potential throughout the grid.  
		if qpc:
			self.q=np.array(map(galaxy.q, self.radii/pc))/pc**3
		else:
			self.q=np.array(map(galaxy.q, self.radii))

		self.vw=np.empty(self.length)
		self.vw[:]=vw
		#self.vw.fill(c*((vw**2/c**2))**0.5)
		#Place mass onto the grid.
		self.veff=veff
		self.scale_heating=scale_heating

		self.eps=eps
		self.place_mass(galaxy)

		#Will store values of time derivatives at each time step
		self.time_derivs=np.zeros(self.length, dtype={'names':['log_rho', 'vel', 's'], 'formats':['float64', 'float64', 'float64']})

		#Initializing the grid using the initial value function f_initial
		for i in range(len(self.radii)):
			self.grid.append(Zone(vw=self.vw[i:i+1], q=self.q[i:i+1],  phi=self.phi[i:i+1], rad=self.radii[i], prims=prims[i], isot=self.isot, gamma=gamma, mu=mu))

		self._add_ghosts(num_ghosts=num_ghosts)

		self.bdry=bdry
		self.bdry_fixed=bdry_fixed

		#Computing differences between all of the grid elements 
		delta=np.diff(self.radii)
		#print delta
		self.delta=np.insert(delta, 0, delta[0])
		#print self.delta

		delta_log=np.diff(np.log(self.radii))
		self.delta_log=np.insert(delta_log, 0, delta_log[0])
		self.first_deriv_coeffs=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.
		self.second_deriv_coeffs=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.)


		self.delta_t=0
		self.time_cur=0
		self.total_time=0
		self.time_target=0

		self.saved=np.empty([0, self.length, len(self.out_fields)])
		self.fdiff=np.empty([0, self.length-1, 2*len(self.cons_fields)+1])

		self.max_change=[]
		self.tol=tol
		self.time_stamps=[]
		self.symbol=symbol

		bash_command('mkdir -p '+self.outdir)
		f=open(self.outdir+'/params', 'w')
		f.write(str(vars(self)))
		log=open(self.outdir+'/log', 'w')
		log.close()
		save=open(self.outdir+'/save', 'w')
		save.close()
		times=open(self.outdir+'/times', 'w')
		times.close()

	#Calculate hearting in cell
	def get_sp_heating(self, i):
		return (0.5*self.grid[i].vel**2+0.5*self.vw[i]**2-(self.gamma)/(self.gamma-1)*(self.grid[i].pres/self.grid[i].rho))
	
	#Update array of conseerved quantities	
	def _cons_update(self):
		#differences in fluxes and source terms
		fdiff=np.empty([2*len(self.cons_fields)+1, self.length-1])
		fdiff[0]=self.radii[1:]
		for i in range(3):
			flux=self.get_field(self.cons_fields[i])[1]
			#get differences in fluxes for all of the cells.
			fdiff[i+1]=np.diff(flux)
			#get the source terms for all of the cells.
			fdiff[i+4]=(self.get_field(self.src_fields[i])[1]*self.delta)[1:]
			
		self.fdiff=np.append(self.fdiff,[np.transpose(fdiff)],0)

	#Check how well conservation holds on grid as a whole.
	def _cons_check(self):
		check=file(self.outdir+'/check', 'w')
		for i in range(len(self.cons_fields)):
			flux=4.*np.pi*self.get_field(self.cons_fields[i])[1]
			fdiff=flux[self.end]-flux[self.start]
			src=4.*np.pi*self.get_field(self.src_fields[i])[1]*self.delta
			integral=np.sum(src[self.start:self.end+1])
			with warnings.catch_warnings():
				pdiff=(fdiff-integral)*100./integral

			check.write(self.cons_fields[i]+'\n')
			pre=['flux1=','flux2=','diff=','src=','pdiff=']
			vals=[flux[self.start],flux[self.end],fdiff,integral,pdiff]
			for j in range(len(vals)):
				check.write(pre[j])
				s='{0:4.3e}'.format(vals[j])
				check.write(s+'\n')
			check.write('____________________________________\n\n')

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

	#Applying boundary conditions
	def _update_ghosts(self):
		if self.bdry=='bp':
			self._s_adjust()
			self._dens_extrapolate()
			self._mdot_adjust()
		else:
			self._extrapolate('rho')
			self._extrapolate('vel')
			self._extrapolate('s')
		for i in range(0, self.start):
			self.grid[i].update_aux()
		for i in range(self.end+1, self.length):
			self.grid[i].update_aux()

	#Constant entropy across the ghost zones
	def _s_adjust(self):
		s_start=self.grid[self.start].s
		for i in range(0, self.start):
			self.grid[i].s=s_start
		s_end=self.grid[self.end].s
		for i in range(self.end+1, self.length):
			self.grid[i].s=s_end

	#Extrapolate densities to the ghost zones
	def _extrapolate(self, field):
		r1=self.grid[self.start].rad
		r2=self.grid[self.start+3].rad
		field1=getattr(self.grid[self.start], field)
		field2=getattr(self.grid[self.start+3], field)
		slope=np.log(field2/field1)/np.log(r2/r1)

		for i in range(0, self.start):
			val=field1*np.exp(slope*np.log(self.grid[i].rad/r1))
			setattr(self.grid[i], field, val)
			if field=='rho':
				self.grid[i].log_rho=np.log(val)
			
		#Updating the end ghost zones
		r1=self.grid[self.end].rad
		r2=self.grid[self.end-3].rad
		field1=getattr(self.grid[self.end], field)
		field2=getattr(self.grid[self.end-3], field)
		slope=np.log(field2/field1)/np.log(r2/r1)
		
		#Updating the end ghost zones, extrapolating using a power law density
		for i in range(self.end+1, self.length):
			val=field1*np.exp(slope*np.log(self.grid[i].rad/r1))
			setattr(self.grid[i], field, val)
			if field=='rho':
				self.grid[i].log_rho=np.log(val)

	#Extrapolate densities to the ghost zones
	def _dens_extrapolate(self):
		r_start=self.grid[self.start].rad
		r_start2=self.grid[self.start+3].rad
		log_rho_start=self.grid[self.start].log_rho
		log_rho_start2=self.grid[self.start+3].log_rho

		#If the inner bdry is fixed...(appropriate for Parker wind)
		if self.bdry_fixed:
			for i in range(1, self.start):
				self.grid[i].log_rho=self._interp_zones(self.grid[i].rad, 0, self.start, 'log_rho')
				self.grid[i].rho=np.exp(self.grid[i].log_rho)
		#Updating the starting ghost zones, extrapolating using rho prop r^-3/2
		else:
			for i in range(0, self.start):
				slope=-3./2.
				#slope=(log_rho_start2-log_rho_start)/np.log(r_start2/r_start)
				log_rho=slope*np.log(self.grid[i].rad/r_start)+log_rho_start
				self.grid[i].log_rho=log_rho
				self.grid[i].rho=np.exp(log_rho)
		#Updating the end ghost zones
		r_end=self.grid[self.end].rad
		log_rho_end=self.grid[self.end].log_rho
		r_end2=self.grid[self.end-3].rad
		log_rho_end2=self.grid[self.end-3].log_rho
		#Updating the end ghost zones, extrapolating using a power law density
		for i in range(self.end+1, self.length):
			slope=-2.
			#slope=(log_rho_end-log_rho_end2)/np.log(r_end/r_end2)
			log_rho=slope*np.log(self.grid[i].rad/r_end)+log_rho_end
			self.grid[i].log_rho=log_rho
			self.grid[i].rho=np.exp(log_rho)

	#Enforce constant mdot across the boundaries (bondary condition for velocity)
	def _mdot_adjust(self):
		#Start zones 
		frho=self.grid[self.start].frho
		for i in range(0, self.start):
			vel=frho/self.grid[i].rho/self.grid[i].rad**2
			self.grid[i].vel=vel
			self.grid[i].update_aux()
		#End zones
		frho=self.grid[self.end].frho
		for i in range(self.end+1, self.length):
			vel=frho/self.grid[i].rho/self.grid[i].rad**2
			self.grid[i].vel=vel
			self.grid[i].update_aux()

	#Get stencil for a particular zone
	def _get_stencil(self, i, left=3, right=3):
		assert i-left>=0
		assert i+right<self.length

		return self.grid[i-left:i+right+1]

	#Evaluate terms of the form div(kappa*df/dr)-->Diffusion like terms (useful for something like the conductivity)
	def get_diffusion(self, i, coeff, field):
		dkappa_dr=self.get_spatial_deriv(i, coeff)
		dfield_dr=self.get_spatial_deriv(i, field)
		d2field_dr2=self.get_spatial_deriv(i, field, second=True)

		if hasattr(coeff, '__call__'):
			kappa=coeff(self.grid[i])
		else:	
			kappa=getattr(self.grid[i], coeff)

		return kappa*d2field_dr2+dkappa_dr*dfield_dr+(2./self.radii[i])*(kappa*dfield_dr)


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
		for j in range(num_zones):
			if hasattr(field, '__call__'):
				field_list[j]=field(stencil[j])
			else:	
				field_list[j]=getattr(stencil[j], field)

		#Coefficients we will use.
		#coeffs=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.

		if second:
			if self.logr:
				#coeffs2=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.*self.delta_log[i])
				return (1./self.grid[i].rad**2/self.delta_log[i]**2)*(np.sum(field_list*self.second_deriv_coeffs)
					-np.sum(field_list*self.first_deriv_coeffs))
			else:
				#coeffs2=np.array([2., -27., 270., -490., 270., -27., 2.])/(180.*self.delta[i])
				return np.sum(field_list*self.second_deriv_coeffs)/self.delta[i]**2
				
		else:
			if self.logr:
				return np.sum(field_list*self.first_deriv_coeffs)/(self.delta_log[i]*self.radii[i])
			else:
				return np.sum(field_list*self.first_deriv_coeffs)/self.delta[i]

	#Calculate laplacian in spherical coords. 
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
		elif field=='log_rho':
			return self.dlog_rho_dt(i)
		elif field=='s':
			return self.ds_dt(i)
		else:
			return 0.

	#Partial derivative of density with respect to time
	def dlog_rho_dt(self, i):
		rad=self.grid[i].rad
		rho=self.grid[i].rho
		vel=self.grid[i].vel

		#return -cs*drho_dr+art_visc*drho_dr_second
		return -vel*self.get_spatial_deriv(i, 'log_rho')-(1/rad**2)*self.get_spatial_deriv(i, 'r2vel')+self.q[i]/rho

	#Partial derivative of density with respect to time
	def drho_dt(self, i):
		rad=self.grid[i].rad

		#return -cs*drho_dr+art_visc*drho_dr_second
		return -(1./rad)**2*self.get_spatial_deriv(i, 'frho')+self.q[i]

	#Evaluating the partial derivative of velocity with respect to time
	def dvel_dt(self, i):
		rad=self.grid[i].rad
		vel=self.grid[i].vel
		rho=self.grid[i].rho
		temp=self.grid[i].temp

		#If the density zero of goes negative return zero to avoid numerical issues
		if rho<=self.floor:
		 	return 0

		# dpres_dr=self.get_spatial_deriv(i, 'pres')
		dlog_rho_dr=self.get_spatial_deriv(i, 'log_rho')
		dtemp_dr=self.get_spatial_deriv(i, 'temp')
		dv_dr=self.get_spatial_deriv(i, 'vel')
		d2v_dr2=self.get_spatial_deriv(i, 'vel', second=True)
		drho_dr=dlog_rho_dr*(rho)

		lap_vel=self.get_laplacian(i, 'vel')
		art_visc=min(self.grid[i].cs,  np.abs(self.grid[i].vel))*(self.radii[self.end]-self.radii[self.start])*lap_vel/self.Re
		if self.visc_scheme=='const_visc':
			pass
		elif self.visc_scheme=='cap_visc':
			art_visc=art_visc*min(1., (self.delta[i]/np.mean(self.delta)))
		else:
			art_visc=art_visc*(self.delta[i]/np.mean(self.delta))

		# if self.visc2:
		# 	art_visc=0.1*self.delta[i]**2*(drho_dr*(dv_dr)**3+3.*rho*(dv_dr)**2*(d2v_dr2))


		return -vel*dv_dr-dlog_rho_dr*kb*temp/(self.mu*mp)-(kb/(self.mu*mp))*dtemp_dr-self.grad_phi[i]+art_visc-(self.q[i]*vel/rho)

	#Evaluating the partial derivative of temperature with respect to time.
	# def dtemp_dt(self, i):
	# 	return 0.

	#Evaluating the partial derivative of entropy with respect to time
	def ds_dt(self, i):
		rho=self.grid[i].rho
		temp=self.grid[i].temp
		vel=self.grid[i].vel
		rad=self.grid[i].rad
		cs=self.grid[i].cs
		ds_dr=self.get_spatial_deriv(i, 's')
		lap_s=self.get_laplacian(i, 's')
		art_visc=min(self.grid[i].cs,  np.abs(self.grid[i].vel))*(self.radii[self.end]-self.radii[self.start])*lap_s/self.Re_s
		if self.visc_scheme=='const_visc':
			pass
		elif self.visc_scheme=='cap_visc':
			art_visc=art_visc*min(1., (self.delta[i]/np.mean(self.delta)))
		else:
			art_visc=art_visc*(self.delta[i]/np.mean(self.delta))


		#return self.q(rad, **self.params_delta)*(0.5*self.vw**2+0.5*vel**2-self.gamma*cs**2/(self.gamma-1))/(rho*temp)-vel*ds_dr#+art_visc*lap_s
		return self.q[i]*self.grid[i].sp_heating/(rho*temp)-vel*ds_dr+art_visc

	#Switch off isothermal equation of state for all zones within our grid.
	def isot_off(self):
		self.set_param('isot', False)
		self.set_param('fields', ['log_rho', 'vel', 's'])
		for zone in self.grid:
			zone.isot=False
			zone.entropy()

	#Set all mass dependent quantities
	def place_mass(self, galaxy):
		self.M_bh=galaxy.params['M']
		self.M_enc_arr=np.array(map(galaxy.M_enc, self.radii/pc))
		print self.M_enc_arr

		self.M_tot=self.M_enc_arr+self.M_bh
		#Potential and potential gradient (Note that we have to be careful of the units here.)
		self.phi_s=np.array(map(galaxy.phi_s, self.radii/pc))/pc
		self.phi_bh=np.array(map(galaxy.phi_bh, self.radii/pc))/pc
		self.phi=self.eps*self.phi_s+self.phi_bh
		#self.phi=self.eps*(np.array(map(galaxy.phi, self.radii/pc))/pc+(G*self.M_bh/self.radii))-G*self.M_bh/self.radii
		self.grad_phi=G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii**2

		#Gravitational radius
		self.rg=G*(self.M_bh)/c**2.
		if self.veff:
			self.vw=self.scale_heating*c*((self.rg/self.radii)*(self.M_tot/self.M_bh))**0.5


	# #Resetting the wind velocity to a new value; useful to turn off heating by winds. 
	# def reset_vw(self, vw):
	# 	self.vw=c*((self.rg/self.radii)*(self.M_tot/self.M_bh)+(vw**2/c**2))**0.5
	# 	for i in range(len(self.grid)):
	# 		self.grid[i].vw=self.vw[i]

	#High-level controller for solution. Several possible stop conditions (max_steps is reached, time is reached, convergence is reached)
	def solve(self, time, max_steps=np.inf):
		self.time_cur=0
		self.time_target=time

		#For purposes of easy backup
		if len(self.saved)==0:
			self.save_pt=0
		else:
			self.save_pt=len(self.saved)-1

		self._evolve(max_steps=max_steps)

		self.write_sol()

	def set_param(self, param, value):
		old=getattr(self,param)
		if param=='M_bh' or param=='M_enc_arr':
			print 'Operation curretly not supported\n'
			return
		elif param=='eps':
			self.eps=value
			self.phi=self.eps*self.phi_s+self.phi_bh
			self.grad_phi=G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii**2
		elif param=='scale_heating' and self.veff:
			self.vw[:]=self.scale_heating*c*((self.rg/self.radii)*(self.M_tot/self.M_bh))**0.5
			for i in range(self.length):
				self.grid[i].vw=self.vw[i:i+1]
		elif param=='vw':
			self.vw[:]=value
			for i in range(self.length):
				self.grid[i].vw=self.vw[i:i+1]
		else:
			setattr(self,param,value)

		log=open(self.outdir+'/log', 'a')
		new=getattr(self, param)
		log.write(param+' old:'+str(old)+' new:'+str(value)+' time:'+str(self.total_time)+'\n')
		log.close()

	#Gradually perturb a given parameter (param) to go to the desired value (target). 
	def solve_adjust(self, time, param, target, n=10, max_steps=np.inf):
		if len(self.saved==0):
			self.save_pt=0
		else:
			self.save_pt=len(self.saved)-1
		self.time_cur=0
		param_cur=getattr(self, param)
	
		interval=time/float(n)
		self.time_target=interval
		delta_param=(target-param_cur)/float(n)
		while not np.allclose(param_cur, target):
			self._evolve(max_steps=max_steps)
			param_cur+=delta_param
			self.time_target+=interval
			self.set_param(param,param_cur)
		self.write_sol()

			
	#Method to write solution info to file
	def write_sol(self):
		#For all of the field we would like to output, output movie.
		# if self.movies:
		# 	for i in range(0, len(self.out_fields)):
		# 		self.animate(index=i)
		#Writing solution
		# mdot_check=self._mdot_check()
		# be_check=self._bernoulli_check()
		# s_check=self._s_check()
		# check=file(self.outdir+'/check', 'w')
		# check.write('mdot: frho1={0:8.7e} frho2={1:8.7e} flux={2:8.7e} mdot={3:8.7e} percent diff={4:8.7e}\n\n'.format(mdot_check[0], mdot_check[1], mdot_check[2], mdot_check[3], mdot_check[4]))
		# check.write('be: be1={0:8.7e} be2={1:8.7e} flux={2:8.7e} \int src={3:8.7e} percent diff={4:8.7e}\n\n'.format(be_check[0], be_check[1], be_check[2], be_check[3], be_check[4]))
		# check.write('s: s1={0:8.7e} s2={1:8.7e} flux={2:8.7e} \int src={3:8.7e} percent diff={4:8.7e}'.format(s_check[0], s_check[1], s_check[2], s_check[3], s_check[4]))
		# check.close()
		self._cons_check()

		np.savez(self.outdir+'/save', a=self.saved, b=self.time_stamps)
		np.savez(self.outdir+'/cons', a=self.fdiff)

		# bash_command('mkdir -p '+self.outdir)
		# bash_command('cp save.npz cons.npz check params log '+self.outdir)
		#plt.clf()

	def backup(self):
		bash_command('mkdir -p '+self.outdir)
		pickle.dump(self, open( self.outdir+'/grid.p', 'wb' ) )

	#Lower level evolution method
	def _evolve(self, max_steps=np.inf):
		#Initialize the number of steps and the progress
		num_steps=0
		pbar=progress.ProgressBar(maxval=self.time_target, fd=sys.stdout).start()
		ninterval=0

		#While we have not yet reached the target time
		while self.time_cur<self.time_target:
			if (self.tinterval>0 and (self.time_cur/self.tinterval)>=ninterval) or (self.tinterval<=0 and num_steps%self.s_interval==0):
				pbar.update(self.time_cur)
				self._cons_update()
				self.save()
				ninterval=ninterval+1

				# if len(self.saved)>2:
				# 	max_change=np.max(np.abs((self.saved[-1]-self.saved[-2])/self.saved[-2]))
				# 	self.max_change.append(max_change)
				# 	# if max_change<self.tol:
				# 	# 	break
				# #np.savetxt(fout, self.saved[-1])
				# #print self.total_time/self.time_target

			#Take step and increment current time
			self._step()
			#Increment the time and steps variables.
			self.time_cur+=self.delta_t
			self.total_time+=self.delta_t
			num_steps+=1
			#If we have exceeded the max number of allowed steps then break
			if num_steps>max_steps:
				print "exceeded max number of allowed steps"
				break
		pbar.finish()
		print

	#Reverts grid to earlier state. Previous solution
	def revert(self, index=None):
		if index==None:
			index=self.save_pt
		for i in range(len(self.grid)):
			self.grid[i].log_rho=np.log(self.saved[index,i,1])
			self.grid[i].vel=self.saved[index,i,2]*self.saved[index,i,-1]
			self.grid[i].temp=self.saved[index,i,3]
			if not self.isot:
				self.grid[i].entropy()
			self.grid[i].update_aux()
		self.saved=self.saved[:index]
		self.time_stamps=self.time_stamps[:index]
		self.fdiff=self.fdiff[:index]
		#Overwriting previous saved files
		np.savetxt(self.outdir+'/save', self.saved.reshape((-1, len(self.out_fields))))
		np.savetxt(self.outdir+'/cons', self.fdiff.reshape((-1, 7)))
		np.savetxt(self.outdir+'/times',self.time_stamps)

	#Create movie of solution
	def animate(self,  analytic_func=None, index=1):
		if analytic_func:
			vec_analytic_func=np.vectorize(analytic_funcs)
		def update_img(n):
			time=self.time_stamps[n]
			sol.set_ydata(self.saved[n*50,:,index])
			label.set_text(str(time))

		#Setting up for plotting
		ymin=np.min(self.saved[:,:,index])
		ymax=np.max(self.saved[:,:,index])
		fig,ax=plt.subplots()
		label=ax.text(0.02, 0.95, '', transform=ax.transAxes)	
                ax.set_xscale('log')
		if index==0:
			ax.set_yscale('log')
		ax.set_ylim(ymin-0.1*np.abs(ymin), ymax+0.1*np.abs(ymax))

		#Plot solution/initial condition/(maybe) analytic solution
		sol,=ax.plot(self.radii, self.saved[0,:,index], self.symbol)
		ax.plot(self.radii, self.saved[0,:,index], 'b')
		if analytic_func:
			analytic_sol,=ax.plot(self.radii, vec_analytic_func(self.radii))
		
		#Exporting animation
		sol_ani=animation.FuncAnimation(fig,update_img,len(self.saved)/50,interval=50, blit=True)
		sol_ani.save('sol_'+self.out_fields[index]+'.mp4', dpi=100)
		plt.clf()

	#Save the state of the grid
	def save(self):
		#fields=['rho', 'vel', 'temp', 'frho']
		grid_prims=np.zeros((len(self.out_fields), self.length))
		for i in range(len(self.out_fields)):
			grid_prims[i]=self.get_field(self.out_fields[i])[1]
			if self.out_fields[i]=='vel':
				grid_prims[i]=grid_prims[i]/self.get_field('cs')[1]

		#Saving the state of the grid within list
		#self.saved.append((self.total_time, np.transpose(grid_prims)))
		self.saved=np.append(self.saved,[np.transpose(grid_prims)],0)
		self.time_stamps.append(self.total_time)
		#Dump to file
		save_handle=file(self.outdir+'/save','a')
		np.savetxt(save_handle, self.saved[-1])
		cons_handle=file(self.outdir+'/cons','a')
		np.savetxt(cons_handle, self.fdiff[-1])
		t_handle=file(self.outdir+'/times', 'a')
		np.savetxt(t_handle, [self.time_stamps[-1]])

	#Clear all of the info in the saved list
	def clear_saved(self):
		self.saved=np.empty([0, self.length, len(self.out_fields)])
		self.fdiff=np.empty([0, self.length-1, 7])
		self.time_stamps=[]
		self.save_pt=0
		self.total_time=0

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









		
































