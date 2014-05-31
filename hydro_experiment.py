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
import inspect

import h5py

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

##Preparing array with initialization file (idea is to go back from save file)
def prepare_start(dat):
	end_state=dat[-70:]
	end_state[:,2]=end_state[:,2]*end_state[:,-1]
	end_state[:,1]=np.log(end_state[:,1])
	start=end_state[:,:4]

	return start

##Dummy function which returns 0
def dummy(r):
	return 0.



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

	def __init__(self, M_bh=10.**6*M_sun, M_enc=dummy, q=dummy, init=None, init_array=None,  n=100, num_ghosts=3, safety=0.6, Re=100., Re_s=100., params=dict(), params_delta=dict(),
		floor=1.e-30, symbol='rs', logr=True, bdry_fixed=False, gamma=5./3., isot=False, tol=1.E-3,  movies=True, mu=1., vw=0., qpc=True, veff=False, outdir='./', scale_heating=1.,
		s_interval=100, eps=0.,  tinterval=-1, visc_scheme='default'):
		print  locals()
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
		self.out_fields=['rad', 'rho', 'vel', 'temp', 'cs']
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
			self.q=np.array(map(q, self.radii/pc))/pc**3
		else:
			self.q=np.array(map(q, self.radii))

		self.vw=np.empty(self.length)
		self.vw[:]=vw
		#self.vw.fill(c*((vw**2/c**2))**0.5)
		#Place mass onto the grid.
		self.veff=veff
		self.scale_heating=scale_heating

		self.eps=eps
		self.place_mass(M_bh, M_enc)

		#Will store values of time derivatives at each time step
		self.time_derivs=np.zeros(n, dtype={'names':['log_rho', 'vel', 's'], 'formats':['float64', 'float64', 'float64']})

		#Initializing the grid using the initial value function f_initial
		for i in range(len(self.radii)):
			self.grid.append(Zone(vw=self.vw[i:i+1], q=self.q[i:i+1],  phi=self.phi[i:i+1], rad=self.radii[i], prims=prims[i], isot=self.isot, gamma=gamma, mu=mu))

		self._add_ghosts(num_ghosts=num_ghosts)
		self.bdry_fixed=bdry_fixed

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

		# self.saved=np.empty([0, self.length, len(self.out_fields)])
		# self.fdiff=np.empty([0, self.length-1, 2*len(self.cons_fields)+1])

		self.max_change=[]
		self.tol=tol
		self.time_stamps=[]
		self.symbol=symbol
		#Preparing output
		self._output_prep()

	#Prepare to write files to output
	def _output_prep(self):	
		#Files which log initial parameters passed to code and changes to the parameters done during the run.
		bash_command('mkdir -p '+self.outdir)
		f=open(self.outdir+'/params', 'w')
		f.write(str(vars(self)))
		log=open(self.outdir+'/log', 'w')
		log.close()
		#Preparing hdf5 file which will store the simulation data.
		self.hdf5_save=h5py.File(self.outdir+'/save.hdf5', 'w')
		sol=self.hdf5_save.create_group('sol')
		cons=self.hdf5_save.create_group('cons')
		src=self.hdf5_save.create_group('src')

		for field in self.out_fields:
			sol.create_dataset(field,(0,self.length),  maxshape=(None, self.length))
			self.save_key(field)
		for i in range(len(self.cons_fields)):
			cons.create_dataset(self.cons_fields[i],(0,self.length-1), maxshape=(None, self.length-1))
			src.create_dataset(self.src_fields[i], (0,self.length-1), maxshape=(None, self.length-1))


	#Calculate hearting in cell
	def get_sp_heating(self, i):
		return (0.5*self.grid[i].vel**2+0.5*self.vw[i]**2-(self.gamma)/(self.gamma-1)*(self.grid[i].pres/self.grid[i].rho))
	
	#Update array of conseerved quantities	
	def _cons_update(self):
		#differences in fluxes and source terms
		for i in range(len(self.cons_fields)):
			flux=self.get_field(self.cons_fields[i])[1]
			#get differences in fluxes for all of the cells.
			dset=self.hdf5_save['cons'][self.cons_fields[i]]
			dset.resize([len(dset)+1, self.length-1])
			self.hdf5_save['cons'][self.cons_fields[i]][-1]=np.diff(flux)[:]
			#get source integrated across each grid cell.
			dset=self.hdf5_save['src'][self.src_fields[i]]
			dset.resize([len(dset)+1, self.length-1])
			self.hdf5_save['src'][self.src_fields[i]][-1]=(self.get_field(self.src_fields[i])[1]*self.delta)[1:]
			
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
		self._extrapolate('rho')
		self._extrapolate('vel')
		self._extrapolate('s')
		for i in range(0, self.start):
			self.grid[i].update_aux()
		for i in range(self.end+1, self.length):
			self.grid[i].update_aux()
		# self._s_adjust()
		# self._dens_extrapolate()
		# self._mdot_adjust()

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
		coeffs=np.array([-1., 9., -45., 0., 45., -9., 1.])/60.

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
	def place_mass(self, M_bh, M_enc=None):
		self.M_bh=M_bh
		#self.M_enc=copy.deepcopy(M_enc)
		if M_enc!=None:
			self.M_enc_arr=np.array(map(M_enc, self.radii/pc))
		else:
			self.M_enc_arr=np.zeros(self.length)
		self.M_tot=self.M_enc_arr+M_bh
		#Potential and potential gradient
		self.phi=-G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii
		self.grad_phi=G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii**2

		#Gravitational radius
		self.rg=G*(M_bh)/c**2.
		if self.veff:
			self.vw=self.scale_heating*c*((self.rg/self.radii)*(self.M_tot/self.M_bh))**0.5

	#High-level controller for solution. Several possible stop conditions (max_steps is reached, time is reached, convergence is reached)
	def solve(self, time, max_steps=np.inf):
		self.time_cur=0
		self.time_target=time

		self._evolve(max_steps=max_steps)
		self._cons_check()


	def set_param(self, param, value):
		old=getattr(self,param)
		if param=='M_bh' or param=='M_enc_arr':
			print 'Operation curretly not supported\n'
			return
			# self.M_bh=value
			# self.M_tot
			# self.phi=-G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii
			# self.grad_phi=G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii**2
		elif param=='eps':
			self.eps=value
			self.phi=-G*(self.M_bh+self.eps*self.M_enc_arr)/self.radii
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
		self._cons_check()

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
				self.save()
				ninterval=ninterval+1

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

	def save_key(self, key):
		dat=self.get_field(key)[1]
		dset=self.hdf5_save['sol'][key]
		dset.resize([len(dset)+1, self.length])
		dset[-1]=dat[:]

	#Save solution to hdf5 file
	def save(self):
		for key in self.hdf5_save['sol'].keys():
			self.save_key(key)
		self._cons_update()

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









		
































