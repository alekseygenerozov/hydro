import sys
import hydro_experiment as hydro
import nuker
import parker
import astropy.constants as const
import numpy as np
import ipyani
import pickle
import matplotlib.pylab as plt

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import re


#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value

galaxies=nuker.nuker_params()

# g1_2=nuker.Galaxy('NGC4551', galaxies, cgs=False, eta=1.) 
def M_enc_simp(r):
	return 4.76E39*(r/1.04)**(2-g1_2.params['gamma'])

#Find the stagnation radius for a particular solution. Take array of radii and velocities
def find_stag(r, v, guess=1.E18):
	interp=interp1d(r, v)
	return fsolve(interp, guess)

def check(outdir, tol=40.):
	check=False
	refloat=r'[+-]?\d+\.?\d*[eE]?[+-]?\d*'
	try:
		f=open(outdir+'/check', 'r')
	except:
		return False
	checkf=f.read()
	
	pdiffs=re.findall(re.compile('pdiff='+refloat), checkf)
	try:
		cons1=re.findall(refloat,pdiffs[0])[0]
		cons2=re.findall(refloat,pdiffs[-1])[0]
		if float(cons1)<tol and float(cons2)<tol:
			check=True
	except:
		pass
	
	return check

#Check solution as code is running
def sol_check(loc, index=2, size=70):
	saved=np.genfromtxt(loc+'/save')
	fig,ax=plt.subplots()
	ax.set_xscale('log')
	if index!=2:
		ax.set_yscale('log')
	plt.plot(saved[0:size,0], saved[0:size,index])
	plt.plot(saved[-size:,0], saved[-size:,index])
	plt.show()
	
#Check solution as code is running
def cons_check(loc, index=2, logy=False, ylim=None,size=70):
	saved=np.genfromtxt(loc+'/cons')
	fig,ax=plt.subplots()
	ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
	if ylim:
		ax.set_ylim(ylim)
	
	plt.plot(saved[-(size-1):,0], saved[-(size-1):,index])
	plt.plot(saved[-(size-1):,0], saved[-(size-1):,index+3])

	return fig
	#plt.show()    

#Analytic expression for Bernoulli parameter.
def be(r, M_bh=10.**7.11*M_sun, M_enc0=7.95E6*M_sun, rho0=1.E-21, r_s=5.E17, vw=1.E8, beta=1.8, sigma=False, shell=False):
	x=r/r_s
	from_s=0.
	if shell:
		from_s=4.*np.pi*G*r_s**2*rho0*(3.-beta)*(x**(4.-2.*beta)-1.)/((1.-beta)*(4.-2.*beta)*(x**(3.-beta)-1.))
	if not sigma:
		return (vw**2/2.)-(G*M_bh/r_s)*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/r_s)*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)+from_s
	else:
		return (vw**2/2.)-(G*M_bh/(2.*r_s))*(3.-beta)/(2.-beta)*(x**(2.-beta)-1)/(x**(3.-beta)-1)-(G*M_enc0/(2.*r_s))*((x**(5.-2.*beta)-1)/(x**(3.-beta)-1))*(3-beta)/(5.-2.*beta)+from_s

def be_check(loc, gal, index=-1, vw=1.E8, beta=1.8, menc=True, sigma=False, shell=False):
	#Load saved data
	saved2=np.load(loc+'/save.npz')['a']
	#Calculate the stagnation radius for the numerical solution.
	r_s=find_stag(saved2[index,:,0], saved2[index,:,2], guess=5.E17)
	if menc:
		M_enc0=gal.M_enc(r_s/pc)
	else:
		M_enc0=0.
	rho0=gal.rho(r_s/pc)/pc**3

	#Calculate the analytic Bernoulli parameter.
	bes=np.empty_like(saved2[index,:,0])
	for i in range(len(saved2[index,:,0])):
		bes[i]=be(saved2[index,i,0], vw=vw, r_s=r_s, M_bh=gal.params['M'], M_enc0=M_enc0, rho0=rho0, beta=beta, sigma=sigma, shell=shell)
	print bes
	fig,ax=plt.subplots()
	plt.loglog()
	ax.plot(saved2[index,:,0], saved2[index,:,-3])
	ax.plot(saved2[index,:,0], bes)
	fig.savefig(loc+'/be_cons_'+gal.name+'.png')
	fig.clf()

	fig,ax=plt.subplots()
	plt.loglog()
	ax.plot(saved2[index,:,0], abs((saved2[index,:,-3]-bes)/bes))
	fig.savefig(loc+'/be_cons2_'+gal.name+'.png')



