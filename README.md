#Hydro
Modelling stellar winds in galactic nuclei


##Requirements
*The following packages are required to run the code. I have only used it with the version numbers in parentheses...*

Scipy (1.8.1)

Numpy (0.14)

Astropy (0.4)

Dill (0.2.1)


##Model 
This code is designed to model the injection of stellar winds in a galactic nucleus. The equations solve are the 1d spherically symmetric hydro equation with source terms accouting for the injection of winds from stars and potentially also including cooling and conductivity...

The relevant equations are summarized here...

Much more detail about our model may be found in (paper citation here)

##Comment on different galaxy models

The code includes three different types of model galaxies  differing in the way their stellar density profile are implemented

1. *Galaxy.* 

	The stellar density is assumed to be zero. Therefore 	the the stars do not contribute to to the 	gravitational potential. However by default there is a contribution from the stellar velocity dispersion: this is given by
 

	Thus the only gravity is from the central black hole, whose mass is stored in the dictionary params 	(under params['M'])


2. *NukerGalaxy.*

	 This assumes that stellar light profile follows a 	Nuker Law (Faber et al. 1997). User shold pass the 	name of a galaxy.  Along with the name the user may 	also pass a dictionary of nuker galaxies (using the 	keyword gdata). Each entry in this dictionary should 	itself be a dictionary of Nuker parameters.
 
	 If no dictionary is passed to gdata one is constructed from the file wm (this is table 1 Wang & Merritt 2004).

	The mass enclosed from the stars contributes to the 	gravitational potential. 

3. *Pow Galaxy.* 

	This galaxy corresponds to a broken power law 	density profile. 

##Example running of code...

The following run was designed to reproduce the results from Quataert 2004, and also illustrates a basic example of running the code.

```python
import galaxy

#Bounds for grid
rmin=3.8E16
rmax=3.E18

'''Init is a dictionary which contain info for initializtion of the grid: rmin is the inner boundary,
rmax is the outer boundary. length (not shown) controls the number of grid points. func_params contain parameters to be passes to the initialization function ''' 
init={'rmin':rmin, 'rmax':rmax, 'func_params':dict(rho_0=1.E-23, temp=1.E7, n=0.)}

#Initialize galaxy
gal=galaxy.Galaxy(init=init)
#Set boundary condition
gal.set_param('bdry', 'bp')
#Set molecular weight 
gal.set_param('mu', 0.5)
gal.set_param('tinterval', None)
#Output directory
gal.set_param('outdir', 'quataert')
#Turn off heating from velocity dispersion
gal.set_param('sigma_heating', False)
#Turn on isothermal evolution
gal.set_param('isot', True)
#Run for sound crossing time
gal.solve(gal.tcross)
#Turn off isothermal evolution
gal.set_param('isot', False)
#Run for another sound crossing time.
gal.solve(gal.tcross)
```

##Summary of galaxy properties, possible values, and defaults
Note that parameters may be adjusted using the set_param method (which accepts the name of a parameter and the value you would like to change it to). 
1. *params* (stores structural parameters of galaxy--the assumed structure is somewhat different for each type of 		     galaxy) 
	*params['M']:black hole mass
	*params['gamma']:slope of stellar density profile (PowGalaxy/NukerGalaxy) or just slope of mass source 

Numerical parameters:
2. *Re* artificial viscosity in the velocity equation
3. *Re_s* artificial viscosity in the entropy equation
4. *bdry* type of condition to use 

5. 

4. *vw_extra* extra heating rate
5. *phi_cond* 
