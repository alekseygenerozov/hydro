#!/usr/bin/env python 
import matplotlib.pylab as plt 
import pickle
import numpy as np
import astropy.constants as const

#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value
h=const.h.cgs.value
pc=const.pc.cgs.value
c=const.c.cgs.value


my_dir='/home/aleksey/Second_Year_Project/hydro/ngc4551_2/vw_crit'

def lambda_c(temp):
    if temp>2.E7:
        return 2.3E-24*(temp/1.E6)**0.5
    else:
        return 1.1E-22*(temp/1.E6)**-0.7
    
def cooling(my_grid, gamma=5./3.):
    gamma=5./3.
    
    fig,ax=plt.subplots()
    plt.loglog()
    plt.plot(my_grid.radii, my_grid.q*(0.5*my_grid.get_field('vel')[1]**2+0.5*my_grid.vw**2))
    plt.plot(my_grid.radii, my_grid.q*(gamma/(gamma-1))*(my_grid.get_field('pres')[1]/my_grid.get_field('rho')[1]))
    lambdas=np.array(map(lambda_c, my_grid.get_field('temp')[1]))
    cooling=lambdas*(my_grid.get_field('rho')[1]/mp)**2
    plt.plot(my_grid.radii, cooling)

    return fig

#Comparing the heating and cooling rates.
grid200=pickle.load( open( my_dir+'/vw_200/new_phi/grid.p', 'rb') )
grid1000=pickle.load( open( my_dir+'/vw_1000.0/new_phi/grid.p', 'rb' ) )

cooling(grid200).savefig(my_dir+'/vw_200/new_phi/cooling_200.png')
cooling(grid1000).savefig(my_dir+'/vw_1000.0/new_phi/cooling_1000.png')