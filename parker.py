
#!/usr/bin/env python

import argparse
import astropy.constants as const
import numpy as np
from scipy import optimize
from scipy.special import lambertw
import hydro as hydro
import cProfile
import matplotlib.pyplot as plt

import subprocess


G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
kb=const.k_B.cgs.value
mp=const.m_p.cgs.value

# floor=2.E-31

# temp=1.e6
# m=1.
# c_s=np.sqrt(kb*temp/mp)
# rc=G*m*M_sun/(2*c_s**2)
# rmin=6.E11
# rmax=5.E12


##Run a command from the bash shell
def bash_command(cmd):
    process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process.communicate()[0]
    # process.wait()
    # return process


def delta_src(rad, mdot=600., delta=1.E10, r_0=1.E12, r_1=2.E12, tol=1.E10):
    #return mdot*(1-np.tanh((rad-r_0)*(rad-r_1)/(4*delta**2)))
    return mdot*(1./(4.*np.pi*rad**2))*(1./(np.sqrt(2.*np.pi)*delta))*np.exp(-(rad-r_0)**2/(2.*delta**2))
    #if r_0-tol<rad<r_0+tol:
    #    return rho_0
    #else:
    #    return 0
    
# #Power law source. Exponent and constant out from a
# def power_src(rad, a=1.*10**-11, n=-2):
#     return a*(rad)**n
def power_src(rad, mdot=6.7*10.**22, eta=-2, r1=2.4e17, r2=1.2e18):
    a=mdot/(4.*np.pi)/((r2**(eta+3)-r1**(eta+3))/(eta+3))
    return a*rad**eta

#Source term from quataert 2004. Essentially a broken power law
def quataert_src(rad, eta=-2., mdotw=6.7*10.**22, r1=2.4e17, r2=1.2e18):
    a=mdotw/(4.*np.pi)/((r2**(eta+3)-r1**(eta+3))/(eta+3))
    if rad<r2 and rad>r1:
        return a*rad**eta
    else:
        return 0.

#Parker wind solution     
def parker(rad, temp=1.E6, mdot=1.E10, m=1., pert=1., log=False):
    c_s=np.sqrt(kb*temp/mp)
    rc=G*m*M_sun/(2*c_s**2)
    
    f=(rc/rad)**4*np.exp(4*(1-(rc/rad))-1)
    if (rad<=rc):
        vel=np.real(np.sqrt(-lambertw(-f)))
    else:
        vel=np.real(np.sqrt(-lambertw(-f, -1)))
    
    vel*=c_s
    vel*=pert
    if log:
        rho=np.log(mdot/(4.*np.pi*(rad)**2*vel))
    else:
        rho=mdot/(4.*np.pi*(rad)**2*vel)
    
    return np.array([rho, vel, temp])

#Bondi solution
def bondi(rad, temp=1.E6, mdot=1.E10, m=1., pert=1., log=True):
    c_s=np.sqrt(kb*temp/mp)
    rc=G*m*M_sun/(2*c_s**2)
    
    #guess=0.5
    f=(rc/rad)**4*np.exp(4*(1-(rc/rad))-1)
    if (rad>=rc):
        vel=-np.sqrt(-lambertw(-f))
    else:
        vel=-np.sqrt(-lambertw(-f, -1))
    #vel=optimize.fsolve(vel_parker, guess, args=(rc, rad))[0]
    
    vel*=c_s
    vel*=pert
    rho=np.log(-mdot/(4.*np.pi*(rad)**2*vel))
    
    return np.array([rho, vel, temp])

# def vel_parker(rad, temp=temp, mdot=1., m=1., pert=1.):
#     return parker(rad, temp=temp, mdot=mdot, m=m)[1]

#def free_fall(rad, mdot=1.E10, m=1., temp=1.e6):
#    vel=-np.sqrt(2*G*m*M_sun/rad)
#    rho=mdot/(4*np.pi*rad**2*np.abs(vel))
#    return np.array([rho, vel, temp])

def background(rad, rho_0=2.E-31, temp=1.E6, log=True, r0=5.E11, n=0.):
    if log:
        rho_0=np.log(rho_0*(rad/r0)**n)
    return np.array([rho_0, 0., temp])
    
def tcross(rmin, rmax, temp):
    return (rmax-rmin)/c_s(temp)

def c_s(temp):
    return np.sqrt(kb*temp/mp)


    


def main():
    parser=argparse.ArgumentParser(description='Wrapper for running hydro code')
    parser.add_argument('-n','--n', help='number of grid zones', type=int, default=200)
    parser.add_argument('-Re', '--Re', help='Reynolds number',type=float, default=90.)
    parser.add_argument('-rmin', '--rmin', help='inner radius of the grid', type=float,default=6.E11)
    parser.add_argument('-rmax', '--rmax', type=float, default=1.2E12)
    parser.add_argument('-M', '--M', help='Central mass',type=float, default=M_sun)
    parser.add_argument('-q','--q', action='store_true')
    parser.add_argument('-logr', '--logr', help='Turn on log grid', action='store_true')
    parser.add_argument('-t', '--temp', help='temperature (assuming isothermal)', type=float, default=1.E6)
    parser.add_argument('-s', '--sym', help='symbol to be used for plotting', default='r')
    parser.add_argument('-i', '--init', help='initial condition function', default='background', choices=['background', 'parker', 'bondi'])

    args=parser.parse_args()
    M=args.M
    n=args.n
    Re=args.Re
    rmin=args.rmin
    rmax=args.rmax
    temp=args.temp
    logr=args.logr
    q=args.q
    sym=args.sym

    if args.init=='bondi':
        init=bondi
    elif args.init=='parker':
        init=parker
    else:
        init=background

    if args.q:
        q=delta_src
    else:
        q=''
    # c_s=np.sqrt(kb*temp/mp)
    # #Sound-crossing time
    # tcross=(rmax-rmin)/c_s
    # d=dict(log=True)

    # grid=hydro.Grid(rmin, rmax, background, M=M_sun, n=100, safety=0.6, Re=100, params=d, floor=floor,
    #     q=power_src, symbol='r', logr=False)
    # grid.evolve(50*tcross,analytic_func=[None, None, None, None, None, None])

    grid=hydro.Grid(rmin, rmax, init, M=M, n=n, safety=0.6, Re=Re, floor=0., symbol=sym, logr=logr, q=q)
    grid.evolve(2*tcross(rmin, rmax, temp),analytic_func=[None, None, None, None, None, None])


    #bash_command('cp '+'tmp tmp_log')

    # grid=hydro.Grid(rmin, rmax, background, M=M_sun, n=100, safety=0.6, Re=150, params=d, floor=floor, q=delta_src,
    #     symbol='rs', logr=False)
    # grid.evolve(3*tcross,analytic_func=[None, None, None, None, None, None])


    print 'be',grid._bernoulli_check()
    print 4*np.pi*(grid.grid[grid.length-1].frho-grid.grid[0].frho)

# for i in range(2,3):
#     rmax=i*1.E12
#     grid=hydro.Grid(rmin, rmax, background, M=M_sun, n=100, safety=0.6, Re=100, params=d, floor=floor, q=delta_src,
#         symbol='r')
#     grid.evolve(5*tcross,analytic_func=[None, None, None, None])
#     plt.plot(grid.radii, grid.saved[-1,:,0])
#     plt.savefig('test'+str(i)+'.eps')


if __name__ == '__main__':
    main()
