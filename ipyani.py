import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)

# <codecell>

from IPython.display import HTML

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

# <codecell>

#Create movie from save.npz or cons.npz file generated by hydro code.
def animate(saved,  index=1, symbol='r', logx=True, logy=False, max_index=-1, ymin=None, ymax=None, times=None, interval=50):
    ind_list=np.array([index]).flatten()
    sym_list=np.array([symbol]).flatten()
    # saved=np.load(fname)['a']
    if not ymin:
        ymin=np.min(saved[:max_index,:,index])
    if not ymax:
        ymax=np.max(saved[:max_index,:,index])
    radii=saved[0,:,0]
    
    fig,ax=plt.subplots()
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_ylim(ymin-0.1*np.abs(ymin), ymax+0.1*np.abs(ymax))
    label=ax.text(0.02, 0.95, '', transform=ax.transAxes)	

    #Plot solution/initial condition/(maybe) analytic solution
    sol={}

    for i in range(len(ind_list)):
        try:
            sol[i],=ax.plot(radii, saved[0,:,ind_list[i]], symbol[i%len(sym_list)])
        except ValueError:
            sol[i],=ax.plot(radii, np.ones(len(radii)))
        #ax.plot(radii, saved[0,:,i], 'b')
    
    def update_img(n):
        for i in range(len(ind_list)):
            try:
                sol[i].set_ydata(saved[n*interval,:,ind_list[i]])
            except ValueError:
                sol[i],=ax.plot(radii, np.ones(len(radii)))
        if np.any(times):
            label.set_text(str(times[n*interval]))
        else:
            label.set_text(str(n))
    #Exporting animation
    sol_ani=animation.FuncAnimation(fig,update_img,len(saved[:max_index,:,index])/interval,interval=50, blit=True)
    return sol_ani


def movie_save(loc, interval=10, ymin=[None, None, None, None, -1, None], ymax=[None, None, None, None, 2, None], logy=[True, True, True, True, False, True], times=None):
    #times=grid.time_stamps/parker.tcross(rmin*pc, rmax*pc, temp)
    # times=None
    files=['/mass_cons.mp4', '/be_cons.mp4', '/s_cons.mp4','/rho.mp4', '/vel.mp4', '/temp.mp4']
    for i in range(3):
        ani=animate(loc+'/cons.npz', index=[i+1, i+4], ymin=ymin[i], ymax=ymax[i], times=times, symbol=['b', 'r'], interval=interval, logy=logy[i])
        ani.save(loc+files[i])
    for i in range(3, 6):
        ani=animate(loc+'/save.npz', index=i-2, ymin=ymin[i], ymax=ymax[i], times=times, symbol='b', interval=interval, logy=logy[i])
        ani.save(loc+files[i])


