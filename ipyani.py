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

#Create movie of solution
def animate(fname,  index=1, symbol='r', logx=True, logy=False, max_index=-1, ymin=None, ymax=None, times=None):
    ind_list=np.array([index]).flatten()
    sym_list=np.array([symbol]).flatten()
    saved=np.load(fname)['a']
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
                sol[i].set_ydata(saved[n*50,:,ind_list[i]])
            except ValueError:
                sol[i],=ax.plot(radii, np.ones(len(radii)))
        if np.any(times):
            label.set_text(str(times[n*50]))
        else:
            label.set_text(str(n))
    #Exporting animation
    sol_ani=animation.FuncAnimation(fig,update_img,len(saved[:max_index,:,index])/50,interval=50, blit=True)
    return sol_ani
