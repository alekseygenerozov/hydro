#!/usr/bin/env python

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3

import numpy as np
from tempfile import NamedTemporaryFile
import argparse

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

def ilabel_line(fig):
	for ax in fig.axes:
		handles,labels=ax.get_legend_handles_labels()
		for i in range(len(handles)):
			mpld3.plugins.connect(fig, mpld3.plugins.LineLabelTooltip(handles[i], label=labels[i]))
  
	return mpld3.display(fig)

def ilabel_point(fig):
	for ax in fig.axes:
		handles,labels=ax.get_legend_handles_labels()
		for i in range(len(handles)):
			#mpld3.plugins.connect(fig, mpld3.plugins.LineLabelTooltip(handles[i], label=labels[i]))
			mpld3.plugins.connect(fig, mpld3.plugins.PointLabelTooltip(handles[i], labels=[labels[i]]))
	 
	return mpld3.display(fig)

