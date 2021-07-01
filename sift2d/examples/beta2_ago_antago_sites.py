from __future__ import print_function
import sys

import sift2d as im
import matplotlib.pyplot as plt


cutoff = 0.3
#Set up
p1 = im.readProfile(sys.argv[1])
p2 = im.readProfile(sys.argv[2])
p1.apply_cutoff(cutoff)
p2.apply_cutoff(cutoff)
#Common set of generic residue positions
common_positions = sorted(list(set([x.resnum for x in p1.get_interacting_chunks(cutoff)] + [y.resnum for y in p2.get_interacting_chunks(cutoff)])))
print("Common:{}\n{}".format(len(common_positions),common_positions))
#Calculating differential profile and saving figure
diff = p2 - p1
vmax = max([x.chunk.max() for x in diff.get_interacting_chunks(cutoff)])
vmin = min([x.chunk.min() for x in diff.get_interacting_chunks(cutoff)])
border = max(-vmin, vmax)
#diff
hm = diff.get_heatmap(common_positions, plt.cm.seismic, vmin=-border, vmax=border)
hm.savefig('difference.jpg', dpi=300)
#Profile #1
p1_fig = p1.get_heatmap(common_positions, plt.cm.Blues, vmax=border)
p1_fig.savefig('profile1.jpg', dpi=300)
print("Profile 1 positions: {}".format([x.resnum for x in p1.get_interacting_chunks(cutoff)]))
#Profile #2
p2_fig = p2.get_heatmap(common_positions, plt.cm.Reds,  vmax=border)
p2_fig.savefig('profile2.jpg', dpi=300)
print("Profile 2 positions: {}".format([x.resnum for x in p2.get_interacting_chunks(cutoff)]))