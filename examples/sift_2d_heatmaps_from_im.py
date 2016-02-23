import sys
import sift2d as im
import matplotlib.pyplot as plt

s2dfp = list(im.SIFt2DReader(sys.argv[1]))[0]

for y in  [x for x in s2dfp.get_interacting_chunks()]:
    print(y.resnum)
    print(y.chunk)
    fig = y.get_heatmap(vmax=s2dfp.get_numpy_array().max())
    fig.savefig('{}_{}.png'.format(s2dfp.receptor_name, y.resnum))
    fig.clf()

fig = s2dfp.get_heatmap([s.resnum for s in s2dfp.get_interacting_chunks()], plt.cm.Greys)
fig.savefig('{}.png'.format(s2dfp.receptor_name), dpi=300)