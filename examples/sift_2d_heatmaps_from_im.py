import sys, optparse, glob
import sift2d as im
import matplotlib.pyplot as plt

parser = optparse.OptionParser()
parser.add_option('-p', '--profile', dest='profile', action='store_true', help='Read the 2D-SIFt profile as an input.')
parser.add_option('-r', '--residues', dest='residues', action='store_true', help='Generate per-residue heatmaps.')


(options, args) = parser.parse_args()

exp_args = []
for arg in args:
    if '*' in arg:
        exp_args.extend(glob.glob(arg))
    else:
        exp_args.append(arg)
for arg in exp_args:
    if options.profile:
        s2dfp = im.readProfile(arg)
    else:
        s2dfp = list(im.SIFt2DReader(arg))[0]

    if options.residues:
        for y in  [x for x in s2dfp.get_interacting_chunks()]:
            fig = y.get_heatmap(vmax=s2dfp.get_numpy_array().max())
            fig.savefig('{}_{}.png'.format(s2dfp.receptor_name, y.resnum))
            fig.clf()

    fig = s2dfp.get_heatmap([s.resnum for s in s2dfp.get_interacting_chunks()], plt.cm.Greys)
    fig.savefig('{}.png'.format(s2dfp.receptor_name), dpi=300)