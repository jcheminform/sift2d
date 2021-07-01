from __future__ import print_function
import optparse, glob, itertools

import sift2d as im
import matplotlib.pyplot as plt


parser = optparse.OptionParser()
parser.add_option('-o', '--options', dest='output', type=str, default='interaction_profile_2dp.dat', help='Specify the output file')
parser.add_option('-u', '--unique', dest='unique', action='store_true', help='Generate profiles only for unique ligands (multiple occurrences of the same ligand are ignored)')
parser.add_option('-p', '--property', dest='property', type=str, help='Take ligand names from structure property <prop>')
parser.add_option('-c', '--cutoff', dest='cutoff', type=float, default=0.3, help='Interactions frequency cutoff.')


(options, args) = parser.parse_args()

sift_list = []
exp_args = []
for arg in args:
    if '*' in arg:
        exp_args.extend(glob.glob(arg))
    else:
        exp_args.append(arg)
for arg in exp_args:
    sift_list.extend(list(im.SIFt2DReader(arg)))
    print("{}\t{}".format(list(im.SIFt2DReader(arg))[0].receptor_name, [x.resnum for x in list(im.SIFt2DReader(arg))[0].get_interacting_chunks()]))
generic_numbers = None
print(len(sift_list))
s2dp = im.SIFt2DProfile(sift_list)
s2dp.calculate_average()
profile_chunks = s2dp.get_interacting_chunks(options.cutoff)
print("Stats for interacting chunks:\nNumber of residues in the profile:\t{}\nSelection string: {}".format(len(profile_chunks), ','.join([str(x.resnum) for x in profile_chunks])))
s2dp.write(filename=options.output)
fig = s2dp.get_heatmap([s.resnum for s in s2dp.get_interacting_chunks(options.cutoff)], plt.cm.Greys)
fig.savefig('common_abl_site.png')