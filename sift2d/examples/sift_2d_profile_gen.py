from __future__ import print_function
import optparse, glob, itertools

import sift2d as im
import matplotlib.pyplot as plt

rows = {
    'H_DONOR' : 'D', 
    'H_ACCEPTOR' : 'A', 
    'HYDROPHOBIC' : 'H', 
    'N_CHARGED' : 'C', 
    'P_CHARGED' : 'C',      
    'AROMATIC' : 'R',
    'POLAR': 'P',
    'ANY' : 'vdW',
    }
interactions = {
    'H_DONOR' : 'Hb', 
    'H_ACCEPTOR' : 'Hb', 
    'HYDROPHOBIC' : 'H',  
    'CHARGED' : 'C',     
    'AROMATIC' : 'R',
    'POLAR': 'P',
    'ANY' : 'vdW',
    }

parser = optparse.OptionParser()
parser.add_option('-o', '--options', dest='output', type=str, default='interaction_profile_2dp.dat', help='Specify the output file')
parser.add_option('-u', '--unique', dest='unique', action='store_true', help='Generate profiles only for unique ligands (multiple occurrences of the same ligand are ignored)')
parser.add_option('-p', '--property', dest='property', type=str, help='Take ligand names from structure property <prop>')
parser.add_option('-g', '--use-generic', dest='generic', action='store_true', help='Use generic numbers from the receptor structure')
parser.add_option('-c', '--cutoff', dest='cutoff', type=float, default=0.3, help='Interactions frequency cutoff.')
parser.add_option('-t', '--table', dest='table', action='store_true', help='Print a table with per ligand interactions.')
parser.add_option('-d', '--feature_table', dest='feat_table', action='store_true', help='Print a table with per ligand interactions. Indicate the most frequently interacting pharmacophore feature.')
parser.add_option('-i', '--interaction_based', dest='interaction', action='store_true', help='Print the most common interactions instead of feature in the table.')
parser.add_option('-w', '--word', dest='wordfriendly', action='store_true', help='Make the table "MS Word friendly" ')


(options, args) = parser.parse_args()

sift_list = []
exp_args = []
for arg in args:
    if '*' in arg:
        exp_args.extend(glob.glob(arg))
    else:
        exp_args.append(arg)
for arg in exp_args:
    print(arg)
    try:
        sift_list.extend(list(im.SIFt2DReader(arg)))
    except:
        continue
    print("{}\t{}".format(list(im.SIFt2DReader(arg))[0].receptor_name, [x.resnum for x in list(im.SIFt2DReader(arg))[0].get_interacting_chunks()]))
generic_numbers = None
if options.generic:
    generic_numbers = sorted(list(set(itertools.chain.from_iterable(x.custom_residues_set for x in sift_list))))
s2dp = im.SIFt2DProfile(sift_list, generic_numbers=generic_numbers)
s2dp.calculate_average()
profile_chunks = s2dp.get_interacting_chunks(options.cutoff)
print("Stats for interacting chunks:\nNumber of residues in the profile:\t{}\nSelection string: {}".format(len(profile_chunks), ','.join([str(x.resnum) for x in profile_chunks])))
s2dp.write(filename=options.output)
if options.table:
    from prettytable import PrettyTable, MSWORD_FRIENDLY
    t = PrettyTable(['Receptor', 'Ligand'] + [x.resnum for x in profile_chunks])
    if options.wordfriendly:
        t.set_style(MSWORD_FRIENDLY)
    for s2 in sift_list:
        line = ['x'  if x.resnum in [y.resnum for y in s2.get_interacting_chunks()] else '' for x in profile_chunks]
        if options.feat_table:
            if options.interaction:
                line = [interactions[s2[x.resnum].get_the_most_frequent_interaction()[1]]  if x.resnum in [y.resnum for y in s2.get_interacting_chunks()] else '' for x in profile_chunks]
            else:
                line = [rows[s2[x.resnum].get_the_most_frequent_feature()[1]]  if x.resnum in [y.resnum for y in s2.get_interacting_chunks()] else '' for x in profile_chunks]
        t.add_row([s2.receptor_name, s2.ligand_name] + line)
    t.add_row(['Avg.', ''] + ['{:.2f}'.format(x.chunk.max()) for x in profile_chunks])
    print(t)
fig = s2dp.get_heatmap([s.resnum for s in s2dp.get_interacting_chunks(options.cutoff)], plt.cm.Greys)
fig.savefig('common_site.png')