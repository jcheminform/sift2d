from __future__ import print_function

import optparse, glob, pickle
from schrodinger import structure

import sift2d as im

parser = optparse.OptionParser()
parser.add_option('-o', '--output', dest='output', default=None, type=str, help='Specify the output file')
parser.add_option('-u', '--unique', dest='unique', action='store_true', help='Generate 2d SIFt only for unique ligands (multiple occurences of the same ligand are ignored)')
parser.add_option('-p', '--property', dest='property', type=str, help='Take ligand names from structure property <prop>')
parser.add_option('-g', '--use-generic', dest='generic', action='store_true', help='Use generic numbers from the receptor structure')
parser.add_option('-c', '--pickle_output', dest='pickle', action='store_true', help='Save output as pickled SIFt2D objects.')

(options, args) = parser.parse_args()
exp_args = []
for arg in args:
    if '*' in arg:
        exp_args.extend(glob.glob(arg))
    else:
        exp_args.append(arg)
for arg in exp_args:
    receptor = structure.StructureReader(arg).next()
    print(receptor.title)
    struct_it = structure.StructureReader(arg, index=2)
    
    s2dg = im.SIFt2DGenerator(receptor, struct_it, use_generic_numbers=options.generic, unique=options.unique, property=options.property)
    if options.output:
        if options.pickle:
            out_fh = open(options.output, 'wb')
            pickle.dump(list(s2dg), out_fh)
            out_fh.close()
        else:
            s2dg.write_all(options.output, 'a')
    else:
        if options.pickle:
            out_fh = open('{!s}_pickled_2dfp.dat'.format(receptor.title), 'wb')
            pickle.dump(list(s2dg), out_fh)
            out_fh.close()
        s2dg.write_all()
