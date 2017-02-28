"""
A controller script to generate 2D-SIFts.

*REQUIRES* schrodinger libraries!
"""
from __future__ import print_function

import glob
import optparse
import pickle
from schrodinger import structure
from schrodinger.structutils import analyze

import sift2d as im

parser_ = optparse.OptionParser()
parser_.add_option(
    '-o',
    '--output',
    dest='output',
    default=None,
    type=str,
    help='Specify the output file')
parser_.add_option(
    '-u',
    '--unique',
    dest='unique',
    action='store_true',
    help='Generate unique 2D-SIFts (multiple occurences of the same ligand are ignored)')
parser_.add_option(
    '--prop',
    dest='property',
    type=str,
    default=None,
    help='Take ligand names from structure property <prop>')
parser_.add_option(
    '-c',
    '--cutoff',
    dest='cutoff',
    type=float,
    default=3.5,
    help='Interactions frequency cutoff.')
parser_.add_option(
    '-g',
    '--use-generic',
    dest='generic',
    action='store_true',
    help='Use generic numbers from the receptor structure')
parser_.add_option(
    '-p',
    '--pdb',
    dest='pdb',
    action='store_true',
    help='Use a single pdb file as input')
parser_.add_option(
    '-m',
    '--maestro',
    dest='maestro',
    action='store_true',
    help="Use a list of files in maestro format (receptor+multiple ligands)."
)
parser_.add_option(
    '-i',
    '--pickle_output',
    dest='pickle',
    action='store_true',
    help='Save output as pickled SIFt2D objects.')

(options, args) = parser_.parse_args()

print(options)
if options.maestro:
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

        s2dg = im.SIFt2DGenerator(
            receptor,
            struct_it,
            use_generic_numbers=options.generic,
            cutoff=options.cutoff,
            unique=options.unique,
            property_=options.property)
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
elif options.pdb:
    print("Processing file {}".format(args[0]))
    pdb_struct = structure.StructureReader(args[0]).next()
    ligands = analyze.find_ligands(pdb_struct)
    print("Found ligands ", [x.pdbres for x in ligands])
    ligands = [x.st for x in ligands]
    if ligands == [] or ligands is None:
        print("This structure does not contain any ligands!")
        exit()
    receptor = pdb_struct.extract(analyze.evaluate_asl(pdb_struct, "protein"))
    s2dg = im.SIFt2DGenerator(
        receptor,
        ligands,
        use_generic_numbers=options.generic,
        cutoff=options.cutoff,
        unique=options.unique,
        property_=options.property)
    if options.output:
        if options.pickle:
            out_fh = open(options.output, 'wb')
            pickle.dump(list(s2dg), out_fh)
            out_fh.close()
        else:
            s2dg.write_all(options.output, 'a')
else:
    receptor = structure.StructureReader(args[0]).next()
    print(receptor.title)
    struct_it = structure.StructureReader(args[1])
    s2dg = im.SIFt2DGenerator(
        receptor,
        struct_it,
        use_generic_numbers=options.generic,
        cutoff=options.cutoff,
        unique=options.unique,
        property_=options.property)
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
