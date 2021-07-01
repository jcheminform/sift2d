from schrodinger.structutils import structalign
from schrodinger import structure
import os,glob, optparse

parser = optparse.OptionParser()
parser.add_option('-p', '--path', dest='outpath', default='fit', type=str, help='Specify the output directory')
parser.add_option('-r', '--ref', dest='reference', type=str, help='Reference structure')

(options,args) = parser.parse_args()
ref_st = structure.StructureReader(options.reference).next()
alignment = structalign.StructAlign()

exp_args = []
for arg in args:
    if '*' in arg:
        exp_args.extend(glob.glob(arg))
    else:
        exp_args.append(arg)

for fname in exp_args:
    if fname == options.reference:
        continue
    mob_st = structure.StructureReader(fname).next()
    alignment.alignStructure(ref_st, mob_st)
    mob_st.write(os.sep.join([options.outpath, fname]))