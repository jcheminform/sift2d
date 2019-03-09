from __future__ import print_function

from schrodinger import structure, structureutil
from schrodinger.structutils import analyze, assignbondorders
from schrodinger.infra import mm

import sys, os, glob

empty_files = []
multi_ligs = []

exp_args = []
for arg in sys.argv[1:]:
    if '*' in arg:
        exp_args.extend(glob.glob(arg))
    else:
        exp_args.append(arg)
for infile in exp_args:
    root,ext = os.path.splitext(infile)
    print("Processing file {}".format(os.path.basename(infile)))
    pdb_struct = next(structure.StructureReader(infile))


    ligands = analyze.find_ligands(pdb_struct)
    print([x.pdbres for x in ligands])
    if ligands == [] or ligands == None:
        empty_files.append(os.path.basename(infile))
        continue
    if len(ligands) > 1:
        multi_ligs.append(os.path.basename(infile))
    receptor = pdb_struct.extract(analyze.evaluate_asl(pdb_struct, "protein"))
    receptor.title = os.path.splitext(os.path.basename(infile))[0]
    out_file = "{}_pv.mae".format(root)
    print(out_file)
    receptor.write(out_file)
    for lig in ligands:
        tmp_st = lig.st #*can't change the title on the fly for some reason
        tmp_st.title = lig.pdbres
        tmp_st.append(out_file)

print("Empty files:\n{}\n\n\n".format('\n'.join(empty_files)))
print("Multiple ligands:\n{}".format('\n'.join(multi_ligs)))
