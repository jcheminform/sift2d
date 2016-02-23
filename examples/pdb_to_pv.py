from __future__ import print_function

from schrodinger import structure, structureutil
from schrodinger.structutils import analyze, assignbondorders
from schrodinger.infra import mm

import sys, os, glob

exp_args = []
for arg in sys.argv[1:]:
    if '*' in arg:
        exp_args.extend(glob.glob(arg))
    else:
        exp_args.append(arg)
for infile in exp_args:
    root,ext = os.path.splitext(infile)
    print("Processing file {}".format(os.path.basename(infile)))
    pdb_struct = structure.StructureReader(infile).next()
    mm.mmlewis_initialize(mm.error_handler)
    mm.mmlewis_pre_add_zobs(pdb_struct)
    assignbondorders.assign_st(pdb_struct, problem_only=True)
    structureutil.add_hydrogens(pdb_struct)
    mm.mmlewis_add_zobs(pdb_struct)

    ligands = analyze.find_ligands(pdb_struct)
    print([x.pdbres for x in ligands])
    if ligands == [] or ligands == None:
        print("This structure does not contain any ligands!")
        continue
    receptor = pdb_struct.extract(analyze.evaluate_asl(pdb_struct, "protein"))
    receptor.title = os.path.splitext(os.path.basename(infile))[0]
    print(receptor.title)
    out_file = "{}_pv.mae".format(root)
    receptor.write(out_file)
    for lig in ligands:
        tmp_st = lig.st #can't change the title on the fly for some reason
        tmp_st.title = lig.pdbres
        tmp_st.append(out_file)