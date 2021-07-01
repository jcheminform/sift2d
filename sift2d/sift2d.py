"""
A module to generate and manipulate 2D-SIFt interaction matrices
and interaction profiles (averaged 2D-SIFts).
"""
from __future__ import print_function

from functools import reduce
import itertools
import math
from string import Template
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from schrodinger.structutils import analyze
    from schrodinger.structutils import interactions
    from schrodinger.structutils import measure
except ImportError:
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures, AddHs
    from rdkit.Chem.rdmolfiles import SDMolSupplier
    from Bio.PDB import PDBParser

#Iteraction type to row/column number
COLUMNS = {
    'ANY' : 0,
    'BACKBONE' : 1,
    'SIDECHAIN' : 2,
    'POLAR' : 3,
    'HYDROPHOBIC' : 4,
    'H_ACCEPTOR' : 5,
    'H_DONOR' : 6,
    'AROMATIC' : 7,
    'CHARGED' : 8
    }
ROWS = {
    'H_DONOR' : 0,
    'H_ACCEPTOR' : 1,
    'HYDROPHOBIC' : 2,
    'N_CHARGED' : 3,
    'P_CHARGED' : 4,
    'AROMATIC' : 5,
    'ANY' : 6,
    }

POLAR_RESIDUES = ["ARG", "ASP", "GLU", "HIS", "ASN", "GLN", "LYS", "SER", "THR", "TYR", "PRO", "TRP", "CYS"]
HB_DONOR = ["ARG", "HIS", "LYS", "ASP", "GLU", "THR", "SER", "TYR", "PRO", "TRP", "ASN", "GLN", "CYS"]
HB_ACCEPTOR = ["ASP", "GLU", "ASN", "GLN"]
HYDROPHOBIC_RESIDUES = [
    "PHE",
    "LEU",
    "ILE",
    "TYR",
    "TRP",
    "VAL",
    "MET",
    "PRO",
    "CYS",
    "ALA",
]
AROMATIC_RESIDUES = ["PHE", "TYR", "TRP"]
POS_CHARGE_RESIDUES = ["ARG", "LYS", "HIS"]
NEG_CHARGE_RESIDUES = ["ASP", "GLU"]

#Residue sets by properties
RESIDUE_SETS = {
    "HYDROPHOBIC_RESIDUES": ["PHE", "LEU", "ILE", "TYR", "TRP",
                             "VAL", "MET", "PRO", "CYS", "ALA",
                             "CYX"],
    "P_CHARGED_RESIDUES": ["ARG", "LYS", "HIS", "HID"],
    "N_CHARGED_RESIDUES": ["ASP", "GLU"],
    "HYDROPHOBIC_RESIDUES": ["PHE", "LEU", "ILE","TYR", "TRP",
                             "VAL", "MET", "PRO", "CYS", "ALA"],
    "AROMATIC_RESIDUES": ["PHE", "TYR", "TRP"],

    }
BACKBONE_ATOMS = ['C', 'N', 'O', 'CA']
#Order of returning the most frequent interactions
INTERACTION_HIERARCHY = ['CHARGED', 'H_ACCEPTOR', 'H_DONOR', 'POLAR', 'AROMATIC', 'HYDROPHOBIC']

# Rules for distance-based interactions
INTERACTION_RULES = {
    'P_CHARGED': [['N_CHARGED_RESIDUES'], ['CHARGED']],
    'N_CHARGED': [['P_CHARGED_RESIDUES'], ['CHARGED']],
    'HYDROPHOBIC': [['HYDROPHOBIC_RESIDUES'], ['HYDROPHOBIC']],
    }
INTERACTION_DICT = {
    'residue_features': 0,
    'byte_to_switch': 1
    }

#SMARTS patterns for ligand features (except AROMATIC)
PH_PATTERNS = {
    'H_DONOR':
    ['[#1][O;X2]', '[#1]S[#6]', '[#1][C;X2]#[C;X2]',
     '[#1][NX3]C(=[NX2])[#6]', '[#1][#7]'],
    'H_ACCEPTOR':
    ['[N;X1]#[#6]', '[N;X1]#CC', '[N;X2](=C~[C,c])C',
     '[N;X2](O)=N[a]', '[N;X2](=N-O)[a]', '[n;X2]1ccccc1',
     '[n;X2]([a])([a])', '[N;X2](=C~[C,c])(~[*])', '[N;X3](C)(C)[N;X3]C',
     '[N;X2](=C)(~[*])', '[N;X2](~[C,c])=[N;X2]', '[n;X2]1c[nH]cc1',
     'O=[S;X4](=O)([!#8])([!#8])', '[O;X2]C', '[O;X2]N', '[O;X1]=[C,c]',
     'o', '[O;X2](C)C', '[O;X2]c1ncccc1', '[O;X2]~[a]', 'O=PO([!#1])',
     '[O;X2]', '[S;X2](C)C', '[S;X2](=C)N', 'O=C[O-]'],
    'P_CHARGED':
    ['[NX3][#6](=[NX2,NX3+])[#6]', '[NX2,NX3+]=[#6]([NH;X3])([NH;X3])',
     '[NX2,NX3+]=[#6]([NX3])([NX3])', 'n1c([NH2])ccnc1([NH2])',
     '[NX2,NX3+]=C([NX3])c1ccccc1', '[NH2;X3,NH3]([#6;X4])',
     '[NH;X3,NH2]([#6;X4])([#6;X4])', '[NX3,NH]([#6;X4])([#6;X4])([#6;X4])',
     'N1CCCCC1', '[+]'],
    'HYDROPHOBIC':
    ['[a]F', '[a]Cl', '[a]Br', '[a]I', '[a]C(F)(F)(F)', '[a][CH2]C(F)(F)(F)',
     '[a]O[CH3]', '[a]S[CH3]', '[a]OC(F)(F)(F)', 'C(F)(F)(F)', 'F', 'Cl', 'Br',
     'I', 'C[S;X2]C', '[S;X2]CC', '[S;X2]C', '[AR0]~[AR0]'],
    'N_CHARGED':
    ['O=C[O-]', 'O=C[OH]', '[S;X4](=O)(=O)([OH])', '[S;X4](=O)(=O)([O-])',
     '[S;X3](=O)([OH])', '[S;X3](=O)([O-])', '[P;X4](=O)([OH])([OH])',
     '[P;X4](=O)([OH])([O-])', '[P;X4](=O)([O-])', '[P;X4](=O)([OH])',
     'n1nc[nH]n1', 'n1ncnn1', '[#1]N([S;X4](=O)(=O))(C(F)(F)(F))', '[-]'],
    }

COULOMB_CUTOFF = 5.0
HBOND_CUTOFF = 2.8
POLAR_CUTOFF = 3.5
VDW_CUTOFF = 4.0
PI_CUTOFF = 6.0

#Output templates
OUTPUT_FORMAT = {
    'txt' : Template("$receptor:$ligand:$start:$fp"),
    }

def get_distance(pos1, pos2):

    return np.sqrt(np.dot(pos1 - pos2, pos1 - pos2))

def get_shortest_distance(res, atom_coords):

    return min(
        [
            np.sqrt(np.dot(x.get_vector() - atom_coords, x.get_vector() - atom_coords))
            for x in res.get_atoms()
        ]
    )

def get_shortest_distance_atoms(atom_set, atom_coords):

    return min(
        [
            np.sqrt(np.dot(x.get_vector() - atom_coords, x.get_vector() - atom_coords))
            for x in atom_set
        ]
    )

#==============================================================================
class SIFt2DChunk:
    """
    Class storing information about interactions for a single residue.
    """

    def __init__(self, resnum=None, data=None):
        """Class storing interaction matrix for a single residue.

        Keyword Arguments:
            resnum {int} -- Residue number (default: {None})
            data {numpy.array} -- A 2D array of shape corresponding to defined interactions.\
             (default: {None})
        """

        self.resnum = resnum
        #*Schrodinger's version of numpy returns shape in form of (xL, yL)
        #*  - can't just compare shape to a tuple (ROWS, COLUMNS)
        if data is None or data.shape != np.zeros((len(ROWS), len(COLUMNS))).shape:
            self.chunk = np.zeros((len(ROWS), len(COLUMNS)), dtype=np.int)
        else:
            self.chunk = data


    def increment_bit(self, row, col):
        """Increase value at given position by 1.

        Arguments:
            row {int} -- Row number
            col {int} -- Column number
        """

        self.chunk[ROWS[row]][COLUMNS[col]] += 1


    def __repr__(self):

        return str(self.chunk)


    def __eq__(self, other):
        """
        Allows comparison of the chunks based on the interaction matrix
        """
        return (self.chunk == other.chunk).all()


    def __ne__(self, other):
        """
        Allows comparison of the chunks based on the interaction matrix
        """
        return not (self.chunk == other.chunk).all()


    def __sub__(self, other):
        """
        Returns the difference between two chunks.
        """
        return SIFt2DChunk(self.resnum, self.chunk - other.chunk)


    def get_row(self, row_num):
        """Returns interactions of given pharmacophore feature.

        Arguments:
            row_num {int} -- A row number from the interaction matrix

        Returns:
            numpy.array -- An interaction matrix slice with interactions for a \
            single pharmacophore feature type.
        """

        return self.chunk[row_num, :]


    def get_column(self, col_num):
        """Returns interactions of given type.

        Arguments:
            col_num {int} -- A column number from the interaction matrix

        Returns:
            numpy.array -- An interaction matrix slice with interactions of a \
            given type.
        """
        return self.chunk[:, col_num]


    def get_the_most_frequent_interaction(self):
        """
        Returns the most frequent interaction.
        """
        imax = (0.0, '')
        for column in INTERACTION_HIERARCHY:
            tmp_max = max(self.chunk[:, COLUMNS[column]])
            if tmp_max > imax[0]:
                imax = (tmp_max, column)
        if imax[1] == 'CHARGED' and self.chunk[ROWS['H_DONOR'], COLUMNS['H_ACCEPTOR']] != 0:
            imax = (imax[0], 'H_ACCEPTOR')
        if imax[1] == 'CHARGED' and self.chunk[ROWS['H_ACCEPTOR'], COLUMNS['H_DONOR']] != 0:
            imax = (imax[0], 'H_DONOR')
        if imax == (0.0, '') and max(self.chunk[:, COLUMNS['ANY']]) != 0.0:
            return (max(self.chunk[:, COLUMNS['ANY']]), 'ANY')
        return imax


    def get_the_most_frequent_feature(self):
        """
        Returns the most frequently interacting pharmacophore feature.
        """
        imax = (0.0, '')
        for row in sorted(ROWS.keys(), reverse=True):
            if row == "ANY" and imax[0] != 0.0:
                break
            tmp_max = max(self.chunk[ROWS[row], :])
            if tmp_max > imax[0]:
                imax = (tmp_max, row)
        return imax



    def get_heatmap(self, vmax=None):
        """
        Returns a standardized heatmap figure.
        """
        heatmap = plt.figure()
        ax = heatmap.gca(xlabel="Residue interactions", ylabel="Ligand features")
        ax.set_yticks(np.arange(len(ROWS))+0.5, minor=False)
        ax.set_yticks(np.arange(len(ROWS)), minor=True)
        ax.set_yticklabels(['D', 'A', 'H', 'N', 'P', 'R', 'vdW'], size='large')
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(COLUMNS))+0.5, minor=False)
        ax.set_xticks(np.arange(len(COLUMNS)), minor=True)
        ax.set_xticklabels(['Any', 'BB', 'SC', 'Polar', 'H', 'A', 'D', 'R', 'Charged'],
                           size='large')
        plt.pcolor(self.chunk, vmax=vmax, cmap=plt.cm.Greys, figure=heatmap)
        ax.grid(which='minor')

        return heatmap



#==============================================================================
class SIFt2D:
    """
    Class storing interaction matrix for the whole receptor.
    """
    generic_chunk = SIFt2DChunk()


    def __init__(self, ligand_name, receptor_name, start, end, custom_residues_set=None):
        """
        Constructor takes a starting and ending residue number.
        """
        self.ligand_name = ligand_name
        self.receptor_name = receptor_name
        self.start = start
        self.end = end
        self._chunks = []
        self._mapping = None
        #For truncated binding sites and sets of generic numbers
        self.custom_residues_set = custom_residues_set
        self._init_chunks()


    def _init_chunks(self):
        """
        Initialize empty chunks for every residue in range.
        """
        if self.custom_residues_set is None:
            for x in range(self.start, self.end+1):
                self._chunks.append(SIFt2DChunk(x))
        else:
            if self.custom_residues_set:
                self._mapping = {}
            for idx, item in enumerate(self.custom_residues_set):
                self._mapping[item] = idx
                self._chunks.append(SIFt2DChunk(item))

    #TODO: Add support for custom numbers for __len__, __iter__ and __getitem__
    def __len__(self):
        """
        Return the number of per residue chunks.
        """
        return len(self._chunks)


    def __iter__(self):
        """
        Iterator over all per residue chunks.
        """
        for chunk in self._chunks:
            yield chunk

    def __getitem__(self, resnum):
        """
        Get chunk(s) related to a given residue or residue range.
        """
        if self._mapping:
            try:
                return self._chunks[self._mapping[resnum]]
            except KeyError:
                raise KeyError("No chunk found for residue {:n}.".format(resnum))
        else:
            try:
                return self._chunks[resnum - self.start]
            except KeyError:
                if resnum > self.end or resnum < self.start:
                    raise KeyError("Residue number {:n} out of range {:n}:{:n}"
                                   .format(resnum, self.start, self.end))
                else:
                    raise KeyError("No chunk found for residue {:n}".format(resnum))

    def __sub__(self, other):
        """
        Compare two SIFt2D objects and return a new differential SIFt2D.
        """
        if self._mapping:
            if self.custom_residues_set == other.custom_residues_set:
                common_residues_set = self.custom_residues_set
                result = SIFt2D("", self.receptor_name, self.start, self.end,
                                self.custom_residues_set)
            else:
                common_residues_set = sorted(list(set(self.custom_residues_set)
                                                  | set(other.custom_residues_set)))
                result = SIFt2D("", self.receptor_name, self.start,
                                len(common_residues_set), common_residues_set)

            for res in common_residues_set:
                if res in self.custom_residues_set:
                    if res in other.custom_residues_set:
                        result.set_chunk(self[res] - other[res], res, True)
                    else:
                        result.set_chunk(self[res] - self.generic_chunk, res, True)
                else:
                    if res in other.custom_residues_set:
                        result.set_chunk(SIFt2DChunk(res) - other[res], res, True)
                    else:
                        print("This situation shouldn't take place. Residue: {}".format(res))
        else:
            result = SIFt2D("", self.receptor_name, min(self.start, other.start),
                            max(self.end, other.end))

            for res in range(min(self.start, other.start), max(self.end, other.end)+1):
                if res < self.start:
                    result.set_chunk(SIFt2DChunk(res) - other[res], res)
                elif res > self.end:
                    result.set_chunk(SIFt2DChunk(res) - other[res], res)
                elif res < other.start:
                    result.set_chunk(self[res] - self.generic_chunk, res)
                elif res > other.end:
                    result.set_chunk(self[res] - self.generic_chunk, res)
                else:
                    try:
                        result.set_chunk(self[res] - other[res], res)
                    except Exception as msg:
                        print("Something gone wrong for residue {!n}:\n{}".format(res, msg))
        return result


    def prepend_chunk(self, custom_residue=None):
        """
        Insert empty chunks at the beginning of the chunk list.
        """
        tmp = []
        #self._chunks
        if custom_residue:
            tmp_mapping = {}
            tmp.append(SIFt2DChunk(custom_residue))
            tmp_mapping[custom_residue] = 0
            for key in self._mapping.keys():
                tmp_mapping[key] = self._mapping[key] + 1


        else:
            tmp.append(SIFt2DChunk(self.start-1))
            self.start -= 1
        tmp.extend(self._chunks)
        self._chunks = tmp


    def append_chunk(self, custom_residue=None):
        """
        Add empty chunk at the end of the chunk list.
        """
        if custom_residue:
            self._chunks.append(SIFt2DChunk(custom_residue))
            self._mapping[custom_residue] = len(self._chunks) -1
        else:
            self._chunks.append(SIFt2DChunk(self.end + 1))
            self.end += 1


    def set_chunk(self, chunk, residue_number, custom_residue=False):
        """
        Replace chunk at given position with the input.
        """
        if chunk.resnum != residue_number:
            raise KeyError("Residue number mismatch {:n}, chunk: {:n}"
                           .format(residue_number, chunk.resnum))

        if custom_residue:
            if residue_number not in self._mapping.keys():
                raise KeyError("Residue {} not found!".format(residue_number))
            self._chunks[self._mapping[residue_number]] = chunk
        else:
            if not self.start <= residue_number <= self.end:
                raise KeyError("Residue {:n} out of range ({:n} {:n})"
                               .format(residue_number, self.start, self.end))
            if self._chunks[residue_number - self.start].resnum == residue_number:
                self._chunks[residue_number - self.start] = chunk


    def sort_chunks(self):
        """
        Applicable only for custom residues sets (residues don't cover given range).
        """
        tmp = []
        tmp_mapping = {}

        for idx, gn in enumerate(sorted(self._mapping.keys)):
            tmp_mapping[gn] = idx
            tmp.append(self._chunks[self._mapping[gn]])
        self._chunks = tmp
        self._mapping = tmp_mapping


    def get_chunks(self):
        """
        Returns a list of chunks.
        """
        if self._mapping:
            return [self._chunks[self._mapping[x]] for x in sorted(self._mapping.keys())]
        return self._chunks


    def get_listed_chunks(self, res_list):
        """
        The function returns a list of SIFt2DChunk objects corresponding
        to the input list of positions. In case of KeyError, an empty chunk is added.
        """
        if self._mapping:
            output = []
            for num in res_list:
                try:
                    output.append(self._chunks[self._mapping[num]])
                except KeyError:
                    output.append(SIFt2DChunk(num))
            return output


    def get_interacting_chunks(self):
        """
        Return out non-zero chunks.
        """
        return [x for x in self._chunks if x != self.generic_chunk]


    def get_numpy_array(self):
        """
        Return the 2D-SIFt in form of a numpy array.
        """

        return np.concatenate([x.chunk for x in self.get_chunks()], axis=1)


    def write(self, format_='txt', filename='', filehandle=None):
        """
        Export 2D-SIFt data into a file or file handle. Supported formats are "txt" and "yaml".
        File specified with @filename is opened in append mode.

        The function returns the output string in the specified format.
        """

        fh = None
        if filename != '':
            fh = open(filename, 'a')
        elif filehandle is not None:
            fh = filehandle
        try:
            interaction_matrix = np.concatenate([x.chunk for x in self._chunks], axis=1)
            fp_string = ';'.join([''.join([str(x) for x in interaction_matrix[y, :]])
                                  for y in range(interaction_matrix.shape[0])])
            if fh is not None:
                fh.write(OUTPUT_FORMAT[format_].substitute(
                    receptor=self.receptor_name, ligand=self.ligand_name,
                    start=self.start, fp=fp_string) + '\n')
        except KeyError:
            print("The specified format is not supported {!s}".format(format_))

        if filename != '':
            fh.close()
        return OUTPUT_FORMAT[format_].substitute(
            receptor=self.receptor_name,
            ligand=self.ligand_name,
            start=self.start,
            fp=fp_string)


    def get_heatmap(self, selected_chunks=None, colormap=plt.cm.Reds, vmin=None, vmax=None):
        """
        Returns a standardized heatmap figure.
        """
        if self._mapping:
            xticklabels = ["{}".format(x).replace('.', 'x') if len(str(x)) > 3
                           else "{:.2f}".format(float(x)).replace('.', 'x')
                           for x in selected_chunks]
        else:
            xticklabels = ["{}".format(x) for x in selected_chunks]

        heatmap = plt.figure(figsize=(20, 6))
        ax = heatmap.gca(xlabel="Residue number", ylabel="Pharmacophore features")
        ax.set_yticks(np.arange(len(ROWS))+0.5, minor=False)
        ax.set_yticks(np.arange(len(ROWS)), minor=True)
        ax.set_yticklabels(['D', 'A', 'H', 'N', 'P', 'R', 'vdW'], size='large')
        ax.invert_yaxis()
        ax.set_xlabel("Residue number", size='large')
        ax.set_ylabel("Pharmacophore features", size='large')

        if selected_chunks:
            ax.set_xticks(np.arange(len(selected_chunks))*len(COLUMNS)+0.5*len(COLUMNS),
                          minor=False)
            ax.set_xticks(np.arange(len(selected_chunks))*len(COLUMNS), minor=True)
            ax.set_xticklabels(xticklabels, rotation="vertical")
            if self._mapping:
                coll = plt.pcolor(
                    np.concatenate([x.chunk for x in [self._chunks[self._mapping[x]]
                                                      if x in self._mapping.keys()
                                                      else self.generic_chunk.chunk
                                                      for x in selected_chunks]],
                                   axis=1),
                    cmap=colormap,
                    figure=heatmap,
                    vmin=vmin,
                    vmax=vmax)
            else:
                coll = plt.pcolor(
                    np.concatenate([x.chunk for x in [self._chunks[x - self.start]
                                                      for x in selected_chunks]],
                                   axis=1),
                    cmap=colormap,
                    figure=heatmap,
                    vmin=vmin,
                    vmax=vmax)
        else:
            coll = plt.pcolor(
                np.concatenate([x.chunk for x in self._chunks], axis=1),
                cmap=colormap,
                figure=heatmap,
                vmin=vmin,
                vmax=vmax)
        ax.grid(b=True, which='minor')
        heatmap.colorbar(coll)

        return heatmap


#==============================================================================
class SIFt2DGenerator:
    """
    The class generating 2D-SIFt matrices for a given set of ligands and a receptor.
    The SIFt bits are generated for every residue in the input file,
    however the output file contains starting residue number.
    """

    generic_chunk = SIFt2DChunk()


    def __init__(self, rec_struct, ligs, cutoff=3.5,
                 use_generic_numbers=False, unique=False, property_=None):

        self.sifts = []
        self.starting_res_num = 0
        self.ending_res_num = 0
        self.seq_len = 0
        self._lig_list = []
        self._rings = []
        self._generic_numbers = None

        self.receptor_st = rec_struct
        self.cutoff = cutoff
        self.hbond_cutoff = 2.5
        if use_generic_numbers:
            self.receptor_st = rec_struct.extract(analyze.get_atoms_from_asl(
                rec_struct,
                "fillres (a. CA and a.pdbbfactor > -8.1 and a.pdbbfactor < 8.1)"))
            self._generic_numbers = self._get_generic_numbers_list()
            print(self._generic_numbers)
            self.receptor_st.title = rec_struct.title
        self._get_receptor_params()

        self.backbone_set = set(analyze.evaluate_asl(self.receptor_st,
                                                     "backbone"))
        for num, lig in enumerate(ligs):
            if property_:
                if unique and lig.property[property_] in self._lig_list:
                    continue
                self.sifts.append(SIFt2D(
                    lig.property[property_],
                    self.receptor_st.title,
                    self.starting_res_num,
                    self.ending_res_num,
                    self._generic_numbers))
                print("Working on ligand {!s} {:n}".format(lig.property[property_], num))
            else:
                if unique and lig.title in self._lig_list:
                    continue
                self.sifts.append(SIFt2D(
                    lig.title,
                    self.receptor_st.title,
                    self.starting_res_num,
                    self.ending_res_num,
                    self._generic_numbers))
                print("Working on ligand {!s} {:n}".format(lig.title, num))

            self.generate_2d_sift(lig, self.sifts[-1])
            if unique and property:
                self._lig_list.append(lig.property[property])
            elif unique:
                self._lig_list.append(lig.title)


    def __iter__(self):
        """
        Iterator over all 2DSIFts
        """
        for sift in self.sifts:
            yield sift


    def _get_receptor_params(self):
        """
        Extract starting and ending residues from structure.
        Prepare rings for finding pi-pi and pi-cation interactions(speedup trick).
        Private function.
        """
        if self._generic_numbers:
            self.starting_res_num = min(self._generic_numbers)
            self.ending_res_num = max(self._generic_numbers)
            self.seq_len = len(self._generic_numbers)
        else:
            self.starting_res_num = min([x.resnum for x in self.receptor_st.residue])
            self.ending_res_num = max([x.resnum for x in self.receptor_st.residue])
            self.seq_len = self.ending_res_num - self.starting_res_num
        self._rings = interactions.gather_rings(self.receptor_st)


    def _get_generic_numbers_list(self):
        """
        Extract the list of generic numbers assigned to the input receptor.
        Following GPCRdb convention, GPCRdb generic numbers are stored
        as b factors of CA atom of each residue.
        """
        return list(map(lambda x: (x.temperature_factor > 0 and x.temperature_factor)
                   or (x.temperature_factor < 0 and -x.temperature_factor + 0.001),
                   analyze.get_atoms_from_asl(
                       self.receptor_st,
                       "a. CA and a.pdbbfactor > -8.1 and a.pdbbfactor < 8.1")))


    def _get_generic_number(self, residue_number):
        """
        Retrieve GPCRdb generic number stored in CA temperature factor.
        """

        num = next(analyze.get_atoms_from_asl(
            self.receptor_st,
            "a. CA and r. {!s}".format(residue_number)
            )).temperature_factor
        if num > 0:
            return num
        else:
            return round(-num + 0.001, 3)

    # def _is_atom_in_feature(self, feat_name, atom):

    #     for subset in self.features[feat_name]:
    #         if atom in subset:
    #             return True
    #     return False

    # def _delete_matching_feture_subset(self, feat_name, atom):

    #     for subset in self.features[feat_name]:
    #         if atom in subset:
    #             self.features[feat_name].remove(subset)


    def generate_2d_sift(self, ligand, matrix):
        """
        Find interactions and encode the interaction matrix.
        """
        features = self.assign_pharm_feats(ligand)
        for residue in self.receptor_st.residue:
            for ftype in features.keys():
                if ftype == 'AROMATIC':
                    continue
                for feat in features[ftype]:
                    active_bits = self.find_feature_interactions(ftype, feat, ligand, residue)
                    if active_bits == []:
                        continue
                    if self._generic_numbers:
                        self.activate_bits(
                            active_bits,
                            ftype,
                            self._get_generic_number(residue.resnum),
                            matrix)
                    else:
                        self.activate_bits(active_bits, ftype, residue.resnum, matrix)
        #Aromatic interactions
        pipi_int = interactions.find_pi_pi_interactions(
            self.receptor_st,
            rings1=self._rings,
            struct2=ligand)
        #ring1 and ring2 are Centroid objects, ring1 comes from receptor,
        # ring2 from ligand, atoms in Centroid.atoms list are sorted
        for ppi in pipi_int:
            for aromatic_feat in features['AROMATIC']:
                if sorted(ppi.ring2.atoms) == sorted(aromatic_feat):
                    if self._generic_numbers:
                        self.activate_bits(
                            ['AROMATIC'],
                            'AROMATIC',
                            self._get_generic_number(
                                self.receptor_st.atom[ppi.ring1.atoms[0]].resnum),
                            matrix)
                    else:
                        self.activate_bits(
                            ['AROMATIC'],
                            'AROMATIC',
                            self.receptor_st.atom[ppi.ring1.atoms[0]].resnum,
                            matrix)
        picat_int = interactions.find_pi_cation_interactions(self.receptor_st, struct2=ligand)
        for pci in picat_int:
            if pci.cation_structure.title == self.receptor_st.title:
                if self._generic_numbers:
                    self.activate_bits(
                        ['CHARGED'],
                        'AROMATIC',
                        self._get_generic_number(
                            self.receptor_st.atom[pci.cation_centroid.atoms[0]].resnum),
                        matrix)
                else:
                    self.activate_bits(
                        ['CHARGED'],
                        'AROMATIC',
                        self.receptor_st.atom[pci.cation_centroid.atoms[0]].resnum,
                        matrix)
            else:
                if self._generic_numbers:
                    self.activate_bits(
                        ['AROMATIC'],
                        'P_CHARGED',
                        self._get_generic_number(
                            self.receptor_st.atom[pci.pi_centroid.atoms[0]].resnum),
                        matrix)
                else:
                    self.activate_bits(
                        ['AROMATIC'],
                        'P_CHARGED',
                        self.receptor_st.atom[pci.pi_centroid.atoms[0]].resnum,
                        matrix)


    def find_feature_interactions(self, feat_name, feat_atoms, ligand_st, residue):
        """
        The function evaluates type/distance-based interactions and h_bonds.
        Aromatic interactions are treated globally (on the receptor level),
        not on the atomic level.
        """
        rec_act_bits = []
        for ratom in residue.getAtomList():
            for atom in feat_atoms:
                atom_atom_dist = measure.measure_distance(
                    ligand_st.atom[atom],
                    self.receptor_st.atom[ratom])
                if atom_atom_dist <= self.cutoff:
                    if feat_name == ['H_ACCEPTOR'] and self.receptor_st.atom[ratom].atomic_number != 1:
                        continue
                    if feat_name == 'ANY':
                        rec_act_bits.append('ANY')
                        if int(ratom) in self.backbone_set:
                            rec_act_bits.append('BACKBONE')
                        else:
                            rec_act_bits.append('SIDECHAIN')
                        continue
                    if analyze.match_hbond(
                            ligand_st.atom[atom],
                            self.receptor_st.atom[ratom],
                            distance_max=self.cutoff):
                        if feat_name not in ["H_DONOR", "H_ACCEPTOR"]:
                            continue
                        if atom_atom_dist <= self.hbond_cutoff:
                            if self.receptor_st.atom[ratom].atomic_number == 1:
                                if self._generic_numbers:
                                    self.activate_bits(
                                        ['H_ACCEPTOR'],
                                        'H_DONOR',
                                        self._get_generic_number(residue.resnum),
                                        self.sifts[-1])
                                else:
                                    self.activate_bits(
                                        ['H_ACCEPTOR'],
                                        'H_DONOR',
                                        residue.resnum,
                                        self.sifts[-1])
                            else:
                                if self._generic_numbers:
                                    self.activate_bits(
                                        ['H_DONOR'],
                                        'H_ACCEPTOR',
                                        self._get_generic_number(residue.resnum),
                                        self.sifts[-1])
                                else:
                                    self.activate_bits(
                                        ['H_DONOR'],
                                        'H_ACCEPTOR',
                                        residue.resnum,
                                        self.sifts[-1])
                        else:
                            if self.receptor_st.atom[ratom].atomic_number == 1:
                                if self._generic_numbers:
                                    self.activate_bits(
                                        ['POLAR'],
                                        'H_DONOR',
                                        self._get_generic_number(residue.resnum),
                                        self.sifts[-1])
                                else:
                                    self.activate_bits(
                                        ['POLAR'],
                                        'H_DONOR',
                                        residue.resnum,
                                        self.sifts[-1])
                            else:
                                if self._generic_numbers:
                                    self.activate_bits(
                                        ['POLAR'],
                                        'H_ACCEPTOR',
                                        self._get_generic_number(residue.resnum),
                                        self.sifts[-1])
                                else:
                                    self.activate_bits(
                                        ['POLAR'],
                                        'H_ACCEPTOR',
                                        residue.resnum,
                                        self.sifts[-1])
                        if int(ratom) in self.backbone_set:
                            rec_act_bits.append('BACKBONE')
                        else:
                            rec_act_bits.append('SIDECHAIN')
                        return list(set(rec_act_bits))

                    if feat_name not in INTERACTION_RULES.keys():
                        continue
                    for r_feat in INTERACTION_RULES[feat_name][INTERACTION_DICT[
                            'residue_features']]:
                        if residue.pdbres.strip() in RESIDUE_SETS[r_feat]:
                            rec_act_bits.extend(
                                INTERACTION_RULES[feat_name][INTERACTION_DICT['byte_to_switch']])
        return list(set(rec_act_bits))


    def activate_bits(self, act_bits, feat, resnum, matrix):
        """
        Increment the value of residue chunk for given interaction.
        """
        for bit in act_bits:
            matrix[resnum].increment_bit(feat, bit)


    def assign_pharm_feats(self, lig_struct):
        """
        SMARTS based assignment of pharmacophore features (using Phase feature patterns).
        Rings are extracted using Structure.ring iterator.
        """
        matches = {}
        for key in PH_PATTERNS.keys():
            matches[key] = []
            for pattern in PH_PATTERNS[key]:
                s = sorted(analyze.evaluate_smarts_canvas(lig_struct, pattern, uniqueFilter=True))
                if s != [] and s not in matches[key]:
                    incl = 0
                    for x in s:
                        for y in matches[key]:
                            if set(x) <= set(y):
                                incl = 1

                    if incl == 1:
                        continue
                    matches[key].extend(s)
            if matches[key] == []:
                del matches[key]
        try:
            matches['HYDROPHOBIC'] = [list(set(reduce(lambda x, y: x+y, matches['HYDROPHOBIC'])))]
        except KeyError:
            pass
        matches['AROMATIC'] = [x.getAtomList() for x in lig_struct.ring]
        matches['ANY'] = [[x.index for x in lig_struct.atom]]
        return matches


    def write_all(self, outfile=None, mode='w'):
        """
        Batch saving all 2D-SIFt object stored in generator.
        """

        print("Writing interaction matrix to a file.")
        if outfile is None:
            outfile = '{!s}_2dfp.dat'.format(self.receptor_st.title)
        outfh = open(outfile, mode)
        if self._generic_numbers:
            outfh.write("#{!s}\n".format(';'.join([str(x) for x in sorted(self._generic_numbers)])))

        for sift in self.sifts:
            sift.write(filehandle=outfh)
        outfh.close()


class Ring:
    def __init__(self, pdb_res=None, coords=None):

        self.residue = None
        self.res_name = "lig"

        if pdb_res:
            at = list(pdb_res)
            self.residue = pdb_res
            self.res_name = self.residue.resname
            self.atoms = [
                x for x in at if x.get_name() not in ["C", "CA", "CB", "N", "O", "OH"]
            ]
            # print([x.get_vector() for x in self.atoms])
            self.center = np.average([x.coord for x in self.atoms], axis=0)

        if coords:
            self.atoms = coords
            self.center = np.average(self.atoms, axis=0)

class SIFt2DGeneratorNoMMLibs:
    """
    The class generating 2D-SIFt matrices for a given set of ligands and a receptor.
    The SIFt bits are generated for every residue in the input file,
    however the output file contains starting residue number.
    """

    generic_chunk = SIFt2DChunk()


    def __init__(self, pdb_file, ligand_file, cutoff=3.5,
                 use_generic_numbers=False, unique=False, property_=None):

        self.sifts = []
        self.starting_res_num = 0
        self.ending_res_num = 0
        self.seq_len = 0
        self._lig_list = []
        self._rings = []
        self._generic_numbers = None

        self.polar_resi = []
        self.hb_resi = []
        self.pos_chg_resi = []
        self.neg_chg_resi = []
        self.all_resi = []

        self._fdef = ChemicalFeatures.BuildFeatureFactory("BaseFeatures.fdef")
        self.ligand_structs = SDMolSupplier(ligand_file)

        self.receptor_name = os.path.splitext(os.path.basename(pdb_file))[0]

        self.receptor_st = self.parse_pdb(pdb_file)
        self.cutoff = cutoff
        self.hbond_cutoff = 2.5

        self._get_receptor_params()

        for num, lig in enumerate(self.ligand_structs):
            if property_:
                if unique and lig.property[property_] in self._lig_list:
                    continue
                self.sifts.append(SIFt2D(
                    lig.GetProp(property_),
                    self.receptor_name,
                    self.starting_res_num,
                    self.ending_res_num,
                    self._generic_numbers))
                print("Working on ligand {!s} {:n}".format(lig.GetProp(property_), num))
            else:
                if unique and lig.title in self._lig_list:
                    continue
                self.sifts.append(SIFt2D(
                    lig.GetProp("_Name"),
                    self.receptor_name,
                    self.starting_res_num,
                    self.ending_res_num,
                    self._generic_numbers))
                print("Working on ligand {!s} {:n}".format(lig.title, num))

            self.generate_2d_sift(lig, self.sifts[-1])
            if unique and property_:
                self._lig_list.append(lig.GetProp(property_))
            elif unique:
                self._lig_list.append(lig.GetProp("_Name"))

    def parse_pdb(self, pdb_file):

        pdb_struct = PDBParser(PERMISSIVE=True, QUIET=True).get_structure(
            "ref", pdb_file
        )[0]
        for chain in pdb_struct:
            for res in chain:
                resname = res.resname
                if resname in RESIDUE_SETS["AROMATIC_RESIDUES"]:
                    # print(list(res))
                    self._rings.append(Ring(res))
                if resname in RESIDUE_SETS["POLAR_RESIDUES"]:
                    self.polar_resi.append(res)
                if resname in RESIDUE_SETS["HYDROPHOBIC_RESIDUES"]:
                    self.hb_resi.append(res)
                if resname in RESIDUE_SETS["P_CHARGED_RESIDUES"]:
                    self.pos_chg_resi.append(res)
                if resname in RESIDUE_SETS["N_CHARGED_RESIDUES"]:
                    self.neg_chg_resi.append(res)
                self.all_resi.append(res)
        return pdb_struct

    def __iter__(self):
        """
        Iterator over all 2DSIFts
        """
        for sift in self.sifts:
            yield sift


    def _get_receptor_params(self):
        """
        Extract starting and ending residues from structure.
        Prepare rings for finding pi-pi and pi-cation interactions(speedup trick).
        Private function.
        """
        if self._generic_numbers:
            self.starting_res_num = min(self._generic_numbers)
            self.ending_res_num = max(self._generic_numbers)
            self.seq_len = len(self._generic_numbers)
        else:
            self.starting_res_num = min([x.resnum for x in self.receptor_st.residue])
            self.ending_res_num = max([x.resnum for x in self.receptor_st.residue])
            self.seq_len = self.ending_res_num - self.starting_res_num
        self._rings = interactions.gather_rings(self.receptor_st)


    def _get_generic_numbers_list(self):
        """
        Extract the list of generic numbers assigned to the input receptor.
        Following GPCRdb convention, GPCRdb generic numbers are stored
        as b factors of CA atom of each residue.
        """
        pass


    def _get_generic_number(self, residue_number):
        """
        Retrieve GPCRdb generic number stored in CA temperature factor.
        """

        pass


    def generate_2d_sift(self, ligand, sift):
        """
        Find interactions and encode the interaction matrix.
        """
        feats = self._fdef.GetFeaturesForMol(ligand)

        mol = AddHs(ligand)
        mol3d = mol.GetConformer()
        for feat in feats:
            fam = feat.GetFamily()
            feat_atoms = feat.GetAtomIds()

            # Charge
            if fam == "PosIonizable":
                for pc_res in self.neg_chg_resi:
                    for feat_atom in feat_atoms:
                        if (
                            get_shortest_distance(
                                pc_res, mol3d.GetAtomPosition(feat_atom)
                            )
                            < COULOMB_CUTOFF
                        ):
                            sift.activate_bits("CHARGED", 'N_CHARGED', pc_res.id[1], sift)
            if fam == "NegIonizable":
                for pc_res in self.pos_chg_resi:
                    for feat_atom in feat_atoms:
                        if (
                            get_shortest_distance(
                                pc_res, mol3d.GetAtomPosition(feat_atom)
                            )
                            < COULOMB_CUTOFF
                        ):
                            sift.activate_bits("CHARGED", 'PCHARGED', pc_res.id[1], sift)

            # H-bond

            if fam == 'Donor':
                for acc_res in self.polar_resi:
                    if acc_res.resname not in HB_ACCEPTOR:
                        continue
                    for feat_atom in feat_atoms:
                        dist = get_shortest_distance(acc_res, mol3d.GetAtomPosition(feat_atom))
                        if dist < HBOND_CUTOFF:
                            sift.activate_bits("H_ACCEPTOR", "H_DONOR", acc_res.id[1], sift)
                        elif dist < POLAR_CUTOFF:
                            sift.activate_bits("POLAR", "H_DONOR", acc_res.id[1], sift)

            if fam == 'Acceptor':
                for don_res in self.polar_resi:
                    if don_res.resname not in HB_DONOR:
                        continue
                    for feat_atom in feat_atoms:
                        dist = get_shortest_distance(don_res, mol3d.GetAtomPosition(feat_atom))
                        if dist < HBOND_CUTOFF:
                            sift.activate_bits("H_DONOR", "H_ACCEPTOR", acc_res.id[1], sift)
                        elif dist < POLAR_CUTOFF:
                            sift.activate_bits("POLAR", "H_ACCEPTOR", acc_res.id[1], sift)

            # Arom
            if fam == 'Aromatic':
                feat_ring = Ring(coords=[mol3d.GetAtomPosition(x) for x in feat_atoms])
                # print(feat_ring.center)
                for pdb_ring in self._rings:
                    # print("Distance from {} {} ring is : {}".format(pdb_ring.residue.id[1], pdb_ring.res_name, get_distance(feat_ring.center, pdb_ring.center)))
                    if get_distance(feat_ring.center, pdb_ring.center) < PI_CUTOFF:
                        sift.activate_bits('AROMATIC', "AROMATIC", pdb_ring.residue.id[1], sift)

            # Hydrophobic

            if fam in ["Hydrophobe", "LumpedHydrophobe"]:
                for hb_res in self.hb_resi:
                    for feat_atom in feat_atoms:
                        if (
                            get_shortest_distance(hb_res, mol3d.GetAtomPosition(feat_atom))
                            < VDW_CUTOFF
                        ):
                            sift.activate_bits("HYDROPHOBIC", "HYDROPHOBIC", hb_res.id[1], sift)
        for res in self.all_resi:
            if res.id[0].startswith('H_'): #skippping HETs
                continue
            SC = []
            BB = []
            for atom in res:
                if atom.get_name() in BACKBONE_ATOMS:
                    BB.append(atom)
                else:
                    SC.append(atom)
            for lig_atom in mol3d.GetPositions():
                if (lig_atom == [0., 0., 0.]).all():
                    continue
                try:
                    if SC != []: #Gly case
                        if get_shortest_distance_atoms(SC, lig_atom) < VDW_CUTOFF:
                            sift.activate_bits("ANY", "ANY", res.id[1], sift)
                            sift.activate_bits("SIDECHAIN", "ANY", res.id[1], sift)
                    if get_shortest_distance_atoms(BB, lig_atom) < VDW_CUTOFF:
                        sift.activate_bits("ANY", "ANY", res.id[1], sift)
                        sift.activate_bits("BACKBONE", "ANY", res.id[1], sift)
                except:
                    pass

        return sift


    def activate_bits(self, act_bits, feat, resnum, matrix):
        """
        Increment the value of residue chunk for given interaction.
        """
        for bit in act_bits:
            matrix[resnum].increment_bit(feat, bit)


    def assign_pharm_feats(self, mol):
        """
        SMARTS based assignment of pharmacophore features (using Phase feature patterns).
        Rings are extracted using Structure.ring iterator.
        """
        feats = self._fdef.GetFeaturesForMol(mol)
        mol = AddHs(mol)
        matches = {
            'H_DONOR' : [],
            'H_ACCEPTOR' : [],
            'HYDROPHOBIC' : [],
            'N_CHARGED' : [],
            'P_CHARGED' : [],
            'AROMATIC' : [],
            'ANY': [x.GetIdx() for x in mol.GetAtoms()]
        }

        for feat in feats:
            fam = feat.GetFamily()
            feat_atoms = feat.GetAtomIds()

            if fam == "PosIonizable":
                matches['P_CHARGED'].append(feat_atoms)
            if fam == "NegIonizable":
                matches['N_CHARGED'].append(feat_atoms)
            if fam == 'Donor':
                matches['H_DONOR'].append(feat_atoms)

            if fam == 'Acceptor':
                matches['H_ACCEPTOR'].append(feat_atoms)
            if fam == 'Aromatic':
                matches['AROMATIC'].append(feat_atoms)

            if fam in ["Hydrophobe", "LumpedHydrophobe"]:
                matches['HYDROPHOBIC'].append(feat_atoms)

        for key in matches:
            if matches[key] == []: del matches[key]

        return matches


    def write_all(self, outfile=None, mode='w'):
        """
        Batch saving all 2D-SIFt object stored in generator.
        """

        print("Writing interaction matrix to a file.")
        if outfile is None:
            outfile = '{!s}_2dfp.dat'.format(self.receptor_st.title)
        outfh = open(outfile, mode)
        if self._generic_numbers:
            outfh.write("#{!s}\n".format(';'.join([str(x) for x in sorted(self._generic_numbers)])))

        for sift in self.sifts:
            sift.write(filehandle=outfh)
        outfh.close()


#==============================================================================
class SIFt2DReader:
    """
    Class for reading stored interaction matrices. Returning iterator of SIFt2D objects read.
    """


    def __init__(self, filename=None, filehandle=None, input_string=None, format_='txt'):

        self._sifts = []
        self.custom_residues_set = None

        if not filename and not filehandle and not input_string:
            raise IOError("Some input must be specified, file name, file handle or input string.")
        if format_ not in OUTPUT_FORMAT:
            raise IOError("Specified format is not supported.")
        if filename:
            fh = open(filename, 'r')
            self._sifts = self.fh_reader(fh, format_)
        elif filehandle:
            self._sifts = self.fh_reader(filehandle, format_)
        else:
            self._sifts = self.format_parser(input_string, format_)


    def fh_reader(self, fh, format_):

        output = []

        resi_defs = fh.readline()
        if resi_defs.startswith('#'):
            self.custom_residues_set = resi_defs[1:].strip().split(';')
        else:
            output.append(self.format_parser(resi_defs, format_))
        for line in fh:
            if line.strip() == '':
                continue
            output.append(self.format_parser(line.strip(), format_))

        return output


    def format_parser(self, line, format_):
        """
        Read the 2D-SIFts from a text string.
        """

        if format_ == 'txt':
            receptor, ligand, start, im = line.strip().split(':')
            start = int(math.floor(float(start)))
            im = im.split(';')
            if len(im) != len(ROWS.keys()):
                raise IOError(
                    "Shape of input interaction matrix does not correspond to defined shape.")
            end = start + len(im[0][::9]) - 1
            im = np.array([[int(x) for x in y] for y in im])
            chunks = np.hsplit(im, len(im[0][::9]))
        if self.custom_residues_set:
            output = SIFt2D(ligand, receptor, start, end, self.custom_residues_set)
            for num, chunk in zip(self.custom_residues_set, chunks):
                output.set_chunk(SIFt2DChunk(num, chunk), num, True)
        else:
            output = SIFt2D(ligand, receptor, start, end)
            for num, chunk in enumerate(chunks):
                output.set_chunk(SIFt2DChunk(start+num, chunk), start+num)
        return output


    def __len__(self):

        return len(self._sifts)


    def __iter__(self):
        """
        Iterate over read SIFt2D objects.
        """
        for sift in self._sifts:
            yield sift



#==============================================================================
class SIFt2DProfile(SIFt2D):
    """
    Class representing averaged SIFt2D objects.
    """

    def __init__(self, sifts=None, start=None, end=None, generic_numbers=None):
        SIFt2D.__init__(self, '', 'average', 0, 0, generic_numbers)
        self._sifts = sifts
        if sifts:
            if not generic_numbers:
                self.start, self.end = self._get_residue_range()
        elif start and end:
            self.start = start
            self.end = end
        self._chunks = []
        self._init_chunks()


    def _get_residue_range(self):
        """
        Return starting and ending residue for given set of input interaction matrices.
        """
        start = min([x.start for x in self._sifts])
        end = max([x.end for x in self._sifts])
        if start > end:
            raise ValueError("Starting residue greater than ending one. Start: {:n} End {:n}"
                             .format(start, end))
        return (start, end)


    def calculate_average(self):
        """
        Average the 2D-SIFt strings.
        """

        if self._mapping:
            gn = self.get_common_generic_residues_list()
            #TODO Czy to dobry pomysl?
            if gn == sorted(self._mapping.keys()):
                avg = np.mean(
                    [[x.chunk for x in y.get_listed_chunks(gn)] for y in self._sifts],
                    axis=0)
                for gn, x in zip(sorted(self._mapping), avg):
                    self.set_chunk(SIFt2DChunk(gn, x), gn, True)
            else:
                print("Nie w domu, ale tez na d")
                print(gn)
                print(self._mapping.keys())
        else:
            self.normalize_sifts()
            avg = np.mean([[x.chunk for x in y.get_chunks()] for y in self._sifts], axis=0)
            for idx, x in enumerate(avg):
                self.set_chunk(SIFt2DChunk(idx + self.start, x), idx + self.start)


    def get_interacting_chunks(self, cutoff=0.3):
        """
        Return chunks with at least one value above the cutoff.
        """
        return [x for x in self._chunks if x.chunk.max() >= cutoff or -x.chunk.min() >= cutoff]


    def get_common_generic_residues_list(self):

        return sorted(list(set(itertools.chain.from_iterable(
            [x.custom_residues_set for x in self._sifts]))))


    def normalize_sifts(self):
        """
        The function evens the lengths of sift matrices.
        """

        if self._mapping:
            print("Not yet supported for generic numbers.")
        start = min([x.start for x in self._sifts])
        end = max([x.end for x in self._sifts])
        for sift in self._sifts:
            if sift.start > start:
                for _ in range(sift.start - start):
                    sift.prepend_chunk()
            if sift.end < end:
                for _ in range(end - sift.end):
                    sift.append_chunk()


    def apply_cutoff(self, cutoff):
        """
        Nullify chunks with values below the cutoff.
        """
        for x in self._chunks:
            if x.chunk.max() < cutoff:
                x.chunk = np.zeros((len(ROWS), len(COLUMNS)), dtype=np.int)


    def write(self, format_='txt', filename='', filehandle=None):
        """
        Export 2D-SIFt profile data into a file or file handle.
        Supported formats are "txt" and "yaml".
        File specified with @filename is opened in append mode.

        The function returns the output string in the specified format.
        """

        fh = None
        if filename != '':
            fh = open(filename, 'w')
        elif filehandle is not None:
            fh = filehandle
        try:
            if self._mapping:
                fh.write("#{!s}\n".format(';'.join([str(x) for x in sorted(self._mapping.keys())])))
            interaction_matrix = np.concatenate([x.chunk for x in self.get_chunks()], axis=1)
            fp_string = ';'.join([' '.join([str(x) for x in interaction_matrix[y, :]])
                                  for y in range(interaction_matrix.shape[0])])
            if fh is not None:
                fh.write(
                    OUTPUT_FORMAT[format_].substitute(
                        receptor=self.receptor_name,
                        ligand=self.ligand_name,
                        start=self.start,
                        fp=fp_string))
        except KeyError:
            print("The specified format is not supported {!s}".format(format_))

        if filename != '':
            fh.close()
        return OUTPUT_FORMAT[format_].substitute(
            receptor=self.receptor_name,
            ligand=self.ligand_name,
            start=self.start, fp=fp_string)


    def __sub__(self, other):
        """
        Compare two SIFt2DProfile objects and return a new differential SIFt2DProfile.
        """
        if self._mapping:
            if self.custom_residues_set == other.custom_residues_set:
                common_residues_set = self.custom_residues_set
                result = SIFt2DProfile(None, self.start, self.end, self.custom_residues_set)
            else:
                common_residues_set = sorted(list(set(self.custom_residues_set)
                                                  | set(other.custom_residues_set)))
                result = SIFt2DProfile(
                    None,
                    self.start,
                    len(common_residues_set),
                    common_residues_set)

            for res in common_residues_set:
                if res in self.custom_residues_set:
                    if res in other.custom_residues_set:
                        result.set_chunk(self[res] - other[res], res, True)
                    else:
                        result.set_chunk(self[res] - self.generic_chunk, res, True)
                else:
                    if res in other.custom_residues_set:
                        result.set_chunk(SIFt2DChunk(res) - other[res], res, True)
                    else:
                        print("This situation shouldn't take place. Residue: {}".format(res))
        else:
            result = SIFt2DProfile(None, min(self.start, other.start), max(self.end, other.end))

            for res in range(min(self.start, other.start), max(self.end, other.end)+1):
                if res < self.start:
                    result.set_chunk(SIFt2DChunk(res) - other[res], res)
                elif res > self.end:
                    result.set_chunk(SIFt2DChunk(res) - other[res], res)
                elif res < other.start:
                    result.set_chunk(self[res] - self.generic_chunk, res)
                elif res > other.end:
                    result.set_chunk(self[res] - self.generic_chunk, res)
                else:
                    try:
                        result.set_chunk(self[res] - other[res], res)
                    except Exception as e:
                        print("Something gone wrong for residue {!n}:\n{}".format(res, e))
        return result

#==============================================================================
class SIFt2DProfileReader:
    """
    Class for reading interaction profiles
    """


    def __init__(self, filename=None, filehandle=None, input_string=None, profile_format='txt'):

        self._profile = None
        self.custom_residues_set = None

        if not filename and not filehandle and not input_string:
            raise IOError("Some input must be specified, file name, file handle or input string.")
        if format not in OUTPUT_FORMAT.keys():
            raise IOError("Specified format is not supported.")
        if filename:
            fh = open(filename, 'r')
            self._profile = self.fh_reader(fh, profile_format=profile_format)
        elif filehandle:
            self._profile = self.fh_reader(filehandle, profile_format=profile_format)
        else:
            self._profile = self.fh_reader(fp_string=input_string, profile_format=profile_format)


    def fh_reader(self, fh=None, fp_string=None, profile_format='txt'):

        output = None
        #The profile should contain 2 lines maximum. Anything more will be ignored
        if fh:
            profile_lines = fh.readlines()
        elif fp_string:
            profile_lines = fp_string.split('\n')
        else:
            return output
        if len(profile_lines) > 1:
            resi_defs = profile_lines[0]
            if resi_defs.startswith('#'):
                self.custom_residues_set = resi_defs[1:].strip().split(';')
                output = self.format_parser(profile_lines[1], profile_format)
            else:
                output = self.format_parser(profile_lines[0], profile_format)
        else:
            output = self.format_parser(profile_lines[0], profile_format)
        return output


    def format_parser(self, line, profile_format):

        output = []
        if profile_format == 'txt':
            _, _, start, im = line.strip().split(':')
            start = int(math.floor(float(start)))
            im = im.split(';')
            if len(im) != len(ROWS.keys()):
                raise IOError(
                    "Shape of input interaction matrix does not correspond to defined shape.")
            end = start + len(im[0].split(' ')[::9]) - 1
            im = np.array([[float(x) for x in y.split(' ')] for y in im])
            chunks = np.hsplit(im, len(im[0][::9]))
        if self.custom_residues_set:
            output = SIFt2DProfile(generic_numbers=self.custom_residues_set)
            for num, chunk in zip(self.custom_residues_set, chunks):
                output.set_chunk(SIFt2DChunk(num, chunk), num, True)
        else:
            output = SIFt2DProfile(start=start, end=end)
            for num, chunk in enumerate(chunks):
                output.set_chunk(SIFt2DChunk(start+num, chunk), start+num)
        return output


    def get_profile(self):
        return self._profile


#==============================================================================
def readProfile(filename='', filehandle=None, input_string=None, profile_format='txt'):

    if filename != '':
        print(filename)
        return SIFt2DProfileReader(filename, profile_format=profile_format).get_profile()
    if filehandle:
        return SIFt2DProfileReader(filehandle=filehandle, profile_format=profile_format).get_profile()
    if input_string:
        return SIFt2DProfileReader(input_string=input_string, profile_format=profile_format).get_profile()
    return



#==============================================================================
if __name__ == '__main__':
    print("This is the library for generation and manipulation of the 2D-SIFt descriptors.")
