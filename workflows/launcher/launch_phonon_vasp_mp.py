from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory, WorkflowFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np
import os
import pymatgen
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


##########################
structure_id = 'mp-5777'
##########################

# Define the supercell size to use to calculate the forces in phonopy
def get_supercell_size(structure, max_atoms=100, crystal_system=None):

    def axis_symmetry(axis, crystal_system):
        symmetry_dict = {'cubic':      [1, 1, 1],
                         'hexagonal':  [1, 1, 2],
                         'tetragonal': [1, 1, 2],
                         'monoclinic': [1, 1, 2],
                         'trigonal':   [1, 1, 2]}

        try:
            return np.where(np.array(symmetry_dict[crystal_system]) == symmetry_dict[crystal_system][axis])[0]
        except KeyError:
            # If symmetry not defined in symmetry_dict or is None
            return np.array([0, 1, 2])

    cell = np.array(structure.cell)
    num_atoms = len(structure.sites)

    supercell_size = [1, 1, 1]
    while True:

        test_cell = np.dot(cell.T, np.diag(supercell_size)).T
        norm = np.linalg.norm(test_cell, axis=1)
        index = np.argmin(norm)
        supercell_size_test = list(supercell_size)

        for i in axis_symmetry(index, crystal_system):
            supercell_size_test[i] += 1

        anum_atoms_supercell = num_atoms * np.prod(supercell_size_test)
        if anum_atoms_supercell > max_atoms:
            atoms_minus =  num_atoms * np.prod(supercell_size)
            atoms_plus = num_atoms * np.prod(supercell_size_test)

            if max_atoms - atoms_minus < atoms_plus - max_atoms:
                return supercell_size
            else:
                return supercell_size_test
        else:
            supercell_size = supercell_size_test


# Decide the pseudopotential file to use for each atom according to request and availability
def get_pseudopotential_labels(functional, symbol_list, ftype=None):

    _, index = np.unique(symbol_list, return_index=True)
    symbol_list_unique = np.array(symbol_list)[np.sort(index)]

    potential_labels =[]
    for symbol in symbol_list_unique:
        psp_dir = os.environ['VASP_PSP_DIR'] + '/POT_GGA_PAW_' + functional
        all_labels = [f[7:-3] for f in os.listdir(psp_dir)  if os.path.isfile(os.path.join(psp_dir, f))]
        candidates = [ s for s in all_labels if symbol == s.split('_')[0]]
        if ftype is not None:
            final = [s for s in candidates if '_{}'.format(ftype) == s[-(len(ftype) + 1):]]
            if len(final) > 0:
                potential_labels.append(final[0])
                continue
        potential_labels.append(candidates[0])
    return potential_labels


# Get the crystal structure and other info. from Materials Project (using pymatgen)
rester = pymatgen.MPRester(os.environ['PMG_MAPI_KEY'])

pmg_structure = rester.get_structure_by_material_id(structure_id)
pmg_band = rester.get_bandstructure_by_material_id(structure_id)

material_name = pmg_structure.formula.replace('1','').replace(' ','')
print (material_name)

spa = SpacegroupAnalyzer(pmg_structure)

conventional = spa.get_conventional_standard_structure()
primitive = spa.get_primitive_standard_structure()

primitive_matrix = np.dot(np.linalg.inv(conventional.lattice.matrix), primitive.lattice.matrix)
primitive_matrix = np.round(primitive_matrix, decimals=6).tolist()

structure = StructureData(pymatgen=conventional).store()

print (conventional)
print (structure)

crystal_system = spa.get_crystal_system()
print ('Crystal system: {}'.format(crystal_system))

supercell_size = get_supercell_size(structure, crystal_system=crystal_system)
supercell = np.diag(supercell_size).tolist()
print ('Supercell shape: {}'.format(supercell_size))


# Criteria to decide VASP input according to band gap
band_gap = pmg_band.get_band_gap()['energy']
if band_gap > 3.0:
    system = 'insulator'
elif band_gap > 0.01:
    system = 'semiconductor'
else:
    system = 'metal'

print ('system: {}'.format(system))


if system == 'insulator' or system == 'semiconductor':
    incar_dict = {
        'NELMIN' : 5,
        'NELM'   : 100,
        'ALGO'   : 38,
        'ISMEAR' : 0,
        'SIGMA'  : 0.05,
        'GGA'    : 'PS'
    }


if system == 'metal':
    incar_dict = {
        'NELMIN' : 5,
        'NELM'   : 100,
        'ALGO'   : 38,
        'ISMEAR' : 1,
        'SIGMA'  : 0.2,
        'GGA'    : 'PS'
    }


pseudo_dict = {'functional': 'PBE',
               'symbols': get_pseudopotential_labels('PBE', conventional.symbol_set)}

print pseudo_dict

# Decide the size of the k-points mesh for electrons according to atom number and crystal structure
# This is implemented in pymatgen which is called in phonon_workflow using 'Automatic' style
if system == 'insulator' or system == 'semiconductor':
    # 100 Kpoints/atom
    kpoints_per_atom = 300

    # 1000 kpoints/atom
if system == 'metal':
    kpoints_per_atom = 1200

kpoints_dict = {'style': 'Automatic',
                'kpoints_per_atom': kpoints_per_atom}

# Cluster machine parameters
machine_dict = {
    'num_machines': 1,
    'parallel_env':'mpi*',
    'tot_num_mpiprocs': 16}

# Phonopy parameteres
phonopy_parameters = {'supercell': supercell,
                      'primitive': primitive_matrix,
                      'distance': 0.01,
                      'mesh': [20, 20, 20],
                      'symmetry_precision': 1e-5}

# Global parameters dictionary that contains all the parameters defined before
wf_parameters = {
     'structure': structure,
     'phonopy_input': {'parameters': phonopy_parameters},
     'input_force': {'code': 'vasp544mpi@boston',
                     'parameters': incar_dict,
                     'resources': machine_dict,
                     'pseudo': pseudo_dict,
                     'kpoints': kpoints_dict},
     'input_optimize': {'code': 'vasp544mpi@boston',
                        'parameters': incar_dict,
                        'resources': machine_dict,
                        'pseudo': pseudo_dict,
                        'kpoints': kpoints_dict}
}

#Define calculation to perform and lauch
WorkflowPhonon = WorkflowFactory('wf_phonon')
wf = WorkflowPhonon(params=wf_parameters, optimize=True, include_born=True)

wf.label = material_name
wf.description = 'PHON {}'.format(structure.get_formula())

wf.start()

print ('pk: {}'.format(wf.pk))

