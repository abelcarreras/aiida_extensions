from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np
import os
import pymatgen

##########################
structure_id = 'mp-12668'
##########################


def round_up_to_odd(f):
    return int((np.ceil(f) - 0.5 ) // 2 * 2 + 1)

def round_up_to_even(f):
    return int(np.ceil(f) // 2 * 2)

# Large k-meshes use odd number, else even
def get_kpoint_mesh_shape(kpoint_per_atom, structure, supercell=(1,1,1)):

    reciprocal_cell =  np.linalg.inv(structure.cell)*2*np.pi
    reciprocal_norm = np.linalg.norm(reciprocal_cell, axis=0)


    num_atoms = len(structure.sites)
    supercell_size = np.product(supercell)

    size = np.power(kpoint_per_atom * num_atoms / supercell_size, 1./3)
    if size > 8:
        size = round_up_to_odd(size)
    else:
        size = round_up_to_even(size)

    return [size, size, size]

def get_supercell_size(structure, max_atoms=100):

    cell = np.array(structure.cell)
    num_atoms = len(structure.sites)

    supercell_size = [1, 1, 1]
    while True:

        test_cell = np.dot(cell.T, np.diag(supercell_size)).T
        norm = np.linalg.norm(test_cell, axis=1)
        index = np.argmin(norm)
        supercell_size_test = list(supercell_size)
        supercell_size_test[index] += 1

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


def get_potential_labels(functional, symbol_list, ftype=None):

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


rester = pymatgen.MPRester(os.environ['PMG_MAPI_KEY'])

pmg_structure = rester.get_structure_by_material_id(structure_id)
pmg_band = rester.get_bandstructure_by_material_id(structure_id)

spa = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(pmg_structure)

conventional = spa.get_conventional_standard_structure()
primitive = spa.get_primitive_standard_structure()

primitive_matrix = np.dot(np.linalg.inv(conventional.lattice_vectors()), primitive.lattice_vectors())
primitive_matrix = np.round(primitive_matrix, decimals=6).tolist()

structure = StructureData(pymatgen=conventional).store()
print structure

crystal_system = spa.get_crystal_system()
print 'Crystal system: {}'.format(crystal_system)

# if crystal_system == 'hexagonal':
#     supercell = [[3, 0, 0],
#                  [0, 3, 0],
#                  [0, 0, 3]]
# else:
#     supercell = [[2, 0, 0],
#                  [0, 2, 0],
#                  [0, 0, 2]]

supercell_size = get_supercell_size(structure)
supercell = np.diag(supercell_size).tolist()
print ('Supercell shape: {}'.format(supercell_size))


# Criteria for INPUT
band_gap = pmg_band.get_band_gap()['energy']
if band_gap > 3.0:
    system = 'insulator'
elif band_gap > 0.01:
    system = 'semiconductor'
else:
    system = 'metal'

print 'system: {}'.format(system)


if system == 'insulator' or system == 'semiconductor':
    incar_dict = {
        'NELMIN' : 5,
        'NELM'   : 100,
        'ALGO'   : 38,
        'ISMEAR' : -5,
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
               'symbols': get_potential_labels('PBE', conventional.symbol_set)}

# Monkhorst-pack
if system == 'insulator' or system == 'semiconductor':
    # 100 Kpoints/atom
    kpoints_per_atom = 100

    # 1000 kpoints/atom
if system == 'metal':
    kpoints_per_atom = 1000

if crystal_system == 'hexagonal':
    style = 'Gamma'
else:
    style = 'Monkhorst'

kpoints_dict = {'style': 'Automatic',
                'kpoints_per_atom': kpoints_per_atom}

# kpoints_shape = get_kpoint_mesh_shape(kpoints_per_atom, structure)
# kpoints_dict = {'style': style,
#                'points': kpoints_shape,
#                'shift': [0.0, 0.0, 0.0]}


# kpoints_shape_supercell = get_kpoint_mesh_shape(kpoints_per_atom, structure, supercell=supercell_size)
# kpoints_dict_supercell = {'style': style,
#                           'points': kpoints_shape_supercell,
#                           'shift': [0.0, 0.0, 0.0]}

# print 'kpoints: {}'.format(kpoints)
# print 'kpoints (supercell): {}'.format(kpoints_shape_supercell)
# print 'shift {}'.format(kshift)

machine_dict = {
    'num_machines': 1,
    'parallel_env':'mpi*',
    'tot_num_mpiprocs': 16}


phonopy_parameters = {'supercell': supercell,
                      'primitive': primitive_matrix,
                      'distance': 0.01,
                      'mesh': [20, 20, 20]}

wf_parameters = {
     'structure': structure,
     'phonopy_input': {'parameters': phonopy_parameters},
     'input_force': {'code': 'vasp541mpi@stern',
                    'parameters': incar_dict,
                    'resources': machine_dict,
                    'pseudo': pseudo_dict,
                    'kpoints': kpoints_dict},
     'input_optimize': {'code': 'vasp541mpi@stern',
                       'parameters': incar_dict,
                       'resources': machine_dict,
                       'pseudo': pseudo_dict,
                       'kpoints': kpoints_dict}
}



#Submit workflow
from aiida.workflows.wf_phonon import WorkflowPhonon
wf = WorkflowPhonon(params=wf_parameters, optimize=True)

wf.label = 'VASP {}'.format(structure.get_formula())
wf.start()
print ('pk: {}'.format(wf.pk))

