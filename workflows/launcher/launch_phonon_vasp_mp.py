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

def get_supercell_size(structure, max_atoms=100):

    cell = np.array(structure.cell)

    num_atoms = len(structure.sites)

    print cell
    print num_atoms
    print np.linalg.norm(cell, axis=1)


    supercell_size = [1, 1, 1]

    while True:

        norm = np.linalg.norm(cell, axis=1)
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

        cell = np.dot(cell.T, np.diag(supercell_size)).T
        print cell


def get_potential_labels(functional, symbol_list, ftype=None):
    potential_labels =[]
    for symbol in np.unique(symbol_list):
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
spa = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(pmg_structure)

conventional = spa.get_conventional_standard_structure()
primitive = spa.get_primitive_standard_structure()

primitive_matrix = np.dot(np.linalg.inv(conventional.lattice_vectors()), primitive.lattice_vectors())
primitive_matrix = np.round(primitive_matrix, decimals=6).tolist()

structure = StructureData(pymatgen=conventional).store()
print structure
# crystal_system = spa.get_crystal_system()
# if crystal_system == 'hexagonal':
#     supercell = [[3, 0, 0],
#                  [0, 3, 0],
#                  [0, 0, 3]]
# else:
#     supercell = [[2, 0, 0],
#                  [0, 2, 0],
#                  [0, 0, 2]]

supercell_size = get_supercell_size(structure)
supercell = np.diag(supercell_size)

print ('Supercell shape: {}'.format(supercell_size))
exit()

incar_dict = {
    'NELMIN' : 5,
    'NELM'   : 100,
    'ENCUT'  : 400,
    'ALGO'   : 38,
    'ISMEAR' : 0,
    'SIGMA'  : 0.01,
    'GGA'    : 'PS'
}

pseudo_dict = {'functional': 'PBE',
               'symbols': get_potential_labels('PBE', conventional.symbol_set)}

# Monkhorst-pack
kpoints_dict = {'points': [2, 2, 2],
                'shift' : [0.0, 0.0, 0.0]}

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

