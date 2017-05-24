from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np
import os
import pymatgen

##########################
structure_id = 'mp-149'
##########################

rester =  pymatgen.MPRester(os.environ['PMG_MAPI_KEY'])
pmg_structure = rester.get_structure_by_material_id(structure_id)
spa = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(pmg_structure)

conventional = spa.get_conventional_standard_structure()
primitive = spa.get_primitive_standard_structure()

primitive_matrix = np.dot(np.linalg.inv(conventional.lattice_vectors()), primitive.lattice_vectors())
np.round(primitive_matrix, decimals=6).tolist()

structure = StructureData(pymatgen=conventional).store()

crystal_system = spa.get_crystal_system()
if crystal_system == 'hexagonal':
    supercell = [[3, 0, 0],
                 [0, 3, 0],
                 [0, 0, 3]]
else:
    supercell = [[2, 0, 0],
                 [0, 2, 0],
                 [0, 0, 2]]

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
               'symbols': np.unique(conventional.symbol_set).tolist()}

# Monkhorst-pack
kpoints_dict = {'points': [2, 2, 2],
                'shift' : [0.0, 0.0, 0.0]}

machine_dict = {
    'num_machines': 1,
    'parallel_env':'mpi*',
    'tot_num_mpiprocs': 16}


ph_dict = ParameterData(dict={'supercell': supercell,
                              'primitive': primitive_matrix,
                              'distance': 0.01,
                              'mesh': [20, 20, 20]}
                       ).store()

wf_parameters = {
     'structure': structure,
     'phonopy_input': ph_dict,
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
wf = WorkflowPhonon(params=wf_parameters, optimize=False)

wf.label = 'VASP {}'.format(structure.get_formula())
wf.start()
print ('pk: {}'.format(wf.pk))

