from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory, WorkflowFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np

# Silicon structure
a = 5.404
cell = [[a, 0, 0],
        [0, a, 0],
        [0, 0, a]]

symbols=['Si'] * 8
scaled_positions = [(0.875,  0.875,  0.875),
                    (0.875,  0.375,  0.375),
                    (0.375,  0.875,  0.375),
                    (0.375,  0.375,  0.875),
                    (0.125,  0.125,  0.125),
                    (0.125,  0.625,  0.625),
                    (0.625,  0.125,  0.625),
                    (0.625,  0.625,  0.125)]

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])

structure.store()

# QE input parameters
qe_dict = {'CONTROL': {
              'calculation': 'scf',
              'restart_mode': 'from_scratch',
              'wf_collect': True,
          },
          'SYSTEM': {
              'ecutwfc': 30.,
              'ecutrho': 240.,
              },
          'ELECTRONS': {
              'conv_thr': 1.e-6,
              }}

pseudo_dict = {'family': 'pbe_test'}

# Monkhorst-pack
kpoints_dict = {'points': [2, 2, 2],
                'shift':  [0.0, 0.0, 0.0]}

# Cluster information
machine_dict = {
    'num_machines': 1,
    'parallel_env':'mpi*',
    'tot_num_mpiprocs': 16}

# Phonopy input parameters
phonopy_parameters = {'supercell': [[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 2]],
                     'primitive': [[0.0, 0.5, 0.5],
                                   [0.5, 0.0, 0.5],
                                   [0.5, 0.5, 0.0]],
                     'distance': 0.01,
                     'mesh': [40, 40, 40],
                     'symmetry_precision': 1e-4}

# Collect workflow input data
wf_parameters = {
     'structure': structure,
     'phonopy_input': {'parameters': phonopy_parameters},
     'input_force': {'code': 'pw@boston',
                    'parameters': qe_dict,
                    'resources': machine_dict,
                    'pseudo': pseudo_dict,
                    'kpoints': kpoints_dict},
     'input_optimize': {'code': 'pw@boston',
                       'parameters': qe_dict,
                       'resources': machine_dict,
                       'pseudo': pseudo_dict,
                       'kpoints': kpoints_dict},

}


#Submit workflow
WorkflowPhonon = WorkflowFactory('wf_phonon')
wf = WorkflowPhonon(params=wf_parameters, optimize=True)

wf.label = 'QE Si'
wf.start()
print ('pk: {}'.format(wf.pk))