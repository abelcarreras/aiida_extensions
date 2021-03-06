from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory, WorkflowFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np

# GaN
cell = [[ 3.1900000572, 0,           0],
        [-1.5950000286, 2.762621076, 0],
        [ 0.0,          0,           5.1890001297]]


scaled_positions=[(0.6666669,  0.3333334,  0.0000000),
                  (0.3333331,  0.6666663,  0.5000000),
                  (0.6666669,  0.3333334,  0.3750000),
                  (0.3333331,  0.6666663,  0.8750000)]

symbols=['Ga', 'Ga', 'N', 'N']

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(), 
                          symbols=symbols[i])

structure.store()


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
               'symbols': np.unique(symbols).tolist()}

# Monkhorst-pack
kpoints_dict = {'points' : [2, 2, 2],
                'shift'  : [0.0, 0.0, 0.0]}

machine_dict = { 
    'num_machines': 1,
    'parallel_env':'mpi*', 
    'tot_num_mpiprocs' : 16}


phonopy_parameters = {'supercell': [[3, 0, 0],
                                    [0, 3, 0],
                                    [0, 0, 3]],
                     'primitive': [[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]],
                     'distance': 0.01,
                     'mesh': [40, 40, 40],
                     'symmetry_precision': 1e-5}

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
                       'kpoints': kpoints_dict},

}

#Submit workflow
WorkflowPhonon = WorkflowFactory('wf_phonon')
wf = WorkflowPhonon(params=wf_parameters, optimize=False)

wf.label = 'VASP_GaN'
wf.start()
print ('pk: {}'.format(wf.pk))

