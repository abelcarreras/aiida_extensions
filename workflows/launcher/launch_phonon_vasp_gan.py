from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

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

kpoints_dict = {'points' : [2, 2, 2],
                'shift'  : [0.0, 0.0, 0.0]}

machine_dict = { 
    'num_machines': 1,
    'parallel_env':'mpi*', 
    'tot_num_mpiprocs' : 16}


ph_dict = ParameterData(dict={'supercell': [[2,0,0],
                                            [0,2,0],
                                            [0,0,2]],
                              'primitive': [[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0]],
                              'distance': 0.01,
                              'mesh' : [20, 20, 20]}
                       ).store()

wf_parameters = {
     'structure': structure,
     'phonopy_input': ph_dict,
     'vasp_force': {'code': 'vasp541mpi@stern',
                    'parameters': incar_dict,
                    'resources': machine_dict,
                    'pseudo': pseudo_dict,
                    'kpoints': kpoints_dict},
     'vasp_optimize': {'code': 'vasp541mpi@stern',
                       'parameters': incar_dict,
                       'resources': machine_dict,
                       'pseudo': pseudo_dict,
                       'kpoints': kpoints_dict},

#    'vasp_input' : {'incar': incar_dict,
#                     'resources': machine_dict},
#     'pseudo' : pseudo_dict,
#     'kpoints' : kpoints_dict,
     'pre_optimize' : 3   # comment this line to skip structure optimization (This key contains the value of ISIF)
}


#Submit workflow
from aiida.workflows.wf_phonon_vasp import WorkflowPhonon
wf = WorkflowPhonon(params=wf_parameters)

wf.start()
print ('pk: {}'.format(wf.pk))

