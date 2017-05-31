from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

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

# VASP input parameters
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


# Cluster information
machine_dict = {
    'num_machines': 1,
    'parallel_env':'mpi*',
    'tot_num_mpiprocs' : 16}


# Phonopy input parameters
ph_dict = ParameterData(dict={'supercell': [[2,0,0],
                                            [0,2,0],
                                            [0,0,2]],
                              'primitive': [[0.0, 0.5, 0.5],
                                            [0.5, 0.0, 0.5],
                                            [0.5, 0.5, 0.0]],
                              'distance': 0.01,
                              'mesh' : [20, 20, 20]}
                       ).store()

# Collect workflow input data
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
                       'kpoints': kpoints_dict},
}


#Submit workflow
from aiida.workflows.wf_gruneisen import WorkflowGruneisen
wf = WorkflowGruneisen(params=wf_parameters, constant_volume=False, pre_optimize=False)

wf.label = 'Gruneisen VASP Si '
wf.start()
print ('pk: {}'.format(wf.pk))