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

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                  # 'pressure': 0.0,  # In Gruneisen workflow this is ignored. Pressure is set in workflow arguments
                  'vmax': 0.000001,  # Angstrom^3
                  'energy_tolerance': 1.0e-25,  # eV
                  'force_tolerance': 1.0e-25,  # eV angstrom
                  'max_evaluations': 1000000,
                  'max_iterations': 500000}

# Cluster information
machine_dict = {
    'num_machines': 1,
    'parallel_env':'mpi*',
    'tot_num_mpiprocs' : 16}


# Phonopy input parameters
phonopy_parameters = {'supercell': [[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 2]],
                     'primitive': [[0.0, 0.5, 0.5],
                                   [0.5, 0.0, 0.5],
                                   [0.5, 0.5, 0.0]],
                     'distance': 0.01,
                     'mesh': [40, 40, 40],
                     'symmetry_precision': 1e-5}

# Silicon(C) Tersoff
tersoff_si = {'Si  Si  Si ': '3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8'}

potential ={'pair_style': 'tersoff',
                          'data': tersoff_si}


# Collect workflow input data
wf_parameters = {
     'structure': structure,
     'phonopy_input': {'parameters': phonopy_parameters},
     'input_force': {'code': 'lammps_force@boston',
                      'potential': potential,
                      'resources': lammps_machine},
     'input_optimize': {'code': 'lammps_optimize@boston',
                         'potential': potential,
                         'parameters': parameters_opt,
                         'resources': lammps_machine},
    }

#Submit workflow
from aiida.workflows.wf_gruneisen_pressure import WorkflowGruneisen
wf = WorkflowGruneisen(params=wf_parameters, pre_optimize=True)  # pressure in kb

wf.label = 'Gruneisen Si lammps'
wf.start()
print ('pk: {}'.format(wf.pk))
