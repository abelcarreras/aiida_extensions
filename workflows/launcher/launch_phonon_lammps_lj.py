from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np


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


potential ={'pair_style': 'lennard_jones',  # epsilon, sigma, cutoff
                          'data': [[1, 1,      1.0,     1.0,   2.5],
                                   [2, 2,      1.0,     1.0,   2.5],
                                   [1, 2,      1.0,     1.0,   2.5]]}

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                  'pressure': 0.0,  # bars
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
     'input_force': {'code': 'lammps_force@boston',
                      'potential': potential,
                      'resources': lammps_machine},
     'input_optimize': {'code': 'lammps_optimize@boston',
                         'potential': potential,
                         'parameters': parameters_opt,
                         'resources': lammps_machine},
    }


#Submit workflow
from aiida.workflows.wf_phonon import WorkflowPhonon
wf = WorkflowPhonon(params=wf_parameters, optimize=True)

wf.label = 'LAMMPS Si'
wf.start()
print ('pk: {}'.format(wf.pk))





