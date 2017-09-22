from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory, WorkflowFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np


cell = [[ 3.987594, 0.000000, 0.000000],
        [-1.993797, 3.453358, 0.000000],
        [ 0.000000, 0.000000, 6.538394]]

symbols=['Ar'] * 2
scaled_positions = [(0.33333,  0.66666,  0.25000),
                    (0.66667,  0.33333,  0.75000)]

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])

structure.store()

# Example LJ parameters for Argon. These may not be accurate at all
potential ={'pair_style': 'lennard_jones',
            #                 epsilon,  sigma, cutoff
            'data': {'1  1':  '0.01029    3.4    5.0',
                     #'2  2':   '1.0      1.0    2.5',
                     #'1  2':   '1.0      1.0    2.5'
                     }}


# Lammps optimization parameters
parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                  # 'pressure': 0.0,  # In phonon workflow this is ignored. Pressure is set in workflow arguments
                  'vmax': 0.000001,  # Angstrom^3
                  'energy_tolerance': 1.0e-25,  # eV
                  'force_tolerance': 1.0e-25,  # eV angstrom
                  'max_evaluations': 1000000,
                  'max_iterations': 500000}

# Cluster resources
lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


# Phonopy input parameters
phonopy_parameters = {'supercell': [[3, 0, 0],
                                    [0, 3, 0],
                                    [0, 0, 3]],
                     'primitive': [[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]],
                     'distance': 0.01,
                     'mesh': [40, 40, 40],
                     'symmetry_precision': 1e-5}
# Cluster resources
phonopy_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}



# Collect workflow input data
wf_parameters = {
     'structure': structure,
     'phonopy_input': {'code': 'phonopy@stern',
                       'parameters': phonopy_parameters,
                       'resources': phonopy_machine},
     'input_force': {'code': 'lammps_force@boston',
                      'potential': potential,
                      'resources': lammps_machine},
     'input_optimize': {'code': 'lammps_optimize@boston',
                        'potential': potential,
                        'parameters': parameters_opt,
                        'resources': lammps_machine},
    }

#Submit workflow
WorkflowPhonon = WorkflowFactory('wf_phonon')
wf = WorkflowPhonon(params=wf_parameters, optimize=True, pressure=0.0)  # pressure in kb

wf.label = 'LAMMPS Lennad-Jones Ar'
wf.start()
print ('pk: {}'.format(wf.pk))





