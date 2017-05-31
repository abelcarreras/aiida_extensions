from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np

cell = [[ 3.759417, 0.000000, 0.000000],
        [-1.879709, 3.255751, 0.000000],
        [ 0.000000, 0.000000, 6.064977]]

symbols=['Na'] * 2
scaled_positions = [(0.33333,  0.66666,  0.25000),
                    (0.66667,  0.33333,  0.75000)]

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])

structure.store()

# LJ parameters extracted from :
# Bhansali, A. P., Bayazitoglu, Y., & Maruyama, S. (1999).
# Molecular dynamics simulation of an evaporating sodium droplet.
# International Journal of Thermal Sciences, 38(1), 66-74

potential ={'pair_style': 'lennard_jones',
            #                 epsilon,  sigma, cutoff
            'data': {'1  1':  '0.0202    3.24    5.0',
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
ph_dict = ParameterData(dict={'supercell': [[3,0,0],
                                            [0,3,0],
                                            [0,0,3]],
                              'primitive': [[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0]],
                              'distance': 0.01,
                              'mesh': [20, 20, 20]}
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
wf = WorkflowPhonon(params=wf_parameters, optimize=True, pressure=0.0)  # pressure in kb

wf.label = 'LAMMPS Lennad-Jones'
wf.start()
print ('pk: {}'.format(wf.pk))
