from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory, load_node, WorkflowFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np

# GaN [-37000 bar  <->  23000 bar]
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
phonopy_parameters = {'supercell': [[3, 0, 0],
                                    [0, 3, 0],
                                    [0, 0, 3]],
                     'primitive': [[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]],
                     'distance': 0.01,
                     'mesh': [40, 40, 40],
                     'symmetry_precision': 1e-5}

# GaN Tersoff
tersoff_gan = {'Ga Ga Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199',
               'N  N  N' : '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77',
               'Ga Ga N' : '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
               'Ga N  N' : '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
               'N  Ga Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
               'N  Ga N ': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000',
               'N  N  Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
               'Ga N  Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000'}


potential ={'pair_style': 'tersoff',
                          'data': tersoff_gan}

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
WorkflowGruneisen = WorkflowFactory('wf_gruneisen_pressure')
#from aiida.workflows.wf_gruneisen_pressure import WorkflowGruneisen
wf = WorkflowGruneisen(params=wf_parameters, pre_optimize=False)  # pressure in kb

wf.label = 'Gruneisen GaN'
wf.start()
print ('pk: {}'.format(wf.pk))
