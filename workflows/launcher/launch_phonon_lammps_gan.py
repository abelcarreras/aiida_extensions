from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory, load_node

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np
def get_gan():
    # GaN [-37000 bar  <->  23000 bar]
    cell = [[ 3.1900000572, 0,           0],
            [-1.5950000286, 2.762621076, 0],
            [ 0.0,          0,           5.1890001297]]



    scaled_positions=[(0.6666669,  0.3333334,  0.0000000),
                      (0.3333331,  0.6666663,  0.5000000),
                      (0.6666669,  0.3333334,  0.3750000),
                      (0.3333331,  0.6666663,  0.8750000)]

    symbols=['Ga', 'Ga', 'N', 'N']
    return cell, scaled_positions, symbols

def get_si():
    # Si [-23000 bar  <->  4700 bar]  Tmax = 500
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

    return cell, scaled_positions, symbols


cell, scaled_positions, symbols = get_gan()

structure = StructureData(cell=cell)

positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])


structure.store()

ph_dict = {'supercell': [[3, 0, 0],
                                            [0, 3, 0],
                                            [0, 0, 3]],
                              'primitive': [[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0]],
                              'distance': 0.01,
                              'mesh': [40, 40, 40]}


dynaphopy_parameters ={'supercell': ph_dict['supercell'],
                       'primitive': ph_dict['primitive'],
                       'mesh': [40, 40, 40],
                       'md_commensurate': True}

# GaN Tersoff
tersoff_gan = {'Ga Ga Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199',
               'N  N  N' : '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77',
               'Ga Ga N' : '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
               'Ga N  N' : '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
               'N  Ga Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
               'N  Ga N ': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000',
               'N  N  Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
               'Ga N  Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000'}

# Silicon(C) Tersoff
tersoff_si = {'Si  Si  Si ': '3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8'}


potential ={'pair_style': 'tersoff',
                          'data': tersoff_gan}

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                  # 'pressure': 0.0,  # In phonon workflow this is ignored. Pressure is set in workflow arguments
                  'vmax': 0.000001,  # Angstrom^3
                  'energy_tolerance': 1.0e-25,  # eV
                  'force_tolerance': 1.0e-25,  # eV angstrom
                  'max_evaluations': 1000000,
                  'max_iterations': 500000}

wf_parameters = {
     'structure': structure,
     'phonopy_input': {'code': 'phonopy@stern',
                       'parameteres': ph_dict,
                       'resources': lammps_machine},
     'input_force': {'code': 'lammps_force@boston',
                      'potential': potential,
                      'resources': lammps_machine},
     'input_optimize': {'code': 'lammps_optimize@boston',
                         'potential': potential,
                         'parameters': parameters_opt,
                         'resources': lammps_machine},
    }

# Submit workflow
from aiida.workflows.wf_phonon import WorkflowPhonon
wf = WorkflowPhonon(params=wf_parameters,  pressure=0.0, optimize=True)  # pressure in kb

wf.label = 'lammps_GaN'
wf.start()
print ('pk: {}'.format(wf.pk))
