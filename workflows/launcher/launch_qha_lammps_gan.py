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




#for symbol, scaled_position in zip(symbols, scaled_positions):
#    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
#                          symbols=symbol)


structure.store()

ph_dict = ParameterData(dict={'supercell': [[3, 0, 0],
                                            [0, 3, 0],
                                            [0, 0, 3]],
                              'primitive': [[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0]],
                              'distance': 0.01,
                              'mesh': [40, 40, 40]}
                       ).store()


dynaphopy_parameters ={'supercell': ph_dict.dict.supercell,
                       'primitive': ph_dict.dict.primitive,
                       'mesh': [40, 40, 40],
                       'md_commensurate': True}


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

#codename = 'lammps_force@stern'
#code = Code.get_from_string(codename)
#calc = code.new_calc(max_wallclock_seconds=3600,
#                     resources={'num_machines': 1,
#                                'parallel_env': 'localmpi',
#                                'tot_num_mpiprocs': 6})

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}

parameters_md = {'timestep': 0.001,
                 'thermostat_variable': 0.5,
                 'equilibrium_steps': 100000,
                 'total_steps': 200000,
                 'dump_rate': 1}

parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                  'pressure': 0.0,  # bars
                  'vmax': 0.000001,  # Anstrom^3
                  'energy_tolerance': 1.0e-25,  # eV
                  'force_tolerance': 1.0e-25,  # eV angstrom
                  'max_evaluations': 1000000,
                  'max_iterations': 500000}

#structure = load_node(11233)

wf_parameters = {
     'structure': structure,
     'phonopy_input': ph_dict,
     'dynaphopy_input': {'code': 'dynaphopy@boston',
                         'parameters': dynaphopy_parameters,
                         'resources': lammps_machine,
                         'temperatures': [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200]},
     'lammps_force': {'code': 'lammps_force@boston',
                       'potential': potential,
                       'resources': lammps_machine},
     'lammps_optimize': {'code': 'lammps_optimize@boston',
                         'potential': potential,
                         'parameters': parameters_opt,
                         'resources': lammps_machine},
     'lammps_md': {'code': 'lammps_comb@boston',
                   'supercell': [3, 3, 3],
                   'potential': potential,
                   'parameters': parameters_md,
                   'resources': lammps_machine},
              #    'pressures': [-30000, -24000, -18000, -12000, -6000, 6000, 12000, 18000, 24000, 30000] # Si
                   'pressures': [-150000, -10000, -50000, 50000, 100000, 140000] # GaN
   #   'pressures': [-30000, -20000, -15000, -10000, -5000, 5000, 10000, 15000, 20000, 30000] # Si
    }

#Submit workflow
#from aiida.workflows.wf_phonon import WorkflowPhonon
#wf = WorkflowPhonon(params=wf_parameters)

#from aiida.workflows.wf_quasiparticle import WorkflowQuasiparticle
#wf = WorkflowQuasiparticle(params=wf_parameters)

from aiida.workflows.wf_qha import WorkflowQHA
wf = WorkflowQHA(params=wf_parameters)

wf.label = 'crystal'
wf.start()
print ('pk: {}'.format(wf.pk))
