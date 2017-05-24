from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

import numpy as np


StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

codename = 'lammps_md@boston'

############################
#  Define input parameters #
############################

a = 5.640772
cell = [[a, 0, 0],
        [0, a, 0],
        [0, 0, a]]

symbols=['Ar'] * 4
scaled_positions = [(0.000,  0.000,  0.000),
                    (0.000,  0.500,  0.500),
                    (0.500,  0.000,  0.500),
                    (0.500,  0.500,  0.000)]

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])

structure.store()


potential ={'pair_style': 'lennard_jones',
            #                 epsilon,  sigma, cutoff
            'data': {'1  1':  '0.01029   3.4    2.5',
                     #'2  2':   '1.0      1.0    2.5',
                     #'1  2':   '1.0      1.0    2.5'
                     }}

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


parameters_md = {'timestep': 0.001,
                 'temperature' : 60,
                 'thermostat_variable': 0.5,
                 'equilibrium_steps': 100,
                 'total_steps': 2000,
                 'dump_rate': 1}


code = Code.get_from_string(codename)

calc = code.new_calc(max_wallclock_seconds=3600,
                     resources=lammps_machine)

calc.label = "test lammps calculation"
calc.description = "A much longer description"
calc.use_code(code)
calc.use_structure(structure)
calc.use_potential(ParameterData(dict=potential))

calc.use_parameters(ParameterData(dict=parameters_md))


test_only = False

if test_only:  # It will not be submitted
    import os
    subfolder, script_filename = calc.submit_test()
    print "Test_submit for calculation (uuid='{}')".format(calc.uuid)
    print "Submit file in {}".format(os.path.join(
                                     os.path.relpath(subfolder.abspath),
                                     script_filename))
else:
    calc.store_all()
    print "created calculation; calc=Calculation(uuid='{}') # ID={}".format(
        calc.uuid, calc.dbnode.pk)
    calc.submit()
    print "submitted calculation; calc=Calculation(uuid='{}') # ID={}".format(
        calc.uuid, calc.dbnode.pk)
