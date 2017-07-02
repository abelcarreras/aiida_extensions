======
LAMMPS
======

To Run a LAMMPS job it is necessay to include the following data:


- LAMMPS code
- Crystal Structure
- LAMMPS parameters dictionary
- LAMMPS potential
- Machine data dictionary


code contains an string that indicates the code and machine
Ex: codename=lammps@stern

Potential dictionary has the following structure:

potential = {'pair_style': ps
             'data': p_data

At this moment pair style can be
ps = 'Tersoff'  or  ps = 'Lennard-Jonnes'

data contains a dictionary with the data of the potential style
Tersoff needs data in the following format:

p_data = {'Si  Si  Si ': '3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8'}
* Following the same format as LAMMPS

Lennard-Jonnes needs data in the following format:

                     epsilon,  sigma, cutoff
p_data =  {'1  1':  '0.01029   3.4    2.5',

potential = ParameterData(dict=p_data)

- machine dictionary format

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


- Parameters is a AiiDA parameters type object that contains different data as a function of the type of calculation




Molecular Dynamics
------------------

Molecular dynamics requiere this output

parameters_md = {'timestep': 0.001,
                 'temperature' : 300,
                 'thermostat_variable': 0.5,
                 'equilibrium_steps': 100,
                 'total_steps': 2000,
                 'dump_rate': 1}

parameters = ParameterData(dict=parameters_md)


Optimization
------------

parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                  'pressure': 0.0,  # bars
                  'vmax': 0.000001,  # Angstrom^3
                  'energy_tolerance': 1.0e-25,  # eV
                  'force_tolerance': 1.0e-25,  # eV angstrom
                  'max_evaluations': 1000000,
                  'max_iterations': 500000}

parameters = ParameterData(dict=parameters_opt)



Forces
------

No parameters are necessary




Launching the calculation
_________________________


- Initialize calc object
code = Code.get_from_string(codename)

calc = code.new_calc(max_wallclock_seconds=3600,
                     resources=lammps_machine)

- Include data
calc.use_code(code)
calc.use_structure(structure)
calc.use_potential(potentials)
calc.use_parameters(parameters)

- Optional labeling
calc.label = "test lammps calculation"
calc.description = "A much longer description"

- submit the job
calc.store_all()
calc.submit()
