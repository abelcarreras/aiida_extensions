from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow

from aiida.orm import load_node
from aiida.orm.calculation.inline import make_inline

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')

import numpy as np


 # Create supercells with displacements to calculate forces
@make_inline
def create_supercells_with_displacements_inline(**kwargs):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()

 # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'])

    phonon.generate_displacements(distance=phonopy_input['distance'])

    cells_with_disp = phonon.get_supercells_with_displacements()

 # Transform cells to StructureData and set them ready to return
    retval = {}

    info = {"structures": []}

    for i, phonopy_supercell in enumerate(cells_with_disp):
        supercell = StructureData(cell=phonopy_supercell.get_cell())
        for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
                                    phonopy_supercell.get_positions()):
            supercell.append_atom(position=position, symbols=symbol)
        k = "structure_{}".format(i)
        retval[k] = supercell
        info["structures"].append(k)

  #  retval["info"] = info

    return retval


# Get forces from phonopy
@make_inline
def get_force_constants_inline(**kwargs):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()

 # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'])


 # Build data_sets from forces of supercells with displacments
    data_sets = phonon.get_displacement_dataset()
    for i, first_atoms in enumerate(data_sets['first_atoms']):
        force = kwargs.pop('force_{}'.format(i)).get_array('forces')[0]
        first_atoms['forces'] = np.array(force, dtype='double', order='c')

 # Calculate and get force constants
    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()

    force_constants = phonon.get_force_constants().tolist()

# Set force constants ready to return

    force_constants = phonon.get_force_constants()
    data = ArrayData()
    data.set_array('force_constants', force_constants)
    data.set_array('force_sets', np.array(data_sets))

    return {'phonopy_output': data}



# Get calculation from phonopy
@make_inline
def phonopy_calculation_inline(**kwargs):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')


 # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'])

    phonon.set_force_constants(force_constants)

    #Normalization factor primitive to unit cell
    normalization_factor = phonon.unitcell.get_number_of_atoms()/phonon.primitive.get_number_of_atoms()

    phonon.set_mesh(phonopy_input['mesh'], is_eigenvectors=True, is_mesh_symmetry=False)
    phonon.set_total_DOS()
    phonon.set_partial_DOS()

    # get DOS (normalized to unit cell)
    total_dos = phonon.get_total_DOS()*normalization_factor
    partial_dos = phonon.get_partial_DOS()*normalization_factor

    # Stores DOS data in DB as a workflow result
    dos = ArrayData()
    dos.set_array('frequency', total_dos[0])
    dos.set_array('total_dos', total_dos[1])
    dos.set_array('partial_dos', partial_dos[1])


    #THERMAL PROPERTIES (per primtive cell)
    phonon.set_thermal_properties()
    t, free_energy, entropy, cv = phonon.get_thermal_properties()

    # Stores thermal properties (per unit cell) data in DB as a workflow result
    thermal_properties = ArrayData()
    thermal_properties.set_array('temperature', t)
    thermal_properties.set_array('free_energy', free_energy*normalization_factor)
    thermal_properties.set_array('entropy', entropy*normalization_factor)
    thermal_properties.set_array('cv', cv*normalization_factor)

    return {'thermal_properties': thermal_properties, 'dos': dos}


class WorkflowPhonon(Workflow):

    def __init__(self, **kwargs):
        super(WorkflowPhonon, self).__init__(**kwargs)

    # Correct scaled coordinates (not in use now)
    def get_scaled_positions_lines(self, scaled_positions):

        for i, vec in enumerate(scaled_positions):
            for j, x in enumerate(vec):
                if x < 0.0:
                    scaled_positions[i][j] += 1.0
                if x >=1:
                    scaled_positions[i][j] -= 1.0
        return


    def generate_calculation_lammps(self, structure, parameters):

        codename = parameters['code']
        code = Code.get_from_string(codename)

        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters['resources'])


        calc.label = "test lammps calculation"
        calc.description = "A much longer description"
        calc.use_code(code)
        calc.use_structure(structure)
        calc.use_potential(ParameterData(dict=parameters['potential']))
        if code.get_input_plugin_name() == 'lammps.optimize':
            calc.use_parameters(ParameterData(dict=parameters['parameters']))
        calc.store_all()

        return calc

    # Starting workflow
    @Workflow.step
    def start(self):
        self.append_to_report('Workflow starting')

        self.next(self.optimize)

    # Prepare supercells with displacements

    @Workflow.step
    def optimize(self):
        self.append_to_report('Optimize')
        parameters = self.get_parameters()

        calc = self.generate_calculation_lammps(parameters['structure'], parameters['lammps_optimize'])
        calc.label = 'lammps optimization'
        self.append_to_report('created calculation with PK={}'.format(calc.pk))
  #      calc = load_node(11191)
        self.attach_calculation(calc)
        self.next(self.displacements)

    @Workflow.step
    def displacements(self):

        opt_calc = self.get_step_calculations(self.optimize)[0]
        self.append_to_report('Energy: {}'.format(opt_calc.res.energy))
        optimized_data = opt_calc.out.output_parameters
        self.add_result('optimized_structure_data', optimized_data)

        from phonopy.structure.atoms import Atoms as PhonopyAtoms
        from phonopy import Phonopy

        self.append_to_report('Displacements')

        parameters = self.get_parameters()

        self.append_to_report('From parameters')
       # structure = parameters['structure']
        structure = opt_calc.out.structure

        self.add_result('final_structure', structure)

        inline_params = {"structure": structure,
                         "phonopy_input": parameters['phonopy_input'],
                         }

        cells_with_disp = create_supercells_with_displacements_inline(**inline_params)[1]

  #      nodes = [ 762, 767, 772, 777]  #for debuging
        for key, cell in cells_with_disp.iteritems():
   #         calc = load_node(nodes[i])  #for debuging
            prefix = "structure_"
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            calc = self.generate_calculation_lammps(cell, parameters['lammps_force'])
            calc.label = 'force_{}'.format(suffix)

            self.append_to_report('created calculation with PK={}'.format(calc.pk))
            self.attach_calculation(calc)

        self.next(self.phonon_calculation)


    # Collects the forces and prepares force constants
    @Workflow.step
    def phonon_calculation(self):
        from phonopy.structure.atoms import Atoms as PhonopyAtoms
        from phonopy import Phonopy

        parameters = self.get_parameters()
        calcs = self.get_step_calculations(self.displacements)

        structure = self.get_result('final_structure')

        self.append_to_report('reading structure')


        inline_params = {'structure': structure,
                         'phonopy_input':parameters['phonopy_input']}

        self.append_to_report('created parameters')

        for calc in calcs:
            data = calc.get_outputs_dict()['array_data']
            inline_params[calc.label] = data
            self.append_to_report('extract force from {}'.format(calc.label))

        # Get the force constants and store it in DB as a Workflow result
        phonopy_data = get_force_constants_inline(**inline_params)[1]

        self.add_result('force_constants', phonopy_data['phonopy_output'])

        inline_params = {'structure': structure,
                         'phonopy_input': parameters['phonopy_input'],
                         'force_constants': phonopy_data['phonopy_output']}

        results = phonopy_calculation_inline(**inline_params)[1]

        self.add_result('thermal_properties', results['thermal_properties'])
        self.add_result('dos', results['dos'])

        self.next(self.exit)

