from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow
from aiida.orm.calculation.job.vasp import vasp as vplugin

from aiida.orm import load_node
from aiida.orm.calculation.inline import make_inline

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
KpointsData = DataFactory('array.kpoints')

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
    disp_cells = {}

    for i, phonopy_supercell in enumerate(cells_with_disp):
        supercell = StructureData(cell=phonopy_supercell.get_cell())
        for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
                                    phonopy_supercell.get_positions()):
            supercell.append_atom(position=position, symbols=symbol)        
        disp_cells["structure_{}".format(i)] = supercell
        
    return disp_cells


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
        # force = kwargs.pop('force_{}'.format(i)).get_array('forces')
        # first_atoms['forces'] = np.array(force, dtype='double', order='c')
        first_atoms['forces'] = kwargs.pop('force_{}'.format(i)).get_array('forces')[0]

 # Calculate and get force constants    
    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()
    
    force_constants = phonon.get_force_constants().tolist()

# Set force constants ready to return

    force_constants = phonon.get_force_constants()
    data = ArrayData()
    data.set_array('force_constants', force_constants)
    data.set_array('force_sets', np.array(data_sets))

    return {'phonopy_output' : data}




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
    dos.set_array('frequency',total_dos[0])
    dos.set_array('total_dos',total_dos[1])
    dos.set_array('partial_dos',partial_dos[1])


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

    def generate_calculation_vasp(self, structure, input_params, pseudo, kpoints, codename, resources, type='energy'):
        import pymatgen as mg
        from pymatgen.io import vasp as vaspio

        ParameterData = DataFactory('parameter')

        code = Code.get_from_string(codename)

        # Set calculation
        calc = code.new_calc(
                    max_wallclock_seconds=3600,
                    resources=resources
        )
        calc.set_withmpi(True)
        calc.label = 'VASP'

        #POSCAR
        calc.use_structure(structure)
#        poscar_param = {'comment': 'Generated ', 
#                        '@module': 'pymatgen.io.vasp.inputs', 
#                        '@class': 'Poscar', 
#                        'true_names': True}
#
#        calc.use_poscar(ParameterData(dict=poscar_param))

        #INCAR
        incar = vaspio.Incar(input_params)
        calc.use_incar(ParameterData(dict=incar.as_dict()))

        #KPOINTS
        if not 'style' in kpoints:
            kpoints['style'] = 'Monkhorst'
        # kpoints = vaspio.Kpoints.monkhorst_automatic(
        #     kpts=kpoints['points'], shift=kpoints['shift']
        # )
        # supported_modes = Enum(("Gamma", "Monkhorst", "Automatic", "Line_mode", "Cartesian", "Reciprocal"))
        kpoints = vaspio.Kpoints(comment='aiida generated',
                                 style=kpoints['style'],
                                 kpts=(kpoints['points'],), kpts_shift=kpoints['shift'])


        calc.use_kpoints(ParameterData(dict=kpoints.as_dict()))

        #POTCAR
        potcar = vaspio.Potcar(symbols=pseudo['symbols'], 
                               functional=pseudo['functional'])
        calc.use_potcar(ParameterData(dict=potcar.as_dict()))

        # Parser settings
        settings = {'PARSER_INSTRUCTIONS': []}
        pinstr = settings['PARSER_INSTRUCTIONS']

        pinstr.append({
            'instr': 'array_data_parser',
            'type': 'data',
            'params': {}
        })
        pinstr.append({
            'instr': 'output_parameters',
            'type': 'data',
            'params': {}
        })

        # additional files to return
        settings.setdefault(
            'ADDITIONAL_RETRIEVE_LIST', [
                'vasprun.xml',
                ]
        )

        calc.use_settings(ParameterData(dict=settings))

        calc.store_all()

        return calc

    # Starting workflow  
    @Workflow.step
    def start(self):
        self.append_to_report('Workflow starting')

        parameters = self.get_parameters()

        if 'pre_optimize' in parameters:
            self.add_attribute('counter', 10) # define max number of optimization iterations
            self.next(self.optimize)
        else:
            self.next(self.displacements)

    # Optimize the structure
    @Workflow.step
    def optimize(self):

        parameters = self.get_parameters()
        vasp_input = parameters['vasp_optimize']['parameters']
        tolerance = 1e-04

        counter = self.get_attribute('counter')

        optimized = self.get_step_calculations(self.optimize)
        if len(optimized):
            last_calc = self.get_step_calculations(self.optimize).latest('id')
            # last_calc = calc[len(calc)-1]
            structure = last_calc.get_outputs_dict()['structure']
            # last_calc = self.get_step_calculations(self.optimize).latest('id')
            forces = last_calc.out.output_array.get_array('forces')
            not_converged_forces = len(np.where(abs(forces) > tolerance)[0])
            self.append_to_report('Not converged forces: '.format(not_converged_forces))
            if not_converged_forces == 0:
                self.next(self.displacements)
        else:
            structure = parameters['structure']

        self.append_to_report('Optimize structure {}/{}'.format(len(optimized)+1,len(optimized)+counter+1))

        # Prepare vasp input for atomic forces calculation
        vasp_input_optimize = dict(vasp_input)
        vasp_input_optimize.update({
            'PREC'   : 'Normal',
            'ISTART' : 0,
            'IBRION' : 2,
            'ISIF'   : parameters['pre_optimize'],
            'NSW'    : 100,
            'LWAVE'  : '.FALSE.',
            'LCHARG' : '.FALSE.',
            'EDIFF'  : 1e-04,
            'EDIFFG' : -0.01,
            'ADDGRID': '.TRUE.',
            'LREAL'  : '.FALSE.'})

        if counter < 5:
            self.append_to_report('Last optimization: use conjugated gradient')
            vasp_input_optimize.update({'IBRION': 1})

        calc = self.generate_calculation_vasp(structure,
                                              vasp_input_optimize,
                                              parameters['vasp_optimize']['pseudo'],
                                              parameters['vasp_optimize']['kpoints'],
                                              parameters['vasp_optimize']['code'],
                                              parameters['vasp_optimize']['resources'])

        calc.label = 'optimization'
        print 'created calculation with PK={}'.format(calc.pk)
        self.attach_calculation(calc)


        if counter < 1:
            self.next(self.displacements)
        else:        
            self.add_attribute('counter', counter - 1)
            self.next(self.optimize)


    # Prepare supercells wuith displacements 
    @Workflow.step
    def displacements(self):

        from phonopy.structure.atoms import Atoms as PhonopyAtoms
        from phonopy import Phonopy

        self.append_to_report('Displacements')

        parameters = self.get_parameters()

        optimized = self.get_step(self.optimize)
        if not isinstance(optimized, type(None)):
            self.append_to_report('Optimized structure')
            # case where we launched calculations
            opt_calc = self.get_step_calculations(self.optimize).latest('id')
     #       last_calc = last_calc[len(calcs)-1]  #Last optimization
            structure = opt_calc.get_outputs_dict()['structure']
            #data = last_calc.get_outputs_dict()['vasp_parameters@Custom_vasprun_parserInstruction']
            #self.add_result('optimized_structure_data', data)
            optimized_data = opt_calc.out.output_parameters
            self.add_result('optimized_structure_data', optimized_data)

        else:
            self.append_to_report('From parameters')
            structure = parameters['structure']

        self.add_result('final_structure', structure)
#        self.add_attribute('structure', structure)

        inline_params = {"structure": structure,
                         "phonopy_input":parameters['phonopy_input'],
                         } 
        
        cells_with_disp = create_supercells_with_displacements_inline(**inline_params)[1]
  
        vasp_input = parameters['vasp_force']['parameters']

        #Prepare vasp input for atomic forces calculation
        vasp_input_forces = dict(vasp_input)
        vasp_input_forces.update({
            'PREC'   : 'Accurate',
            'ISTART' : 0,
            'IBRION' : -1,
            'NSW'    : 1,
            'LWAVE'  : '.FALSE.',
            'LCHARG' : '.FALSE.',
            'EDIFF'  : 1e-08,
            'ADDGRID': '.TRUE.',
            'LREAL'  : '.FALSE.'})

  #      nodes = [ 762, 767, 772, 777]  #for debuging
        for i, cell in enumerate(cells_with_disp.iterkeys()):
   #         calc = load_node(nodes[i])  #for debuging
            calc = self.generate_calculation_vasp(cells_with_disp['structure_{}'.format(i)],
                                                  vasp_input_forces,
                                                  parameters['vasp_force']['pseudo'],
                                                  parameters['vasp_force']['kpoints'],
                                                  parameters['vasp_force']['code'],
                                                  parameters['vasp_force']['resources'])
            calc.label = 'force_{}'.format(i)
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
#        structure = self.get_attribute('structure')

        structure = self.get_result('final_structure')



#        structure = parameters['structure']

        self.append_to_report('reading structure')


        inline_params = {'structure': structure,
                         'phonopy_input':parameters['phonopy_input']}

        self.append_to_report('created parameters')

        for calc in calcs:
            data = calc.get_outputs_dict()['output_array']
            inline_params[calc.label] = data
            self.append_to_report('extract force from {}'.format(calc.label))

 #       for calc in calcs:
 #           data = calc.get_outputs_dict()['vasp_parameters@Custom_vasprun_parserInstruction']
 #           inline_params[calc.label] = data
 #           self.append_to_report('extract force from {}'.format(calc.label))


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


