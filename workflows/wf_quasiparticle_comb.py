# Under development

from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow
from aiida.workflows.wf_phonon import WorkflowPhonon
from aiida.orm import load_node, load_workflow
from aiida.orm.calculation.inline import make_inline

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
import numpy as np

#@make_inline
def generate_supercell2_inline(**kwargs):

    structure = kwargs.pop('structure')
    supercell = StructureData(cell=structure.cell)

    for site in structure.sites:
        supercell.append_atom(position=site.position,
                              symbols=site.kind_name)

    return {"supercell": supercell}


@make_inline
def generate_supercell_inline(**kwargs):
    import itertools

    structure = kwargs.pop('structure')
    supercell = kwargs.pop('supercell').dict.supercell

    symbols = [site.kind_name for site in structure.sites]
    positions=np.array([site.position for site in structure.sites])

    position_super_cell = []
    for k in range(positions.shape[0]):
        for r in itertools.product(*[range(i) for i in supercell[::-1]]):
            position_super_cell.append(positions[k,:] + np.dot(np.array(r[::-1]), structure.cell))
    position_super_cell = np.array(position_super_cell)

    symbol_super_cell = []
    for j in range(positions.shape[0]):
        symbol_super_cell += [symbols[j]] * np.prod(supercell)

    supercell = StructureData(cell=np.dot(structure.cell, np.diag(supercell)))

    for i, position in enumerate(position_super_cell):
        supercell.append_atom(position=position.tolist(),
                              symbols=symbol_super_cell[i])

    return {"supercell": supercell}


class WorkflowQuasiparticle(Workflow):
    def __init__(self, **kwargs):
        super(WorkflowQuasiparticle, self).__init__(**kwargs)

    def generate_md_combinate(self, supercell, structure, parameters_md, parameters_dyna, force_constants):

        codename = parameters_md['code']
        code = Code.get_from_string(codename)

        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters_md['resources'])


        calc.label = "md lammps calculation"
        calc.description = "A much longer description"
        calc.use_code(code)
        calc.use_structure(structure)
        calc.use_supercell(supercell)
        calc.use_parameters(ParameterData(dict=parameters_md['parameters']))
        calc.use_parameters_dynaphopy(ParameterData(dict=parameters_dyna['parameters']))
        calc.use_force_constants(force_constants)
        calc.use_potential(ParameterData(dict=parameters_md['potential']))
        calc.store_all()

        return calc

    def generate_md_lammps(self, structure, parameters):

        codename = parameters['code']
        code = Code.get_from_string(codename)

        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters['resources'])

        calc.label = "md lammps calculation"
        calc.description = "A much longer description"
        calc.use_code(code)
        calc.use_structure(structure)
        calc.use_parameters(ParameterData(dict=parameters['parameters']))
        calc.use_potential(ParameterData(dict=parameters['potential']))
        calc.store_all()

        return calc

    def generate_calculation_dynaphopy(self, structure, force_constants, parameters, trajectory):

        codename = parameters['code']
        code = Code.get_from_string(codename)
        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters['resources'])
      #  calc.label = "dynaphopy calculation"
      #  calc.description = "A much longer description"
        calc.use_code(code)

        calc.use_structure(structure)
        calc.use_parameters(ParameterData(dict=parameters['parameters']))
        calc.use_force_constants(force_constants)
        calc.use_trajectory(trajectory)
        calc.store_all()

        return calc


    # Calculates the reference crystal structure (optimize it if requested)
    @Workflow.step
    def start(self):
        self.append_to_report('Starting workflow_workflow')
        self.append_to_report('Phonon calculation of base structure')

        wf_parameters = self.get_parameters()

        wf = WorkflowPhonon(params=wf_parameters)
        wf.store()

   #     wf = load_workflow(127)
        self.attach_workflow(wf)
        wf.start()

        md_code = Code.get_from_string(wf_parameters['lammps_md']['code'])
        if md_code.get_input_plugin_name() == 'lammps.combinate':
            self.next(self.md_combinate)
        else:
            self.next(self.md_lammps)

    # Generate the volume expanded cells
    @Workflow.step
    def md_lammps(self):
        self.append_to_report('Temperatures expansion calculations')

        wf_parameters = self.get_parameters()
        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        temperatures = np.array(wf_parameters['dynaphopy_input']['temperatures'])
       # temperatures = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])
        inline_params = {'structure': structure,
                         'supercell': ParameterData(dict=wf_parameters['lammps_md'])}

        supercell = generate_supercell_inline(**inline_params)[1]['supercell']

  #      nodes = [11504, 11507, 11510, 11513, 11516]
        for i, temperature in enumerate(temperatures):
            wf_parameters_md = dict(wf_parameters['lammps_md'])
            wf_parameters_md['parameters']['temperature'] = temperature

            calc = self.generate_md_lammps(supercell, wf_parameters_md)
            calc.label = 'temperature_{}'.format(temperature)
        #    calc = load_node(nodes[i])
            self.append_to_report('created MD calculation with PK={}'.format(calc.pk))
            self.attach_calculation(calc)

        self.next(self.dynaphopy)

    # Collects the forces and prepares force constants
    @Workflow.step
    def dynaphopy(self):

        wf_parameters = self.get_parameters()

        harmonic_force_constants = self.get_step(self.start).get_sub_workflows()[0].get_result('force_constants')
        harmonic_dos = self.get_step(self.start).get_sub_workflows()[0].get_result('dos')
        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        self.add_result('force_constants', harmonic_force_constants)
        self.add_result('dos', harmonic_dos)

        calcs = self.get_step_calculations(self.md_lammps)

     #   nodes = [11578, 11580, 11582, 11584, 11586]
        for i, calc in enumerate(calcs):
            trajectory = calc.out.trajectory_data
            dynaphopy_input = dict(wf_parameters['dynaphopy_input'])
            dynaphopy_input['parameters']['temperature'] = calc.inp.parameters.dict.temperature
            dyna_calc = self.generate_calculation_dynaphopy(structure,
                                                            harmonic_force_constants,
                                                            dynaphopy_input,
                                                            trajectory)
            dyna_calc.label = calc.label
       #     dyna_calc = load_node(nodes[i])

            self.append_to_report('created QP calculation with PK={}'.format(dyna_calc.pk))
            self.attach_calculation(dyna_calc)

        self.next(self.collect)


    @Workflow.step
    def md_combinate(self):
        self.append_to_report('Temperatures expansion calculations')

        wf_parameters = self.get_parameters()
        harmonic_force_constants = self.get_step(self.start).get_sub_workflows()[0].get_result('force_constants')
        harmonic_dos = self.get_step(self.start).get_sub_workflows()[0].get_result('dos')
        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        self.add_result('force_constants', harmonic_force_constants)
        self.add_result('dos', harmonic_dos)

        temperatures = np.array(wf_parameters['dynaphopy_input']['temperatures'])
       # temperatures = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])
        inline_params = {'structure': structure,
                         'supercell': ParameterData(dict=wf_parameters['input_md'])}

        supercell = generate_supercell_inline(**inline_params)[1]['supercell']
  #      nodes = [11504, 11507, 11510, 11513, 11516]
        for i, temperature in enumerate(temperatures):
            wf_parameters_md = dict(wf_parameters['input_md'])
            wf_parameters_md['parameters']['temperature'] = temperature
            dynaphopy_input = dict(wf_parameters['dynaphopy_input'])
            dynaphopy_input['parameters']['temperature'] = temperature
            calc = self.generate_md_combinate(supercell, structure, wf_parameters_md, dynaphopy_input, harmonic_force_constants)
            calc.label = 'temperature_{}'.format(temperature)
        #    calc = load_node(nodes[i])
            self.append_to_report('created MD calculation with PK={}'.format(calc.pk))
            self.attach_calculation(calc)

        self.next(self.collect)

    # Collects the forces and prepares force constants
    @Workflow.step
    def collect(self):

        free_energy = []
        entropy = []
        temperature = []
        cv = []

        # get the phonon for 0 K
        temperature = [0]
        free_energy = [self.get_step('start').get_sub_workflows()[0].get_result('thermal_properties').get_array('free_energy')[0]]
        entropy = [self.get_step('start').get_sub_workflows()[0].get_result('thermal_properties').get_array('entropy')[0]]
        cv = [self.get_step('start').get_sub_workflows()[0].get_result('thermal_properties').get_array('cv')[0]]

        # get temperature dependent properties from dynaphopy
        wf_parameters = self.get_parameters()
        md_code = Code.get_from_string(wf_parameters['input_md']['code'])
        if md_code.get_input_plugin_name() == 'lammps.combinate':
            calcs = self.get_step_calculations(self.md_combinate)
        else:
            calcs = self.get_step_calculations(self.dynaphopy)

        for calc in calcs:
            thermal_properties = calc.out.thermal_properties

            temperature.append(thermal_properties.dict.temperature)
            entropy.append(thermal_properties.dict.entropy)
            free_energy.append(thermal_properties.dict.free_energy)
            cv.append(thermal_properties.dict.cv)


        order = np.argsort(temperature)
        array_data = ArrayData()
        array_data.set_array('temperature', np.array(temperature)[order])
        array_data.set_array('free_energy', np.array(free_energy)[order])
        array_data.set_array('entropy',  np.array(entropy)[order])
        array_data.set_array('cv', np.array(cv)[order])
        array_data.store()

        self.add_result('thermal_properties', array_data)

        # Pass the final properties from phonon workflow
        optimized_data = self.get_step(self.start).get_sub_workflows()[0].get_result('optimized_structure_data')
        final_structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')
        self.add_result('optimized_structure_data', optimized_data)
        self.add_result('final_structure', final_structure)

        self.next(self.exit)