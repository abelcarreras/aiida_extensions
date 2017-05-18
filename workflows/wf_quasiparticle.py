from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow
from aiida.workflows.wf_phonon import WorkflowPhonon
from aiida.orm import load_node, load_workflow
from aiida.orm.calculation.inline import make_inline

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
import numpy as np


# @make_inline
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

        if 'use_optimized_structure_for_md' in kwargs:
            self._use_optimized_structure_for_md = kwargs['use_optimized_structure_for_md']
        else:
            self._use_optimized_structure_for_md = True  # By default optimized structure is used

        if 'optimize' in kwargs:
            self._optimize = kwargs['optimize']
        else:
            self._optimize = True  # By default optimization is done


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

        wf = WorkflowPhonon(params=wf_parameters, optimize=self._optimize)
        wf.store()

        # wf = load_workflow(127)
        self.attach_workflow(wf)
        wf.start()


        self.next(self.md_lammps)

    # Generate the volume expanded cells
    @Workflow.step
    def md_lammps(self):

        wf_parameters = self.get_parameters()

        if self._use_optimized_structure_for_md:
            structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')
        else:
            structure = wf_parameters['structure']

        inline_params = {'structure': structure,
                         'supercell': ParameterData(dict=wf_parameters['input_md'])}

        supercell = generate_supercell_inline(**inline_params)[1]['supercell']


        calc = self.generate_md_lammps(supercell, wf_parameters['input_md'])
#        self.append_to_report('created MD calculation with PK={}'.format(calc.pk))
        self.attach_calculation(calc)

        self.next(self.dynaphopy)

    # Collects the forces and prepares force constants
    @Workflow.step
    def dynaphopy(self):

        wf_parameters = self.get_parameters()

        harmonic_force_constants = self.get_step(self.start).get_sub_workflows()[0].get_result('force_constants')
        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        self.add_result('force_constants', harmonic_force_constants)

        md_calc = self.get_step_calculations(self.md_lammps)[0]

        dyna_calc = self.generate_calculation_dynaphopy(structure,
                                                        harmonic_force_constants,
                                                        wf_parameters['dynaphopy_input'],
                                                        md_calc.out.trajectory_data)

        self.attach_calculation(dyna_calc)
        self.next(self.collect)

    # Collects the forces and prepares force constants
    @Workflow.step
    def collect(self):

        # Get the thermal properties at 0 K from phonopy calculation
        self.add_result('h_thermal_properties',  self.get_step('start').get_sub_workflows()[0].get_result('thermal_properties'))

        # Get the thermal properties at finite temperature from dynaphopy calculation
        calc = self.get_step_calculations(self.dynaphopy)[0]

        self.add_result('thermal_properties', calc.out.thermal_properties)

        # Pass the final properties from phonon workflow
        self.add_result('quasiparticle_data', calc.out.quasiparticle_data)
        self.add_result('r_force_constants', calc.out.array_data)

        optimization_data = self.get_step(self.start).get_sub_workflows()[0].get_result('optimized_structure_data')
        final_structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        self.add_result('optimized_structure_data', optimization_data)
        self.add_result('final_structure', final_structure)

        self.next(self.exit)