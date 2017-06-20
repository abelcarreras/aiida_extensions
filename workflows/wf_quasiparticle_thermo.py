from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow
from aiida.workflows.wf_phonon import WorkflowPhonon
from aiida.orm import load_node, load_workflow
from aiida.orm.calculation.inline import make_inline

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
import numpy as np


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

        if 'optimize' in kwargs:
            self._optimize = kwargs['optimize']
        else:
            self._optimize = True  # By default optimization is done


    def generate_md_dynaphopy(self, structure, parameters_md, parameters_dynaphopy, force_constants, temperature=None):

        if temperature is not None:
            parameters_md = dict(parameters_md)
            parameters_md['parameters']['temperature'] = temperature

        codename = parameters_md['code']
        code = Code.get_from_string(codename)

        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters_md['resources'])

        calc.label = "test lammps calculation"
        calc.description = "A much longer description"
        calc.use_code(code)
        calc.use_structure(structure)
        calc.use_potential(ParameterData(dict=parameters_md['potential']))
        calc.use_parameters(ParameterData(dict=parameters_md['parameters']))
        calc.use_force_constants(force_constants)
        calc.use_parameters_dynaphopy(ParameterData(dict=parameters_dynaphopy['parameters']))
        calc.use_supercell_md(ParameterData(dict={'shape': parameters_md['supercell']}))

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


        self.next(self.dynaphopy)

    # Generate the volume expanded cells
    @Workflow.step
    def dynaphopy(self):

        wf_parameters = self.get_parameters()

        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')
        self.add_result('final_structure', structure)

        harmonic_force_constants = self.get_step(self.start).get_sub_workflows()[0].get_result('force_constants')
        self.add_result('force_constants', harmonic_force_constants)

        for t in range(100, 200, 50):
            calc = self.generate_md_dynaphopy(structure,
                                              wf_parameters['input_md'],
                                              wf_parameters['dynaphopy_input'],
                                              harmonic_force_constants,
                                              temperature=t)
            calc.label = t
            self.append_to_report('created MD calculation with PK={} and temperature={}'.format(calc.pk, t))
            self.attach_calculation(calc)

        self.next(self.collect)

    # Collects the forces and prepares force constants
    @Workflow.step
    def collect(self):

        # Get the thermal properties at 0 K from phonopy calculation
        self.add_result('h_thermal_properties',  self.get_step('start').get_sub_workflows()[0].get_result('thermal_properties'))
        optimization_data = self.get_step(self.start).get_sub_workflows()[0].get_result('optimized_structure_data')
        self.add_result('optimized_structure_data', optimization_data)

        # Get the thermal properties at finite temperature from dynaphopy calculation
        for calc in self.get_step_calculations(self.dynaphopy):

            temperature = float(calc.label)
            self.add_result('thermal_properties_{}'.format(temperature), calc.out.thermal_properties)
            self.add_result('quasiparticle_data_{}'.format(temperature), calc.out.quasiparticle_data)
            self.add_result('r_force_constants_{}'.format(temperature), calc.out.array_data)

        self.next(self.exit)