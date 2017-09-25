from aiida.orm import Code, DataFactory, WorkflowFactory
from aiida.orm.workflow import Workflow
from aiida.orm.calculation.inline import make_inline

#from aiida.workflows.wf_quasiparticle_thermo import WorkflowQuasiparticle
# from aiida.orm import load_node, load_workflow

import numpy as np


WorkflowQuasiparticle = WorkflowFactory('wf_quasiparticle_thermo')

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')

@make_inline
def create_volumes_inline(**kwargs):
    import numpy as np
    initial_structure = kwargs['structure']
    volume_relations = kwargs['volumes'].get_dict()['relations']

    structures = {}
    for i, vol in enumerate(volume_relations):
        cell = np.array(initial_structure.cell) * vol
        structure = StructureData(cell=cell)
        for site in initial_structure.sites:
            structure.append_atom(position=np.array(site.position) * vol, symbols=site.kind_name)
        structures["structure_{}".format(i)] = structure

    return structures


class Wf_scan_quasiparticleWorkflow(Workflow):
    def __init__(self, **kwargs):
        super(Wf_scan_quasiparticleWorkflow, self).__init__(**kwargs)


    @Workflow.step
    def start(self):
        self.append_to_report('Starting scan workflow')

        self.next(self.pressure_expansions)

    # Generate the volume expanded cells
    @Workflow.step
    def volume_expansions(self):
        self.append_to_report('Volume expansion calculations')
        wf_parameters = self.get_parameters()

        inline_params = {'structure': wf_parameters['structure'],
                         'volumes': ParameterData(dict={ 'relations': [1.01, 1.0, 0.99]})}  # plus, minus

        cells = create_volumes_inline(**inline_params)[1]

        for i, structures in enumerate(cells.iterkeys()):
            structure_vol = cells['structure_{}'.format(i)]
            self.append_to_report('structure_{}: {}'.format(i, structure_vol.pk))
            wf_parameters_volume = dict(wf_parameters)
            wf_parameters_volume['structure'] = structure_vol

            # Submit workflow

            wf = WorkflowQuasiparticle(params=wf_parameters_volume, optimize=False)
            # wf = load_workflow(list[i])

            wf.store()

            self.attach_workflow(wf)
            wf.start()

        self.next(self.collect_data)


    @Workflow.step
    def pressure_expansions(self):
        self.append_to_report('Volume expansion calculations')
        wf_parameters = self.get_parameters()

        # pressures = [150, 100, 50, 0, -50, -90]
        pressures = wf_parameters['scan_pressures']
        for p in pressures:

            # Submit workflow

            wf = WorkflowQuasiparticle(params=wf_parameters, optimize=True, pressure=p)
            # wf = load_workflow(list[i])

            wf.store()

            self.attach_workflow(wf)
            wf.start()

        self.next(self.collect_data)


    # Collects the forces and prepares force constants
    @Workflow.step
    def collect_data(self):

        # Remove duplicates
        if self.get_step('pressure_expansions') is not None:
            wf_complete_list = list(self.get_step('pressure_expansions').get_sub_workflows())

        if self.get_step('volume_expansions') is not None:
            wf_complete_list = list(self.get_step('volume_expansions').get_sub_workflows())

        volumes = []
        electronic_energies = []
        temperatures = []
        fe_phonon = []
        entropy = []
        cv = []

        for wf_test in wf_complete_list:
            thermal_properties = wf_test.get_result('thermal_properties')
            optimized_data = wf_test.get_result('optimized_structure_data')
            final_structure = wf_test.get_result('final_structure')

            electronic_energies.append(optimized_data.dict.energy)
            volumes.append(final_structure.get_cell_volume())
            temperatures = thermal_properties.get_array('temperature')
            fe_phonon.append(thermal_properties.get_array('free_energy'))
            entropy.append(thermal_properties.get_array('entropy'))
            cv.append(thermal_properties.get_array('cv'))

        sort_index = np.argsort(volumes)

        volumes = np.array(volumes)[sort_index]
        electronic_energies = np.array(electronic_energies)[sort_index]
        temperatures = np.array(temperatures)
        fe_phonon = np.array(fe_phonon).T[:, sort_index]
        entropy = np.array(entropy).T[:, sort_index]
        cv = np.array(cv).T[:, sort_index]


        thermal_properties = ArrayData()
        thermal_properties.set_array('volumes', volumes)
        thermal_properties.set_array('electronic_energies', electronic_energies)

        thermal_properties.set_array('temperature', temperatures)
        thermal_properties.set_array('free_energy', fe_phonon)
        thermal_properties.set_array('entropy', entropy)
        thermal_properties.set_array('cv', cv)
        thermal_properties.store()

        self.add_result('thermal_properties', thermal_properties)

        self.append_to_report('Finishing Scan Quasiparticle')

        self.next(self.exit)