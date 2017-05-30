
from aiida import load_dbenv
from aiida.orm import Code, DataFactory
from aiida.orm.calculation.inline import make_inline

from aiida.orm.workflow import Workflow


from aiida.workflows.wf_phonon import WorkflowPhonon
from aiida.workflows.wf_gruneisen import WorkflowGruneisen

from aiida.orm import load_workflow

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')

import numpy as np

@make_inline
def calculate_qha_inline(**kwargs):
    from phonopy import PhonopyQHA
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    import numpy as np

    #   structures = kwargs.pop('structures')
    #   optimized_data = kwargs.pop('optimized_data')
    #   thermodyamic_properties = kwargs.pop('thermodyamic_properties')

    entropy = []
    cv = []
    #  volumes = []
    fe_phonon = []
    temperatures = None

    #   structures = [key for key, value in kwargs.items() if 'structure' in key.lower()]
    #   for key in structures:
    #       volumes.append(kwargs.pop(key).get_cell_volume())

    volumes = [value.get_cell_volume() for key, value in kwargs.items() if 'structure' in key.lower()]
    electronic_energies = [value.get_dict()['energy'] for key, value in kwargs.items() if
                           'optimized_data' in key.lower()]

    thermal_properties_list = [key for key, value in kwargs.items() if 'thermal_properties' in key.lower()]

    for key in thermal_properties_list:
        thermal_properties = kwargs[key]
        fe_phonon.append(thermal_properties.get_array('free_energy'))
        entropy.append(thermal_properties.get_array('entropy'))
        cv.append(thermal_properties.get_array('cv'))
        temperatures = thermal_properties.get_array('temperature')

    # Arrange data sorted by volume and transform them to numpy array
    sort_index = np.argsort(volumes)

    volumes = np.array(volumes)[sort_index]
    electronic_energies = np.array(electronic_energies)[sort_index]
    temperatures = np.array(temperatures)
    fe_phonon = np.array(fe_phonon).T[:, sort_index]
    entropy = np.array(entropy).T[:, sort_index]
    cv = np.array(cv).T[:, sort_index]

    opt = np.argmin(electronic_energies)

    # Check minimum energy volume is within the data
    if np.ma.masked_less_equal(volumes, volumes[opt]).mask.all():
        print ('higher volume structures are necessary to compute')
        exit()
    if np.ma.masked_greater_equal(volumes, volumes[opt]).mask.all():
        print ('Lower volume structures are necessary to compute')
        exit()

        #   print volumes.shape
        #   print electronic_energies.shape
        #   print temperatures.shape
        #   print fe_phonon.shape
        #   print cv.shape
        #   print entropy.shape

    qha_output = ArrayData()
    qha_output.set_array('volumes', volumes)
    qha_output.set_array('electronic_energies', electronic_energies)
    qha_output.set_array('temperatures', temperatures)
    qha_output.set_array('fe_phonon', fe_phonon)
    qha_output.set_array('cv', cv)
    qha_output.set_array('entropy', entropy)

    # Calculate QHA
    phonopy_qha = PhonopyQHA(np.array(volumes),
                             np.array(electronic_energies),
                             eos="vinet",
                             temperatures=np.array(temperatures),
                             free_energy=np.array(fe_phonon),
                             cv=np.array(cv),
                             entropy=np.array(entropy),
                             #                         t_max=options.t_max,
                             verbose=False)

    #   print phonopy_qha.get_gibbs_temperature()

    qha_output = ArrayData()
    qha_output.set_array('volumes', volumes)
    qha_output.set_array('electronic_energies', electronic_energies)
    qha_output.set_array('temperatures', temperatures)
    qha_output.set_array('fe_phonon', fe_phonon)
    qha_output.set_array('cv', cv)
    qha_output.set_array('entropy', entropy)

    return {'qha_output': qha_output}

    # Get data
    helmholtz_volume = phonopy_qha.get_helmholtz_volume()
    thermal_expansion = phonopy_qha.get_thermal_expansion()
    volume_temperature = phonopy_qha.get_volume_temperature()
    heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
    volume_expansion = phonopy_qha.get_volume_expansion()
    gibbs_temperature = phonopy_qha.get_gibbs_temperature()

    qha_output = ArrayData()
    #    qha_output.set_array('temperature', temperatures)
    #    qha_output.set_array('helmholtz_volume', np.array(helmholtz_volume))
    #    qha_output.set_array('thermal_expansion', np.array(thermal_expansion))
    #    qha_output.set_array('volume_temperature', np.array(volume_temperature))
    #    qha_output.set_array('heat_capacity_P_numerical', np.array(heat_capacity_P_numerical))
    #    qha_output.set_array('volume_expansion', np.array(volume_expansion))
    #    qha_output.set_array('gibbs_temperature', np.array(gibbs_temperature))
    #   qha_output.store()

    return {'qha_output': qha_output}


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


class WorkflowQHA(Workflow):
    def __init__(self, **kwargs):
        super(WorkflowQHA, self).__init__(**kwargs)
        if 'expansion_method' in kwargs:
            self._expansion_method = kwargs['expansion_method']
        else:
            self._expansion_method = 'pressure'  # By default expansion method is pressure

    # Calculates the reference crystal structure (optimize it if requested)
    @Workflow.step
    def start(self):
        self.append_to_report('Starting workflow_workflow')
        self.append_to_report('Phonon calculation of base structure')

        wf_parameters = self.get_parameters()
        # self.append_to_report('crystal: ' + wf_parameters['structure'].get_formula())


        wf = WorkflowGruneisen(params=wf_parameters, pre_optimize=True)
        wf.store()

        #        wf = load_workflow(30)
        self.attach_workflow(wf)
        wf.start()

        if self._expansion_method == 'pressure':
            self.next(self.pressure_expansions)
        elif self._expansion_method == 'volume':
            self.next(self.volume_expansions)
        else:
            self.next(self.exit)


    # Generate the volume expanded cells
    @Workflow.step
    def pressure_expansions(self):
        self.append_to_report('Pressure expansion calculations')
        wf_parameters = self.get_parameters()
        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        test_pressures = [-10, -5, 5, 10]  # in kbar

        for pressure in test_pressures:
            self.append_to_report('pressure: {}'.format(pressure))

            # Submit workflow
            wf = WorkflowGruneisen(params=wf_parameters, pressure=pressure, optimize=True)
            wf.store()

            # wf = load_workflow(wfs_test[i])

            self.attach_workflow(wf)
            wf.start()

        self.next(self.collect_data)

    @Workflow.step
    def volume_expansions(self):
        self.append_to_report('Volume expansion calculations')
        wf_parameters = self.get_parameters()
        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        inline_params = {'structure': structure,
                         'volumes': ParameterData(dict={'relations': [0.9, 0.95, 1.02, 1.05] })}

        cells = create_volumes_inline(**inline_params)[1]

        #        wfs_test = [150, 152, 154, 156]
        for i, structures in enumerate(cells.iterkeys()):
            structure2 = cells['structure_{}'.format(i)]
            self.append_to_report('structure_{}: {} / {}'.format(i, structure2.pk, structure2.get_cell_volume()))
            wf_param_vol = dict(wf_parameters)
            wf_param_vol['structure'] = structure

            # Submit workflow
            wf = WorkflowGruneisen(params=wf_param_vol, optimize=True, constant_volume=True)
            wf.store()

            #           wf = load_workflow(wfs_test[i])

            self.attach_workflow(wf)
            wf.start()

        self.next(self.collect_data)

    # Collects the forces and prepares force constants
    @Workflow.step
    def collect_data(self):

        electronic_energies = []
        volumes = []
        # Get the calculations from workflow
        from itertools import chain

        wf_list = list(chain(self.get_steps()[1].get_sub_workflows(),
                       self.get_step(self.start).get_sub_workflows()))

        # wf_list = self.get_steps()[1].get_sub_workflows()
        for wf in wf_list:
            energy = wf.get_result('optimized_structure_data').get_dict()['energy']
            volume = wf.get_result('final_structure').get_cell_volume()
            self.append_to_report('{} {}'.format(volume, energy))
            electronic_energies.append(energy)
            volumes.append(volume)

        data = ArrayData()
        data.set_array('energy', np.array(electronic_energies))
        data.set_array('volume', np.array(volumes))
        data.store()
        self.add_result('data', data)

        self.append_to_report('Finishing workflow_workflow')
        self.next(self.exit)
