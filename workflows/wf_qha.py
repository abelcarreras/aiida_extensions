from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow
from aiida.orm.calculation.inline import make_inline

from aiida.workflows.wf_gruneisen_pressure import WorkflowGruneisen
from aiida.workflows.wf_phonon import WorkflowPhonon

from aiida.orm import load_workflow

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')

import numpy as np


def check_dos_stable(freq, dos, tol=1e-6):

    mask_neg = np.ma.masked_less(freq, 0.0).mask
    mask_pos = np.ma.masked_greater(freq, 0.0).mask
    int_neg = -np.trapz(np.multiply(dos[mask_neg], freq[mask_neg]), x=freq[mask_neg])
    int_pos = np.trapz(np.multiply(dos[mask_pos], freq[mask_pos]), x=freq[mask_pos])

    if int_neg / int_pos > tol:
        return False
    else:
        return True


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
            self.append_to_report('Error no method defined')
            self.next(self.exit)


    # Generate the volume expanded cells
    @Workflow.step
    def pressure_expansions(self):
        self.append_to_report('Pressure expansion calculations')
        wf_parameters = self.get_parameters()
        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')
        prediction = self.get_step(self.start).get_sub_workflows()[0].get_result('thermal_expansion_prediction')
        stresses = prediction.get_array('stresses')

        test_pressures = [stresses[0], stresses[1]]  # in kbar

        total_range = test_pressures[1] - test_pressures[0]
        interval = total_range


        self.add_attribute('npoints', 5)

        self.add_attribute('test_range', test_pressures)
        self.add_attribute('total_range', total_range)
        self.add_attribute('max', None)
        self.add_attribute('min', None)
        self.add_attribute('interval',interval)


        for pressure in test_pressures:
            self.append_to_report('pressure: {}'.format(pressure))

            # Submit workflow
            wf = WorkflowPhonon(params=wf_parameters, pressure=pressure, optimize=True)
            wf.store()

            # wf = load_workflow(wfs_test[i])

            self.attach_workflow(wf)
            wf.start()
        self.next(self.collect_data)

    @Workflow.step
    def collect_data(self):

        wf_parameters = self.get_parameters()

        self.get_step_calculations(self.optimize).latest('id')

        n_points = wf_parameters['n_points']

        test_range = self.get_attribute('test_range')
        total_range = self.get_attribute('total_range')
        interval = self.get_attribute('interval')

        max = self.get_attribute('max')
        min = self.get_attribute('min')

        wf_max = None
        wf_min = None

        #wf_min, wf_max = list(self.get_step('pressure_expansions').get_sub_workflows())[-2:]
        for wf_test in self.get_step('pressure_expansions').get_sub_workflows():
            if wf_test.get_attribute('pressure') == test_range[0]:
                wf_min = wf_test
            if wf_test.get_attribute('pressure') == test_range[1]:
                wf_max = wf_test

        if wf_max is None or wf_min is None:
            self.append_to_report('Something wrong with volumes: {}'.format(test_range))
            self.next(self.exit)

        total_dos_min = wf_min.get_result('dos').get_array('total_dos')
        total_dos_max = wf_max.get_result('dos').get_array('total_dos')
        frequency_min = wf_min.get_result('dos').get_array('frequency')
        frequency_max = wf_max.get_result('dos').get_array('frequency')
        #pressure_min = wf_min.get_attribute('pressure')
        #pressure_max = wf_max.get_attribute('pressure')

        #a = check_dos_stable(frequency_min, total_dos_min, tol=1e-6)

        ok_inf = check_dos_stable(frequency_min, total_dos_min, tol=1e-6)
        ok_sup = check_dos_stable(frequency_max, total_dos_max, tol=1e-6)

        if not ok_sup:
            test_range[1] = test_range[0] + 0.5 * total_range
            interval = interval * 0.5

        if not ok_inf:
            test_range[0] = test_range[1] - 0.5 * total_range
            interval = interval * 0.5

        if ok_inf and ok_sup:
            if max is None or test_range[1] > max:
                max = test_range[1]

            if min is None or test_range[0] < min:
                min = test_range[0]

            if total_range / interval < n_points:
                # test_range[1] = test_range[0] + 3.0/2.0 * total_range
                test_range[1] = test_range[0] + np.ceil((1.5 * total_range) / interval) * interval

                # interval = interval * 3/2
            else:
                print 'max', max
                print 'min', min
                self.next(self.complete)
                return

        total_range = test_range[1] - test_range[0]

        self.add_attribute('total_range', total_range)
        self.add_attribute('max', max)
        self.add_attribute('min', min)
        self.add_attribute('interval', interval)

        test_pressures = [test_range[0], test_range[1]]  # in kbar


        # Remove duplicates
        for wf_test in self.get_step('pressure_expansions').get_sub_workflows():
            for pressure in test_pressures:

                if wf_test.get_attribute('pressure') == pressure:
                    test_pressures.remove(pressure)

        for pressure in test_pressures:
            self.append_to_report('pressure: {}'.format(pressure))

            # Submit workflow
            wf = WorkflowPhonon(params=wf_parameters, pressure=pressure, optimize=True)
            wf.store()

            # wf = load_workflow(wfs_test[i])

            self.attach_workflow(wf)
            wf.start()

        self.next(self.collect_data)

    @Workflow.step
    def complete(self):

        wf_parameters = self.get_parameters()

        self.get_step_calculations(self.optimize).latest('id')

        interval = self.get_attribute('interval')

        max = self.get_attribute('max')
        min = self.get_attribute('min')

        n_points = int((max - min) / interval) + 1

        test_pressures =  [min + interval * i for i in range(n_points)]

        # Remove duplicates
        for wf_test in self.get_step('pressure_expansions').get_sub_workflows():
            for pressure in test_pressures:

                if wf_test.get_attribute('pressure') == pressure:
                    test_pressures.remove(pressure)

        for pressure in test_pressures:
            self.append_to_report('pressure: {}'.format(pressure))

            # Submit workflow
            wf = WorkflowPhonon(params=wf_parameters, pressure=pressure, optimize=True)
            wf.store()

            # wf = load_workflow(wfs_test[i])

            self.attach_workflow(wf)
            wf.start()

        self.next(self.exit)

