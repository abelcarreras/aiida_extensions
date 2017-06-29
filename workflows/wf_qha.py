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
from phonopy import PhonopyQHA


def check_dos_stable(wf, tol=1e-6):

    try:
        dos = wf.get_result('dos').get_array('total_dos')
        freq = wf.get_result('dos').get_array('frequency')
    except ValueError:
        return False

    mask_neg = np.ma.masked_less(freq, 0.0).mask
    mask_pos = np.ma.masked_greater(freq, 0.0).mask

    if mask_neg.any() == False:
        return True

    if mask_pos.any() == False:
        return False

    int_neg = -np.trapz(np.multiply(dos[mask_neg], freq[mask_neg]), x=freq[mask_neg])
    int_pos = np.trapz(np.multiply(dos[mask_pos], freq[mask_pos]), x=freq[mask_pos])

    if int_neg / int_pos > tol:
        return False
    else:
        return True


@make_inline
def calculate_qha_inline(**kwargs):

    from phonopy import PhonopyQHA
    import numpy as np

#    thermal_properties_list = [key for key, value in kwargs.items() if 'thermal_properties' in key.lower()]
#    optimized_structure_data_list = [key for key, value in kwargs.items() if 'optimized_structure_data' in key.lower()]
    structure_list = [key for key, value in kwargs.items() if 'final_structure' in key.lower()]

    volumes = []
    electronic_energies = []
    fe_phonon = []
    entropy = []
    cv = []

    for i in range(len(structure_list)):
       # volumes.append(kwargs.pop(key).get_cell_volume())
       volumes.append(kwargs.pop('final_structure_{}'.format(i)).get_cell_volume())
       electronic_energies.append(kwargs.pop('optimized_structure_data_{}'.format(i)).dict.energy)
       thermal_properties = kwargs.pop('thermal_properties_{}'.format(i))
       temperatures = thermal_properties.get_array('temperature')
       fe_phonon.append(thermal_properties.get_array('free_energy'))
       entropy.append(thermal_properties.get_array('entropy'))
       cv.append(thermal_properties.get_array('cv'))

    sort_index = np.argsort(volumes)

    temperatures = np.array(temperatures)
    volumes = np.array(volumes)[sort_index]
    electronic_energies = np.array(electronic_energies)[sort_index]
    fe_phonon = np.array(fe_phonon).T[:, sort_index]
    entropy = np.array(entropy).T[:, sort_index]
    cv = np.array(cv).T[:, sort_index]



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

    # Get data
    qha_temperatures = phonopy_qha._qha._temperatures[:phonopy_qha._qha._max_t_index]
    helmholtz_volume = phonopy_qha.get_helmholtz_volume()
    thermal_expansion = phonopy_qha.get_thermal_expansion()
    volume_temperature = phonopy_qha.get_volume_temperature()
    heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
    volume_expansion = phonopy_qha.get_volume_expansion()
    gibbs_temperature = phonopy_qha.get_gibbs_temperature()


    qha_output = ArrayData()

    qha_output.set_array('temperature', np.array(qha_temperatures))
    qha_output.set_array('helmholtz_volume', np.array(helmholtz_volume))
    qha_output.set_array('thermal_expansion', np.array(thermal_expansion))
    qha_output.set_array('volume_temperature', np.array(volume_temperature))
    qha_output.set_array('heat_capacity_P_numerical', np.array(heat_capacity_P_numerical))
    qha_output.set_array('volume_expansion', np.array(volume_expansion))
    qha_output.set_array('gibbs_temperature', np.array(gibbs_temperature))


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

        if 'manual' in kwargs:
            self._manual = kwargs['manual']
        else:
            self._manual = False  # By default expansion method is pressure

        if 'only_grune' in kwargs:
            self._only_grune = kwargs['only_grune']
        else:
            self._only_grune = False  # By default expansion method is pressure


    # Calculates the reference crystal structure (optimize it if requested)
    @Workflow.step
    def start(self):
        self.append_to_report('Starting workflow_workflow')
        self.append_to_report('Phonon calculation of base structure')

        if self._manual:
            self.next(self.pressure_manual_expansions)
            return

        wf_parameters = self.get_parameters()
        # self.append_to_report('crystal: ' + wf_parameters['structure'].get_formula())

        wf = WorkflowGruneisen(params=wf_parameters, pre_optimize=True)
        wf.store()

        #wf = load_workflow(802)
        self.attach_workflow(wf)
        wf.start()

        if self._only_grune:
            self.next(self.pressure_gruneisen)
            return

        if self._expansion_method == 'pressure':
            self.next(self.pressure_expansions)
        elif self._expansion_method == 'volume':
            self.next(self.volume_expansions)
        else:
            self.append_to_report('Error no method defined')
            self.next(self.exit)

    # Direct manual stresses expanasions
    @Workflow.step
    def pressure_manual_expansions(self):
        self.append_to_report('Manual pressure expansion calculations')
        wf_parameters = self.get_parameters()

        test_pressures = wf_parameters['scan_pressures'] # in kbar

        # wfs_test = [821, 820]
        for i, pressure in enumerate(test_pressures):
            self.append_to_report('pressure: {}'.format(pressure))

            # Submit workflow
            wf = WorkflowPhonon(params=wf_parameters, pressure=pressure, optimize=True)
            wf.store()

            self.attach_workflow(wf)
            wf.start()
        self.next(self.qha_calculation_write_files)

    # Auto expansion just using Gruneisen prediction
    @Workflow.step
    def pressure_gruneisen(self):
        self.append_to_report('Trust Gruneisen expansion (For empirical potentials)')
        wf_parameters = self.get_parameters()

        prediction = self.get_step(self.start).get_sub_workflows()[0].get_result('thermal_expansion_prediction')
        stresses = prediction.get_array('stresses')

        n_points = wf_parameters['n_points']
        test_pressures = np.linspace(-1.0*np.max(stresses), np.max(stresses), n_points)  # in kbar

        # wfs_test = [821, 820]
        for i, pressure in enumerate(test_pressures):
            self.append_to_report('pressure: {}'.format(pressure))

            # Submit workflow
            wf = WorkflowPhonon(params=wf_parameters, pressure=pressure, optimize=True)
            wf.store()

            self.attach_workflow(wf)
            wf.start()
        self.next(self.qha_calculation_write_files)


    # Auto expansion by searching real DOS limits (hopping algorithm)
    @Workflow.step
    def pressure_expansions(self):
        self.append_to_report('Pressure expansion calculations')
        wf_parameters = self.get_parameters()
        # structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')
        prediction = self.get_step(self.start).get_sub_workflows()[0].get_result('thermal_expansion_prediction')
        stresses = prediction.get_array('stresses')

        test_pressures = [np.min([0.0, np.min(stresses)]), np.max([0.0, np.max(stresses)])]  # in kbar

        total_range = test_pressures[1] - test_pressures[0]
        interval = total_range/2.0

        self.add_attribute('npoints', 5)

        self.add_attribute('test_range', test_pressures)
        self.add_attribute('total_range', total_range)
        self.add_attribute('max', None)
        self.add_attribute('min', None)
        self.add_attribute('interval', interval)
        self.add_attribute('clock', 1)

        # wfs_test = [821, 820]
        for i, pressure in enumerate(test_pressures):
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

        self.append_to_report('--- collect step ------')

        wf_parameters = self.get_parameters()

        # self.get_step_calculations(self.optimize).latest('id')

        n_points = wf_parameters['n_points']

        test_range = self.get_attribute('test_range')
        total_range = self.get_attribute('total_range')
        interval = self.get_attribute('interval')
        clock = self.get_attribute('clock')

        max = self.get_attribute('max')
        min = self.get_attribute('min')

        wf_max = None
        wf_min = None

        self.append_to_report('test range {}'.format(test_range))
        self.append_to_report('interval {}'.format(interval))

        wf_complete_list = list(self.get_step('pressure_expansions').get_sub_workflows())
        if self.get_step('collect_data') is not None:
            wf_complete_list += list(self.get_step('collect_data').get_sub_workflows())

        #wf_min, wf_max = list(self.get_step('pressure_expansions').get_sub_workflows())[-2:]
        for wf_test in wf_complete_list:
            if wf_test.get_attribute('pressure') == test_range[0]:
                wf_min = wf_test
            if wf_test.get_attribute('pressure') == test_range[1]:
                wf_max = wf_test

        if wf_max is None or wf_min is None:
            self.append_to_report('Something wrong with volumes: {}'.format(test_range))
            self.next(self.exit)

        ok_inf = check_dos_stable(wf_min, tol=1e-6)
        ok_sup = check_dos_stable(wf_max, tol=1e-6)


        self.append_to_report('DOS stable | inf:{} sup:{}'.format(ok_inf, ok_sup))

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

            self.append_to_report('n_point estimation {}'.format(total_range / interval))
            if total_range / interval < n_points:
                # test_range[1] = test_range[0] + 3.0/2.0 * total_range
#                 test_range[1] = test_range[0] + np.ceil((1.5 * total_range) / interval) * interval
                if clock == 1:
                    test_range[1] = test_range[0] + np.ceil((1.5 * total_range) / interval) * interval
                else:
                    test_range[0] = test_range[1] - np.ceil((1.5 * total_range) / interval) * interval

                clock = -clock

                    # interval = interval * 3/2
            else:
                self.append_to_report('Exit: min {}, max {}'.format(min, max))

                self.next(self.complete)
                return

        total_range = test_range[1] - test_range[0]

        self.add_attribute('test_range', test_range)
        self.add_attribute('total_range', total_range)
        self.add_attribute('max', max)
        self.add_attribute('min', min)
        self.add_attribute('interval', interval)

        self.add_attribute('clock', clock)

        test_pressures = [test_range[0], test_range[1]]  # in kbar

        # Be efficient
        good = [wf_test.get_attribute('pressure') for wf_test in wf_complete_list
                if check_dos_stable(wf_test, tol=1e-6)]
        good = np.sort(good)

        self.append_to_report('GOOD pressure list {}'.format(good))

        if len(np.diff(good)) > 0:
            pressure_additional_list = np.arange(np.min(good), np.max(good), interval)
            self.append_to_report('GOOD additional list {}'.format(pressure_additional_list))
            test_pressures += pressure_additional_list.tolist()
            test_pressures = np.unique(test_pressures).tolist()

        self.append_to_report('pressure list {}'.format(test_pressures))

        # Remove duplicates
        for wf_test in wf_complete_list:
            for pressure in test_pressures:

                if wf_test.get_attribute('pressure') == pressure:
                    test_pressures.remove(pressure)

        self.append_to_report('pressure list (no duplicates){}'.format(test_pressures))

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

        # self.get_step_calculations(self.optimize).latest('id')

        interval = self.get_attribute('interval')

        max = self.get_attribute('max')
        min = self.get_attribute('min')

        n_points = int((max - min) / interval) + 1

        test_pressures = [min + interval * i for i in range(n_points)]

        self.append_to_report('final pressure list: {}'.format(test_pressures))

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

        self.next(self.qha_calculation_write_files)

    @Workflow.step
    def qha_calculation(self):

        interval = self.get_attribute('interval')

        max = self.get_attribute('max')
        min = self.get_attribute('min')

        n_points = int((max - min) / interval) + 1
        test_pressures = [min + interval * i for i in range(n_points)]

        # Remove duplicates
        wf_complete_list = list(self.get_step('pressure_expansions').get_sub_workflows())
        wf_complete_list += list(self.get_step('collect_data').get_sub_workflows())
        wf_complete_list += list(self.get_step('complete').get_sub_workflows())

        inline_params = {}

        for wf_test in wf_complete_list:
            for i, pressure in enumerate(test_pressures):
                if wf_test.get_attribute('pressure') == pressure:
                    thermal_properties = wf_test.get_result('thermal_properties')
                    optimized_data = wf_test.get_result('optimized_structure_data')
                    final_structure = wf_test.get_result('final_structure')

                    inline_params.update({'thermal_properties_{}'.format(i): thermal_properties})
                    inline_params.update({'optimized_structure_data_{}'.format(i): optimized_data})
                    inline_params.update({'final_structure_{}'.format(i): final_structure})

        qha_result = calculate_qha_inline(**inline_params)

        self.add_result('qha_output', qha_result)

        self.next(self.exit)


    @Workflow.step
    def qha_calculation_write_files(self):

        interval = self.get_attribute('interval')

        max = self.get_attribute('max')
        min = self.get_attribute('min')

        n_points = int((max - min) / interval) + 1
        test_pressures = [min + interval * i for i in range(n_points)]


        # Remove duplicates
        wf_complete_list = list(self.get_step('pressure_expansions').get_sub_workflows())
        wf_complete_list += list(self.get_step('collect_data').get_sub_workflows())
        wf_complete_list += list(self.get_step('complete').get_sub_workflows())

        volumes = []
        electronic_energies = []
        temperatures = []
        fe_phonon = []
        entropy = []
        cv = []

        for wf_test in wf_complete_list:
            for pressure in test_pressures:
                if wf_test.get_attribute('pressure') == pressure:
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

        # Get data
        qha_temperatures = phonopy_qha._qha._temperatures[:phonopy_qha._qha._max_t_index]
        helmholtz_volume = phonopy_qha.get_helmholtz_volume()
        thermal_expansion = phonopy_qha.get_thermal_expansion()
        volume_temperature = phonopy_qha.get_volume_temperature()
        heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
        volume_expansion = phonopy_qha.get_volume_expansion()
        gibbs_temperature = phonopy_qha.get_gibbs_temperature()
        bulk_modulus = phonopy_qha.get_bulk_modulus()


        def get_file_from_numpy_array(data):
            import StringIO
            output = StringIO.StringIO()
            for line in np.array(data).astype(str):
                output.write('       '.join(line) + '\n')
            output.seek(0)
            return output

        data_folder = self.current_folder.get_subfolder('DATA_FILES')
        data_folder.create()

        data_folder.create_file_from_filelike(get_file_from_numpy_array(zip(qha_temperatures, gibbs_temperature)),
                                              'gibbs_temperature')
        data_folder.create_file_from_filelike(get_file_from_numpy_array(zip(qha_temperatures, volume_expansion)),
                                              'volume_expansion')
        data_folder.create_file_from_filelike(get_file_from_numpy_array(zip(qha_temperatures, volume_temperature)),
                                              'volume_temperature')
        data_folder.create_file_from_filelike(get_file_from_numpy_array(zip(qha_temperatures, thermal_expansion)),
                                              'thermal_expansion')
        data_folder.create_file_from_filelike(get_file_from_numpy_array(zip(qha_temperatures, heat_capacity_P_numerical)),
                                              'heat_capacity_P_numerical')
        data_folder.create_file_from_filelike(get_file_from_numpy_array(zip(qha_temperatures, bulk_modulus)),
                                              'bulk_modulus')

        self.append_to_report('QHA properties calculated and written in files')

        self.next(self.exit)

