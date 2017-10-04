from aiida.orm import Code, DataFactory, WorkflowFactory
from aiida.orm.workflow import Workflow
from aiida.orm.calculation.inline import make_inline

#from aiida.workflows.wf_phonon import WorkflowPhonon
from aiida.orm import load_node, load_workflow

import numpy as np

WorkflowPhonon = WorkflowFactory('wf_phonon')
StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')


def thermal_expansion(volumes, electronic_energies, gruneisen, stresses=None, t_max=1000, t_step=10):

    fit_ve = np.polyfit(volumes, electronic_energies, 2)

    test_volumes = np.arange(volumes[0] * 0.8, volumes[0] * 1.2, volumes[0] * 0.01)
    electronic_energies = np.array([np.polyval(fit_ve, i) for i in test_volumes])

    gruneisen.set_thermal_properties(test_volumes, t_min=0, t_max=t_max, t_step=t_step)
    tp = gruneisen.get_thermal_properties()

    gruneisen.get_phonon()
    normalize = gruneisen.get_phonon().unitcell.get_number_of_atoms() / gruneisen.get_phonon().primitive.get_number_of_atoms()

    free_energy_array = []
    cv_array = []
    entropy_array = []
    total_free_energy_array = []
    for energy, tpi in zip(electronic_energies, tp.get_thermal_properties()):
        temperatures, free_energy, entropy, cv = tpi.get_thermal_properties()
        free_energy_array.append(free_energy)
        entropy_array.append(entropy)
        cv_array.append(cv)
        total_free_energy_array.append(free_energy / normalize + energy)

    total_free_energy_array = np.array(total_free_energy_array)

    fit = np.polyfit(test_volumes, total_free_energy_array, 2)

    min_volume = []
    e_min = []
    for j, t in enumerate(temperatures):
        min_v = -fit.T[j][1] / (2 * fit.T[j][0])
        e_min.append(np.polyval(fit.T[j], min_v))
        min_volume.append(min_v)

    if stresses is not None:

        from scipy.optimize import curve_fit, OptimizeWarning

        try:
            # Fit to an exponential equation
            def fitting_function(x, a, b, c):
                return np.exp(-b * (x + a)) + c

            p_b = 0.1
            p_c = -200
            p_a = -np.log(-p_c) / p_b - volumes[0]

            popt, pcov = curve_fit(fitting_function, volumes, stresses, p0=[p_a, p_b, p_c], maxfev=100000)
            min_stress = fitting_function(min_volume, *popt)

        except OptimizeWarning:
            # Fit to a quadratic equation
            fit_vs = np.polyfit(volumes, stresses, 2)
            min_stress = np.array([np.polyval(fit_vs, v) for v in min_volume])
    else:
        min_stress = None

    return temperatures, min_volume, min_stress


def get_path_using_seekpath(structure, band_resolution=30):
    import seekpath

    cell = structure.cell
    positions = [site.position for site in structure.sites]
    scaled_positions = np.dot(positions, np.linalg.inv(cell))
    numbers = np.unique([site.kind_name for site in structure.sites], return_inverse=True)[1]
    structure2 = (cell, scaled_positions, numbers)
    path_data = seekpath.get_path(structure2)

    labels = path_data['point_coords']

    band_ranges = []
    for set in path_data['path']:
        band_ranges.append([labels[set[0]], labels[set[1]]])

    bands = []
    for q_start, q_end in band_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)

    return {'ranges': band_ranges,
            'labels': path_data['path']}


def get_phonon(structure, force_constants, phonopy_input):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'],
                     symprec=phonopy_input['symmetry_precision'])

    phonon.set_force_constants(force_constants)

    return phonon


def get_commensurate_points(structure, phonopy_input):

    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
    from phonopy import Phonopy

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'],
                     symprec=phonopy_input['symmetry_precision'])

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()

    dynmat2fc = DynmatToForceConstants(primitive, supercell)
    com_points = dynmat2fc.get_commensurate_points()

    return com_points


@make_inline
def phonopy_gruneisen_inline(**kwargs):
    from phonopy import PhonopyGruneisen

    phonopy_input = kwargs.pop('phonopy_input').get_dict()

    structure_origin = kwargs.pop('structure_origin')

    phonon_plus = get_phonon(kwargs.pop('structure_plus'),
                             kwargs.pop('force_constants_plus').get_array('force_constants'),
                             phonopy_input)

    phonon_minus = get_phonon(kwargs.pop('structure_minus'),
                              kwargs.pop('force_constants_minus').get_array('force_constants'),
                              phonopy_input)

    phonon_origin = get_phonon(structure_origin,
                               kwargs.pop('force_constants_origin').get_array('force_constants'),
                               phonopy_input)

    gruneisen = PhonopyGruneisen(phonon_origin,  # equilibrium
                                 phonon_plus,  # plus
                                 phonon_minus)  # minus

    gruneisen.set_mesh(phonopy_input['mesh'], is_gamma_center=False, is_mesh_symmetry=True)

    # Band structure
    bands = get_path_using_seekpath(structure_origin)

    gruneisen.set_band_structure(bands['ranges'], 51)
    band_structure_gruneisen = gruneisen.get_band_structure()._paths

    q_points = np.array([band[0] for band in band_structure_gruneisen])
    q_path = np.array([band[5] for band in band_structure_gruneisen])
    frequencies = np.array([band[4] for band in band_structure_gruneisen])
    gamma = np.array([band[2] for band in band_structure_gruneisen])
    distances = np.array([band[1] for band in band_structure_gruneisen])
    eigenvalues = np.array([band[3] for band in band_structure_gruneisen])
    band_labels = np.array(bands['labels'])

    # build band structure
    band_structure_array = ArrayData()
    band_structure_array.set_array('q_points', q_points)
    band_structure_array.set_array('q_path', q_path)
    band_structure_array.set_array('frequencies', frequencies)
    band_structure_array.set_array('gruneisen', gamma)
    band_structure_array.set_array('distances', distances)
    band_structure_array.set_array('eigenvalues', eigenvalues)
    band_structure_array.set_array('labels', band_labels)

    # mesh
    mesh = gruneisen.get_mesh()
    frequencies_mesh = np.array(mesh.get_frequencies())
    gruneisen_mesh = np.array(mesh.get_gruneisen())

    # build mesh
    mesh_array = ArrayData()
    mesh_array.set_array('frequencies', frequencies_mesh)
    mesh_array.set_array('gruneisen', gruneisen_mesh)


    # Thermal expansion approximate prediction
    volumes = np.array([phonon_origin.unitcell.get_volume(),
                        phonon_plus.unitcell.get_volume(),
                        phonon_minus.unitcell.get_volume()])

    energy_pressure = kwargs.pop('energy_pressure')
    energies = energy_pressure.get_array('energies')
    stresses = energy_pressure.get_array('stresses')

    temperatures, min_volumes, min_stresses = thermal_expansion(volumes,
                                                                energies,
                                                                gruneisen,
                                                                stresses=stresses,
                                                                t_max=1000,
                                                                t_step=5)
    # build mesh
    thermal_expansion_prediction = ArrayData()
    thermal_expansion_prediction.set_array('stresses', np.array(min_stresses))
    thermal_expansion_prediction.set_array('volumes', np.array(min_volumes))
    thermal_expansion_prediction.set_array('temperatures', np.array(temperatures))

    return {'band_structure': band_structure_array, 'mesh': mesh_array, 'thermal_expansion_prediction': thermal_expansion_prediction}


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


class Wf_gruneisen_pressureWorkflow(Workflow):
    def __init__(self, **kwargs):
        super(Wf_gruneisen_pressureWorkflow, self).__init__(**kwargs)
        if 'pre_optimize' in kwargs:
            self._pre_optimize = kwargs['pre_optimize']
        else:
            self._pre_optimize = True  # By default pre-optimization is done

        if 'include_born' in kwargs:
            self._include_born = kwargs['include_born']
        else:
            self._include_born = False  # By default not include born

        if 'pressure' in kwargs:
            self._pressure = kwargs['pressure']
        else:
            self._pressure = 0.0  # By default pre-optimization is done

        if 'p_displacement' in kwargs:
            self._p_displacement = kwargs['p_displacement']
        else:
            self._p_displacement = 2  # in Kbar




    # Calculates the reference crystal structure (optimize it if requested)
    @Workflow.step
    def start(self):
        self.append_to_report('Starting workflow_workflow')
        self.append_to_report('Phonon calculation of base structure')

        self.add_attribute('pressure', self._pressure)
        self.add_attribute('include_born', self._include_born)

        self.add_attribute('p_displacement', self._p_displacement)

        if not self._pre_optimize:
            self.next(self.pressure_expansions_direct)
            return

        wf_parameters = self.get_parameters()
        # self.append_to_report('crystal: ' + wf_parameters['structure'].get_formula())

        self.append_to_report('pressure grune: {}'.format(self._pressure))

        wf = WorkflowPhonon(params=wf_parameters,
                            optimize=True,
                            constant_volume=False,
                            pressure=self._pressure,
                            include_born=self._include_born)
        # wf = load_workflow(440)

        wf.store()
        self.attach_workflow(wf)
        wf.start()

        self.next(self.pressure_expansions)

    # Generate the volume expanded cells optimizing at different external pressures
    @Workflow.step
    def pressure_expansions(self):
        self.append_to_report('Pressure expansion calculations')
        wf_parameters = self.get_parameters()

        structure = self.get_step('start').get_sub_workflows()[0].get_result('final_structure')
        self.append_to_report('optimized structure volume: {}'.format(structure.pk))

        p_displacement = self.get_attribute('p_displacement')
        pressure_differences = [-p_displacement, p_displacement]
        for p in pressure_differences:
            pressure = self.get_attribute('pressure') + p

            self.append_to_report('pressure: {}'.format(pressure))

            wf = WorkflowPhonon(params=wf_parameters,
                                optimize=True,
                                pressure=pressure,
                                include_born=self.get_attribute('include_born'))
            # wf = load_workflow(list[i])

            wf.store()

            self.attach_workflow(wf)
            wf.start()

        self.add_attribute('pressure_differences', pressure_differences)

        self.next(self.collect_data)

    # Generate the volume expanded cells optimizing at constant volume
    @Workflow.step
    def pressure_expansions_direct(self):
        self.append_to_report('Pressure expansion direct calculations')
        wf_parameters = self.get_parameters()

        structure = wf_parameters['structure']
        self.append_to_report('structure volume: {}'.format(structure.pk))

        # list = [751, 752, 753]
        p_displacement = self.get_attribute('p_displacement')
        pressure_differences = [-p_displacement, 0, p_displacement]
        for i, p in enumerate(pressure_differences):
            pressure = self.get_attribute('pressure') + p

            self.append_to_report('pressure: {}'.format(pressure))

            wf = WorkflowPhonon(params=wf_parameters, optimize=True,
                                pressure=pressure,
                                include_born=self.get_attribute('include_born'))
            # wf = load_workflow(list[i])

            wf.store()

            self.attach_workflow(wf)
            wf.start()

        self.add_attribute('pressure_differences', pressure_differences)

        self.next(self.collect_data)

    # Collects the forces and prepares force constants
    @Workflow.step
    def collect_data(self):

        parameters_phonopy = self.get_parameters()['phonopy_input']

        if self.get_step('pressure_expansions') is not None:
            wf_origin = self.get_step('start').get_sub_workflows()[0]
            wf_plus, wf_minus = self.get_step('pressure_expansions').get_sub_workflows()
        else:
            wf_plus, wf_origin, wf_minus = self.get_step('pressure_expansions_direct').get_sub_workflows()

        self.append_to_report('WF_PLUS: {}'.format(wf_plus.pk))
        self.append_to_report('WF_MINUS: {}'.format(wf_minus.pk))
        self.append_to_report('WF_ORIGIN: {}'.format(wf_origin.pk))

        # ExpansionExpansion
        energies = [wf_origin.get_result('optimized_structure_data').dict.energy,
                    wf_plus.get_result('optimized_structure_data').dict.energy,
                    wf_minus.get_result('optimized_structure_data').dict.energy]

        pressures = [wf_origin.get_attribute('pressure'),
                     wf_plus.get_attribute('pressure'),
                     wf_minus.get_attribute('pressure')]

        vpe_array = ArrayData()
        vpe_array.set_array('energies', np.array(energies))
        vpe_array.set_array('stresses', np.array(pressures))
        vpe_array.store()

        self.append_to_report('reading structure')

        inline_params = {'structure_origin': wf_origin.get_result('final_structure'),
                         'structure_plus':   wf_plus.get_result('final_structure'),
                         'structure_minus':  wf_minus.get_result('final_structure'),
                         'force_constants_origin': wf_origin.get_result('force_constants'),
                         'force_constants_plus':   wf_plus.get_result('force_constants'),
                         'force_constants_minus':  wf_minus.get_result('force_constants'),
                         'energy_pressure': vpe_array,
                         'phonopy_input': ParameterData(dict=parameters_phonopy['parameters'])}

        # Do the phonopy Gruneisen parameters calculation
        results = phonopy_gruneisen_inline(**inline_params)[1]

        self.add_result('final_structure', wf_origin.get_result('final_structure'))
        self.add_result('optimized_structure_data', wf_origin.get_result('optimized_structure_data'))
        self.add_result('band_structure', results['band_structure'])
        self.add_result('mesh', results['mesh'])
        self.add_result('thermal_expansion_prediction', results['thermal_expansion_prediction'])

        self.append_to_report('Finishing Gruneisen workflow')

        self.next(self.exit)