from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow
from aiida.orm.calculation.inline import make_inline

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
KpointsData = DataFactory('array.kpoints')

import numpy as np


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

    bands =[]
    for q_start, q_end in band_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)

    return {'ranges': bands,
            'labels': path_data['path']}


def get_born_parameters(phonon, born_charges, epsilon, symprec=1e-5):
    from phonopy.structure.cells import get_primitive, get_supercell
    from phonopy.structure.symmetry import Symmetry
    from phonopy.interface import get_default_physical_units

    print ('inside born parameters')
    pmat = phonon.get_primitive_matrix()
    smat = phonon.get_supercell_matrix()
    ucell = phonon.get_unitcell()

    print pmat
    print smat
    print ucell

    num_atom = len(born_charges)
    assert num_atom == ucell.get_number_of_atoms(), \
        "num_atom %d != len(borns) %d" % (ucell.get_number_of_atoms(),
                                          len(born_charges))

    inv_smat = np.linalg.inv(smat)
    scell = get_supercell(ucell, smat, symprec=symprec)
    pcell = get_primitive(scell, np.dot(inv_smat, pmat), symprec=symprec)
    p2s = np.array(pcell.get_primitive_to_supercell_map(), dtype='intc')
    p_sym = Symmetry(pcell, is_symmetry=True, symprec=symprec)
    s_indep_atoms = p2s[p_sym.get_independent_atoms()]
    u2u = scell.get_unitcell_to_unitcell_map()
    u_indep_atoms = [u2u[x] for x in s_indep_atoms]
    reduced_borns = born_charges[u_indep_atoms].copy()

    factor = get_default_physical_units('vasp')['nac_factor']  # born charges in VASP units

    born_dict = {'born': reduced_borns, 'dielectric': epsilon, 'factor': factor}

    print ('final born dict', born_dict)

    return born_dict


@make_inline
def standardize_cell_inline(**kwargs):
    import spglib
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    import itertools

    structure = kwargs.pop('structure')

    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    structure_data = (structure.cell,
                      bulk.get_scaled_positions(),
                      bulk.get_atomic_numbers())

    #lattice, refined_positions, numbers = spglib.refine_cell(structure_data, symprec=1e-5)
    lattice, standardized_positions, numbers = spglib.standardize_cell(structure_data,
                                                                       symprec=1e-5,
                                                                       to_primitive=0,
                                                                       no_idealize=1)

    print lattice, standardized_positions, numbers
    print [site.kind_name for site in structure.sites]
    standardized_bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                                     scaled_positions=standardized_positions,
                                     cell=lattice)

    # create new aiida structure object
    standarized = StructureData(cell=lattice)
    for position, symbol in zip(standardized_bulk.get_positions(), bulk.get_chemical_symbols()):
        standarized.append_atom(position=position,
                                      symbols=symbol)

    return {'standardized_structure': standarized}


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
                     primitive_matrix=phonopy_input['primitive'],
                     symprec=phonopy_input['symmetry_precision'])

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


# Get force constants from phonopy
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

    print bulk
    print
    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     symprec=phonopy_input['symmetry_precision'])

    phonon.generate_displacements(distance=phonopy_input['distance'])

    # Build data_sets from forces of supercells with displacments
    data_sets = phonon.get_displacement_dataset()

    for i, first_atoms in enumerate(data_sets['first_atoms']):
        first_atoms['forces'] = kwargs.pop('force_{}'.format(i)).get_array('forces')[-1]


    # Calculate and get force constants
    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()

    # force_constants = phonon.get_force_constants().tolist()
    force_constants = phonon.get_force_constants()

    # Set force sets and force constants array to return
    data = ArrayData()
    data.set_array('force_sets', np.array(data_sets))
    data.set_array('force_constants', force_constants)

    return {'phonopy_output': data}


@make_inline
def get_force_sets_inline(**kwargs):
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
                     symprec=phonopy_input['symmetry_precision'])

    phonon.generate_displacements(distance=phonopy_input['distance'])

    # Build data_sets from forces of supercells with displacments
    data_sets = phonon.get_displacement_dataset()
    for i, first_atoms in enumerate(data_sets['first_atoms']):
        first_atoms['forces'] = kwargs.pop('force_{}'.format(i)).get_array('forces')[-1]

    data = ArrayData()
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
    bands = get_path_using_seekpath(structure)

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     symprec=phonopy_input['symmetry_precision'])

    phonon.set_force_constants(force_constants)

    try:
        print ('trying born')
        nac_data = kwargs.pop('nac_data')
        born = nac_data.get_array('born_charges')
        epsilon = nac_data.get_array('epsilon')

        phonon.set_nac_params(get_born_parameters(phonon, born, epsilon))
        print ('born succeed')
    except:
        pass

    # Normalization factor primitive to unit cell
    norm_primitive_to_unitcell = phonon.unitcell.get_number_of_atoms() / phonon.primitive.get_number_of_atoms()

    phonon.set_band_structure(bands['ranges'])

    phonon.set_mesh(phonopy_input['mesh'], is_eigenvectors=True, is_mesh_symmetry=False)
    phonon.set_total_DOS(tetrahedron_method=True)
    phonon.set_partial_DOS(tetrahedron_method=True)

    # get band structure
    band_structure_phonopy = phonon.get_band_structure()
    q_points = np.array(band_structure_phonopy[0])
    q_path = np.array(band_structure_phonopy[1])
    frequencies = np.array(band_structure_phonopy[2])
    band_labels = np.array(bands['labels'])

    # stores band structure
    band_structure = ArrayData()
    band_structure.set_array('q_points', q_points)
    band_structure.set_array('q_path', q_path)
    band_structure.set_array('frequencies', frequencies)
    band_structure.set_array('labels', band_labels)

    # get DOS (normalized to unit cell)
    total_dos = phonon.get_total_DOS() * norm_primitive_to_unitcell
    partial_dos = phonon.get_partial_DOS() * norm_primitive_to_unitcell

    # Stores DOS data in DB as a workflow result
    dos = ArrayData()
    dos.set_array('frequency', total_dos[0])
    dos.set_array('total_dos', total_dos[1] * norm_primitive_to_unitcell)
    dos.set_array('partial_dos', partial_dos[1] * norm_primitive_to_unitcell)
    dos.set_array('partial_symbols', np.array(phonon.primitive.symbols))

    # THERMAL PROPERTIES (per primtive cell)
    phonon.set_thermal_properties()
    t, free_energy, entropy, cv = phonon.get_thermal_properties()

    # Stores thermal properties (per mol) data in DB as a workflow result
    thermal_properties = ArrayData()
    thermal_properties.set_array('temperature', t)
    thermal_properties.set_array('free_energy', free_energy * norm_primitive_to_unitcell)
    thermal_properties.set_array('entropy', entropy * norm_primitive_to_unitcell)
    thermal_properties.set_array('cv', cv * norm_primitive_to_unitcell)

    return {'thermal_properties': thermal_properties, 'dos': dos, 'band_structure': band_structure}


class Wf_phononWorkflow(Workflow):
#class WorkflowPhonon(Workflow):
    def __init__(self, **kwargs):
        super(Wf_phononWorkflow, self).__init__(**kwargs)
        #super(Wf_phononWorkflow, **kwargs).__init__()

        if 'optimize' in kwargs:
            self._optimize = kwargs['optimize']
        else:
            self._optimize = True  # By default optimization is done

        if 'constant_volume' in kwargs:
            self._constant_volume = kwargs['constant_volume']
        else:
            self._constant_volume = False  # By default constant pressure optimization is done

        if 'pressure' in kwargs:
            self._pressure = kwargs['pressure']
        else:
            self._pressure = 0.0  # By default pre-optimization is done

    # Correct scaled coordinates (not in use now)
    def get_scaled_positions_lines(self, scaled_positions):

        for i, vec in enumerate(scaled_positions):
            for j, x in enumerate(vec):
                if x < 0.0:
                    scaled_positions[i][j] += 1.0
                if x >= 1:
                    scaled_positions[i][j] -= 1.0
        return

    def generate_calculation_lammps(self, structure, parameters, type='optimize', pressure=0.0):

        if type == 'born_charges':
            return None

        codename = parameters['code']
        code = Code.get_from_string(codename)

        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters['resources'])

        calc.label = "test lammps calculation"
        calc.description = "A much longer description"
        calc.use_code(code)
        calc.use_structure(structure)
        calc.use_potential(ParameterData(dict=parameters['potential']))

        #if code.get_input_plugin_name() == 'lammps.optimize':
        if type == 'optimize':
            lammps_parameters = dict(parameters['parameters'])
            lammps_parameters.update({'pressure': pressure})  # pressure kb
            calc.use_parameters(ParameterData(dict=lammps_parameters))

        calc.store_all()

        return calc

    def generate_calculation_phonopy(self, structure, parameters, data_sets):

        code = Code.get_from_string(parameters['code'])

        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters['resources'])

        calc.use_code(code)
        calc.use_structure(structure)
        calc.use_parameters(ParameterData(dict=parameters['parameters']))
        calc.use_data_sets(data_sets)

        calc.store_all()

        return calc

    def generate_calculation_qe(self, structure, parameters, type='optimize', pressure=0.0):
        # On development
        code = Code.get_from_string(parameters['code'])

        calc = code.new_calc(max_wallclock_seconds=3600,
                             resources=parameters['resources'])

        parameters_qe = dict(parameters['parameters'])

        if type == 'optimize':
            parameters_qe['CONTROL'].update({'calculation': 'vc-relax'})
            parameters_qe['CELL'] = {'press': pressure,
                                     'press_conv_thr': 1.e-2,
                                     'cell_dynamics': 'bfgs',  # Quasi-Newton algorithm
                                     'cell_dofree': 'all'}     # Degrees of movement
            parameters_qe['IONS'] = {'ion_dynamics': 'bfgs'}



        parameters_qe['CONTROL'].update({'tstress': True,
                                         'tprnfor': True,
                                         'etot_conv_thr': 1.e-8,
                                         'forc_conv_thr': 1.e-6})


        calc.use_structure(structure)
        calc.use_code(code)
        calc.use_parameters(ParameterData(dict=parameters_qe))

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh(parameters['kpoints']['points'])
        calc.use_kpoints(kpoints)

        if 'family' in parameters['pseudo']:
            calc.use_pseudos_from_family(parameters['pseudo']['family'])

        calc.store_all()

        return calc



    def generate_calculation_vasp(self, structure, parameters, type='optimize', pressure=0.0):
        # import pymatgen as mg
        from pymatgen.io import vasp as vaspio

        ParameterData = DataFactory('parameter')

        code = Code.get_from_string(parameters['code'])

        # Set calculation
        calc = code.new_calc(
            max_wallclock_seconds=3600,
            resources=parameters['resources']
        )
        calc.set_withmpi(True)
        calc.label = 'VASP'

        # POSCAR
        calc.use_structure(structure)

        # INCAR
        incar = parameters['parameters']

        if type == 'optimize':
            vasp_input_optimize = dict(incar)
            vasp_input_optimize.update({
                'PREC': 'Accurate',
                'ISTART': 0,
                'IBRION': 2,
                'ISIF': 3,
                'NSW': 100,
                'LWAVE': '.FALSE.',
                'LCHARG': '.FALSE.',
                'EDIFF': 1e-08,
                'EDIFFG': -1e-08,
                'ADDGRID': '.TRUE.',
                'LREAL': '.FALSE.',
                'PSTRESS': pressure}) # unit: kb -> kB
            incar = vasp_input_optimize

        if type == 'optimize_constant_volume':
            vasp_input_optimize = dict(incar)
            vasp_input_optimize.update({
                'PREC': 'Accurate',
                'ISTART': 0,
                'IBRION': 2,
                'ISIF': 4,
                'NSW': 100,
                'LWAVE': '.FALSE.',
                'LCHARG': '.FALSE.',
                'EDIFF': 1e-08,
                'EDIFFG': -1e-08,
                'ADDGRID': '.TRUE.',
                'LREAL': '.FALSE.'})
            incar = vasp_input_optimize

        if type == 'forces':
            vasp_input_forces = dict(incar)
            vasp_input_forces.update({
                'PREC': 'Accurate',
                'ISYM': 0,
                'ISTART': 0,
                'IBRION': -1,
                'NSW': 1,
                'LWAVE': '.FALSE.',
                'LCHARG': '.FALSE.',
                'EDIFF': 1e-08,
                'ADDGRID': '.TRUE.',
                'LREAL': '.FALSE.'})
            incar = vasp_input_forces

        if type == 'born_charges':
            vasp_input_epsilon = dict(incar)
            vasp_input_epsilon.update({
                'PREC': 'Accurate',
                'LEPSILON': '.TRUE.',
                'ISTART': 0,
                'IBRION': 1,
                'NSW': 0,
                'LWAVE': '.FALSE.',
                'LCHARG': '.FALSE.',
                'EDIFF': 1e-08,
                'ADDGRID': '.TRUE.',
                'LREAL': '.FALSE.'})
            incar = vasp_input_epsilon

        # KPOINTS
        kpoints = parameters['kpoints']
        if 'kpoints_per_atom' in kpoints:
            kpoints = vaspio.Kpoints.automatic_density(structure.get_pymatgen_structure(), kpoints['kpoints_per_atom'])
#            num_kpoints = np.product(kpoints.kpts)
#            if num_kpoints < 4:
#                incar['ISMEAR'] = 0
#                incar['SIGMA'] = 0.05

        else:
            if not 'style' in kpoints:
                kpoints['style'] = 'Monkhorst'
            # supported_modes = Enum(("Gamma", "Monkhorst", "Automatic", "Line_mode", "Cartesian", "Reciprocal"))
            kpoints = vaspio.Kpoints(comment='aiida generated',
                                     style=kpoints['style'],
                                     kpts=(kpoints['points'],), kpts_shift=kpoints['shift'])

        calc.use_kpoints(ParameterData(dict=kpoints.as_dict()))
        # update incar (just in case something changed with kpoints)
        incar = vaspio.Incar(incar)
        calc.use_incar(ParameterData(dict=incar.as_dict()))


        # POTCAR
        pseudo = parameters['pseudo']
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

    def generate_calculation(self, structure, parameters, type='optimize'):
        code = Code.get_from_string(parameters['code'])
        plugin = code.get_attrs()['input_plugin'].split('.')[0]
        pressure = self.get_attribute('pressure')

        if plugin == 'lammps':
            return self.generate_calculation_lammps(structure, parameters, type=type, pressure=pressure)
        elif plugin == 'vasp':
            return self.generate_calculation_vasp(structure, parameters, type=type, pressure=pressure)
        elif plugin == 'quantumespresso':
            return self.generate_calculation_qe(structure, parameters, type=type, pressure=pressure)
        else:
            self.append_to_report('The plugin: {}, is not implemented in this workflow'.format(plugin))
            self.next(self.exit)
            return None


    # Starting workflow
    @Workflow.step
    def start(self):
        self.append_to_report('Workflow starting')

        self.add_attribute('counter', 10)  # define max number of optimization iterations

        if self._optimize:
            self.next(self.optimize)
        else:
            self.next(self.displacements)

        self.add_attribute('pressure', self._pressure)
        self.add_attribute('constant_volume', self._constant_volume)


    # Optimize the structure
    @Workflow.step
    def optimize(self):

        parameters = self.get_parameters()
        pressure = self.get_attribute('pressure')
        tolerance_forces = 1e-08
        tolerance_stress = 1e-04

        counter = self.get_attribute('counter')

        optimized = self.get_step_calculations(self.optimize)
        if len(optimized):
            last_calc = self.get_step_calculations(self.optimize).latest('id')
            try:
                structure = last_calc.out.output_structure


                forces = last_calc.out.output_array.get_array('forces')[-1]
                stresses = last_calc.out.output_array.get_array('stress')

                not_converged_forces = len(np.where(abs(forces) > tolerance_forces)[0])
                if len(stresses.shape) > 2:
                    stresses = stresses[-1] * 10

                not_converged_stress = len(np.where(abs(np.diag(stresses)-pressure) > tolerance_stress)[0])
                np.fill_diagonal(stresses, 0.0)
                not_converged_stress += len(np.where(abs(stresses) > tolerance_stress)[0])

                not_converged = not_converged_forces + not_converged_stress

                self.append_to_report('Not converged forces: {}'.format(not_converged_forces))
                self.append_to_report('Not converged stresses: {}'.format(not_converged_stress))

                if not_converged == 0:
                    self.next(self.displacements)
                    return

            except AttributeError:
                structure = last_calc.inp.structure

        else:
            structure = parameters['structure']

        # Standardize structure using spglib
        structure = standardize_cell_inline(structure=structure)[1]['standardized_structure']

        self.append_to_report('Optimize structure {}/{}'.format(len(optimized) + 1, len(optimized) + counter + 1))

        if self.get_attribute('constant_volume'):
            calc = self.generate_calculation(structure, parameters['input_optimize'], type='optimize_constant_volume')
        else:
            calc = self.generate_calculation(structure, parameters['input_optimize'], type='optimize')

        calc.label = 'optimization'
        print ('created calculation with PK={}'.format(calc.pk))
        self.attach_calculation(calc)

        if counter < 1:
            self.add_attribute('counter', 10)  # define max number of displacement iterations
            self.next(self.displacements)
        else:
            self.add_attribute('counter', counter - 1)
            self.next(self.optimize)

    # Prepare supercells with displacements
    @Workflow.step
    def displacements(self):

        counter = self.get_attribute('counter')
        self.append_to_report('Displacements')

        parameters = self.get_parameters()
        parameters_phonopy = parameters['phonopy_input']

        if len(self.get_step_calculations(self.displacements)):
            calcs = self.get_step_calculations(self.displacements)
            all_calc_ok = True
            for calc in calcs:
                if calc.label != 'FAILED' and not 'output_array' in calc.get_outputs_dict():

                    if counter < 1:
                        self.append_to_report('calc {}: Force calculation failed!')
                        self.next(self.exit)
                        return

                    self.append_to_report('calc {} FAILED, repeating..'.format(calc.label))
                    repeat_calc = self.generate_calculation(calc.inp.structure,
                                                            parameters['input_force'],
                                                            type='forces')
                    repeat_calc.label = calc.label
                    self.attach_calculation(repeat_calc)
                    calc.label = 'FAILED'
                    all_calc_ok = False
                    self.add_attribute('counter', counter - 1)

            if all_calc_ok:
                self.next(self.born_charges)
                return
        else:
            optimized = self.get_step('optimize')

            if optimized is not None:
                self.append_to_report('Optimized structure')
                opt_calc = self.get_step_calculations(self.optimize).latest('id')
                structure = opt_calc.get_outputs_dict()['output_structure']
                optimized_data = opt_calc.out.output_parameters
                self.add_result('optimized_structure_data', optimized_data)

            else:
                self.append_to_report('Initial structure')
                structure = parameters['structure']

            self.add_result('final_structure', structure)

            inline_params = {"structure": structure,
                             "phonopy_input": ParameterData(dict=parameters_phonopy['parameters']),
                             }

            cells_with_disp = create_supercells_with_displacements_inline(**inline_params)[1]

            # nodes = [ 762, 767, 772, 777]  #for debuging
            for i, cell in enumerate(cells_with_disp.iterkeys()):
                calc = self.generate_calculation(cells_with_disp['structure_{}'.format(i)],
                                                 parameters['input_force'], type='forces')

                calc.label = 'force_{}'.format(i)
                self.append_to_report('created calculation with PK={}'.format(calc.pk))
                self.attach_calculation(calc)

        self.next(self.displacements)

    #        if 'code' in parameters['phonopy_input']:
    #            self.append_to_report('Remote phonon calculation')
    #            self.next(self.force_constants_calculation_remote)
    #        else:
    #            self.append_to_report('Local phonon calculation')
    #            self.next(self.force_constants_calculation)


    @Workflow.step
    def born_charges(self):

        parameters = self.get_parameters()
        structure = self.get_result('final_structure')  # Collects the forces and prepares force constants

        calc = self.generate_calculation(structure, parameters['input_optimize'], type='born_charges')
        calc.label = 'single point'

        if calc is not None:
            self.attach_calculation(calc)

        if 'code' in parameters['phonopy_input']:
            self.append_to_report('Remote phonon calculation')
            self.next(self.force_constants_calculation_remote)
        else:
            self.append_to_report('Local phonon calculation')
            self.next(self.force_constants_calculation)
        return


    @Workflow.step
    def force_constants_calculation(self):

        parameters = self.get_parameters()
        parameters_phonopy = parameters['phonopy_input']

        calcs = list(self.get_step_calculations(self.displacements))

        structure = self.get_result('final_structure')

        self.append_to_report('reading structure')

        inline_params = {'structure': structure,
                         'phonopy_input': ParameterData(dict=parameters_phonopy['parameters'])}

        self.append_to_report('created parameters')

        for calc in calcs:
            if calc.label != 'FAILED':
                inline_params[calc.label] = calc.out.output_array
                self.append_to_report('extract force from {}'.format(calc.label))

        # Get the force constants and store it in DB as a Workflow result
        phonopy_data = get_force_constants_inline(**inline_params)[1]

        self.add_result('force_constants', phonopy_data['phonopy_output'])

        self.next(self.phonon_calculation)


    @Workflow.step
    def force_constants_calculation_remote(self):

        parameters = self.get_parameters()
        parameters_phonopy = parameters['phonopy_input']

        calcs = list(self.get_step_calculations(self.displacements))

        structure = self.get_result('final_structure')

        self.append_to_report('reading structure')

        inline_params = {'structure': structure,
                         'phonopy_input': ParameterData(dict=parameters_phonopy['parameters'])}

        self.append_to_report('created parameters')

        for calc in calcs:
            if calc.label != 'FAILED':
                data = calc.get_outputs_dict()['output_array']
                inline_params[calc.label] = data
                self.append_to_report('extract force from {}'.format(calc.label))

        # Get the force constants and store it in DB as a Workflow result
        phonopy_data = get_force_sets_inline(**inline_params)[1]

        calc = self.generate_calculation_phonopy(structure, parameters_phonopy, phonopy_data['phonopy_output'])
        self.attach_calculation(calc)

        self.next(self.phonon_calculation)


    @Workflow.step
    def phonon_calculation(self):
        #        self.add_result('force_constants', phonopy_data['phonopy_output'])

        parameters_phonopy = self.get_parameters()['phonopy_input']

        born_calc = self.get_step_calculations(self.born_charges)[0]
        born_charges = born_calc.get_outputs_dict()['output_array']

        remote_phonopy = self.get_step('force_constants_calculation_remote')

        if remote_phonopy is None:
            force_constants = self.get_result('force_constants')
        else:
            calc = self.get_step_calculations(self.force_constants_calculation_remote)[0]
            force_constants = calc.get_outputs_dict()['array_data']
            self.add_result('force_constants', force_constants)

        structure = self.get_result('final_structure')

        inline_params = {'structure': structure,
                         'phonopy_input': ParameterData(dict=parameters_phonopy['parameters']),
                         'force_constants': force_constants,
                         'nac_data': born_charges}

        results = phonopy_calculation_inline(**inline_params)[1]

        self.add_result('thermal_properties', results['thermal_properties'])
        self.add_result('dos', results['dos'])
        self.add_result('band_structure', results['band_structure'])

        self.next(self.exit)

