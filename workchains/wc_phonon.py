# Works run by the daemon (using submit)

from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida.work.workchain import WorkChain, ToContext
from aiida.work.workfunction import workfunction

from aiida.orm import Code, CalculationFactory, load_node, DataFactory

from aiida.orm.data.base import Str, Float, Bool
from aiida.orm.data.force_sets import ForceSets
from aiida.orm.data.force_constants import ForceConstants
from aiida.orm.data.band_structure import BandStructureData
from aiida.orm.data.phonon_dos import PhononDosData

from aiida.workflows.wc_optimize import OptimizeStructure
from aiida.work.workchain import _If, _While

import numpy as np
from generate_inputs import *

def generate_phonopy_params(code, structure, ph_settings, machine, force_sets):
    """
    Generate inputs parameters needed to do a remote phonopy calculation

    :param code: Code object of phonopy
    :param structure: StructureData Object that constains the crystal structure unit cell
    :param ph_settings: ParametersData object containing a dictionary with the phonopy input data
    :param machine: ParametersData object containing a dictionary with the computational resources information
    :param force_sets: ForceSetssData object containing the atomic forces and displacement information
    :return: Calculation process object, input dictionary
    """
    PhonopyCalculation = CalculationFactory('phonopy')

    # The inputs
    inputs = PhonopyCalculation.process().get_inputs_template()

    # code
    inputs.code = code

    # structure
    inputs.structure = structure

    # parameters
    inputs.parameters = ph_settings

    # resources
    inputs._options.resources = machine.dict.resources
    inputs._options.max_wallclock_seconds = machine.dict.max_wallclock_seconds

    # data_sets
    inputs.data_sets = force_sets

    return PhonopyCalculation.process(), inputs


@workfunction
def create_supercells_with_displacements_using_phonopy(structure, ph_settings):
    """
    Use phonopy to create the supercells with displacements to calculate the force constants by using
    finite displacements methodology

    :param structure: StructureData object
    :param phonopy_input: ParametersData object containing a dictionary with the data needed for phonopy
    :return: A set of StructureData Objects containing the supercells with displacements
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    import numpy as np

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     supercell_matrix=ph_settings.dict.supercell,
                     primitive_matrix=ph_settings.dict.primitive,
                     symprec=ph_settings.dict.symmetry_precision)

    phonon.generate_displacements(distance=ph_settings.dict.distance)

    cells_with_disp = phonon.get_supercells_with_displacements()

    # Transform cells to StructureData and set them ready to return
    data_sets = phonon.get_displacement_dataset()
    data_sets_object = ForceSets(data_sets=data_sets)

    disp_cells = {'data_sets': data_sets_object}
    for i, phonopy_supercell in enumerate(cells_with_disp):
        supercell = StructureData(cell=phonopy_supercell.get_cell())
        for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
                                    phonopy_supercell.get_positions()):
            supercell.append_atom(position=position, symbols=symbol)
        disp_cells["structure_{}".format(i)] = supercell

    return disp_cells


@workfunction
def create_forces_set(**kwargs):
    """
    Build data_sets from forces of supercells with displacments

    :param forces_X: ArrayData objects that contain the atomic forces for each supercell with displacement, respectively (X is integer)
    :param data_sets: ForceSetsData object that contains the displacements info (This info should match with forces_X)
    :return: ForceSetsData object that contains the atomic forces and displacements info (datasets dict in phonopy)

    """
    data_sets = kwargs.pop('data_sets')
    force_sets = ForceSets(data_sets=data_sets.get_data_sets())

    forces = []
    for i in range(data_sets.get_number_of_displacements()):
        forces.append(kwargs.pop('forces_{}'.format(i)).get_array('forces')[0])

    force_sets.set_forces(forces)

    return {'force_sets': force_sets}

@workfunction
def add_nac_to_force_constants(force_constants, array_data):
    """
    Create a new ForceConstants object with Born charges info from VASP array output included

    :param force_constants: original force constants
    :param array_data: ArrayData object that contains the dielectric tensor and Born charges
    :return: force_constants: ForceConstants object
    """

    force_constants_nac = ForceConstants(array=force_constants.get_array(),
                                         born_charges=array_data.get_array('born_charges'),
                                         epsilon=array_data.get_array('epsilon'))

    return {'force_constants': force_constants_nac}

@workfunction
def get_force_constants_from_phonopy(structure, ph_settings, force_sets):
    """
    Calculate the force constants locally using phonopy

    :param structure:
    :param phonopy_input: ParameterData object that contains phonopy settings
    :param force_sets: ForceSetsData object that contains the atomic forces and displacements info (datasets dict in phonopy)
    :return: ForceConstantsData object containing the 2nd order force constants calculated with phonopy
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     ph_settings.dict.supercell,
                     primitive_matrix=ph_settings.dict.primitive,
                     symprec=ph_settings.dict.symmetry_precision)

    phonon.generate_displacements(distance=ph_settings.dict.distance)

    # Build data_sets from forces of supercells with displacments
    phonon.set_displacement_dataset(force_sets.get_force_sets())
    phonon.produce_force_constants()

    force_constants = ForceConstants(array=phonon.get_force_constants())

    return {'force_constants': force_constants}



def get_path_using_seekpath(phonopy_structure, band_resolution=30):
    import seekpath

    cell = phonopy_structure.get_cell()
    scaled_positions = phonopy_structure.get_scaled_positions()
    numbers = phonopy_structure.get_atomic_numbers()

    structure = (cell, scaled_positions, numbers)
    path_data = seekpath.get_path(structure)

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

#    return {'ranges': bands,
#            'labels': path_data['path']}

    band_structure = BandStructureData(bands=bands,
                                       labels=path_data['path'],
                                       unitcell=phonopy_structure.get_cell())
    return band_structure


def get_born_parameters(phonon, born_charges, epsilon, symprec=1e-5):
    from phonopy.structure.cells import get_primitive, get_supercell
    from phonopy.structure.symmetry import Symmetry
    from phonopy.interface import get_default_physical_units
    from phonopy.interface.vasp import _get_borns

    # print ('inside born parameters')
    pmat = phonon.get_primitive_matrix()
    smat = phonon.get_supercell_matrix()
    #smat = np.identity(3)
    ucell = phonon.get_unitcell()
    num_atom = len(born_charges)
    assert num_atom == ucell.get_number_of_atoms(), \
        "num_atom %d != len(borns) %d" % (ucell.get_number_of_atoms(),
                                          len(born_charges))


    print 'pmat', pmat
    print 'smat', smat
    print 'ucell', ucell



    print born_charges
    inv_smat = np.linalg.inv(smat)
    scell = get_supercell(ucell, smat, symprec=symprec)
    pcell = get_primitive(scell, np.dot(inv_smat, pmat), symprec=symprec)
    p2s = np.array(pcell.get_primitive_to_supercell_map(), dtype='intc')
    p_sym = Symmetry(pcell, is_symmetry=True, symprec=symprec)
    s_indep_atoms = p2s[p_sym.get_independent_atoms()]
    u2u = scell.get_unitcell_to_unitcell_map()
    u_indep_atoms = [u2u[x] for x in s_indep_atoms]
    print u_indep_atoms
    reduced_borns = born_charges[u_indep_atoms].copy()

    pcell = get_primitive(scell, np.dot(inv_smat, pmat), symprec=symprec)
    print 'map', pcell.get_primitive_to_supercell_map()
    print 'map', pcell.get_supercell_to_primitive_map()
    print 'map', scell.get_supercell_to_unitcell_map()
    print 'map', scell.get_unitcell_to_supercell_map()

    factor = get_default_physical_units('vasp')['nac_factor']  # born charges in VASP units

    reduced_borns, epsilon, s_indep_atoms = _get_borns(ucell,born_charges, epsilon, primitive_matrix=pmat, supercell_matrix=smat,symprec=symprec)

    print reduced_borns
    born_dict = {'born': reduced_borns, 'dielectric': epsilon, 'factor': factor}

    exit()
    return born_dict


@workfunction
def get_properties_from_phonopy(structure, ph_settings, force_constants):
    """
    Calculate DOS and thermal properties using phonopy (locally)
    :param structure: StructureData Object
    :param ph_settings: Parametersdata object containing a dictionary with the data needed to run phonopy:
            supercells matrix, primitive matrix and q-points mesh.
    :param force_constants: ForceConstantsData object containing the 2nd order force constants
    :param nac_data: ArrayData object from a single point calculation data containing dielectric tensor and Born charges
    :return: phonopy thermal properties and DOS
    """

    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     supercell_matrix=ph_settings.dict.supercell,
                     primitive_matrix=ph_settings.dict.primitive,
                     symprec=ph_settings.dict.symmetry_precision)

    phonon.set_force_constants(force_constants.get_array())

    if force_constants.epsilon_and_born_exist():
        print ('use born charges')
        born_parameters = get_born_parameters(phonon,
                                              force_constants.get_born_charges(),
                                              force_constants.get_epsilon(),
                                              ph_settings.dict.symmetry_precision)
        print bulk.get_cell()
        print born_parameters

        phonon.set_nac_params(born_parameters)

    # Normalization factor primitive to unit cell
    normalization_factor = phonon.unitcell.get_number_of_atoms()/phonon.primitive.get_number_of_atoms()

    # DOS
    phonon.set_mesh(ph_settings.dict.mesh, is_eigenvectors=True, is_mesh_symmetry=False)
    phonon.set_total_DOS()
    phonon.set_partial_DOS()

    total_dos = phonon.get_total_DOS()
    partial_dos = phonon.get_partial_DOS()

    dos = PhononDosData(frequencies=total_dos[0],
                        dos=total_dos[1]*normalization_factor,
                        partial_dos=partial_dos[1]*normalization_factor,
                        atom_labels=np.array(phonon.primitive.symbols))

    # THERMAL PROPERTIES (per primtive cell)
    phonon.set_thermal_properties()
    t, free_energy, entropy, cv = phonon.get_thermal_properties()

    # Stores thermal properties (per unit cell) data in DB as a workflow result
    thermal_properties = ArrayData()
    thermal_properties.set_array('temperature', t)
    thermal_properties.set_array('free_energy', free_energy * normalization_factor)
    thermal_properties.set_array('entropy', entropy * normalization_factor)
    thermal_properties.set_array('cv', cv * normalization_factor)

    # BAND STRUCTURE
    band_structure = get_path_using_seekpath(phonon.get_primitive())
    phonon.set_band_structure(band_structure.get_bands())
    band_structure.set_band_structure_phonopy(phonon.get_band_structure())

    return {'thermal_properties': thermal_properties, 'dos': dos, 'band_structure': band_structure}

class PhononPhonopy(WorkChain):
    """
    Workchain to do a phonon calculation using phonopy

    :param structure: StructureData object that contains the crystal structure unit cell
    :param ph_settings: ParametersData object that contains a dictionary with the data needed to run phonopy:
                                  'supercell': [[2,0,0],
                                                [0,2,0],
                                                [0,0,2]],
                                  'primitive': [[1.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0]],
                                  'distance': 0.01,
                                  'mesh': [40, 40, 40],
                                  # 'code': 'phonopy@boston'  # include this to run phonopy remotely otherwise run phonopy localy

    :param es_settings: ParametersData object that contains a dictionary with the setting needed to calculate the electronic structure.
                        The structure of this dictionary strongly depends on the software (VASP, QE, LAMMPS, ...)
    :param optimize: Set true to perform a crystal structure optimization before the phonon calculation (default: True)
    :param pressure: Set the external pressure (stress tensor) at which the optimization is performed in KBar (default: 0)
    """
    @classmethod
    def define(cls, spec):
        super(PhononPhonopy, cls).define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("machine", valid_type=ParameterData)
        spec.input("ph_settings", valid_type=ParameterData)
        spec.input("es_settings", valid_type=ParameterData)
        # Optional arguments
        spec.input("optimize", valid_type=Bool, required=False, default=Bool(True))
        spec.input("pressure", valid_type=Float, required=False, default=Float(0.0))

        spec.outline(_If(cls.use_optimize)(cls.optimize), cls.create_displacement_calculations, cls.get_force_constants, cls.calculate_phonon_properties)

    def use_optimize(self):
        return self.inputs.optimize

    def optimize(self):
        print('start phonon {}'.format(self.pid))
        print ('start optimize')
        future = submit(OptimizeStructure,
                        structure=self.inputs.structure,
                        machine=self.inputs.machine,
                        es_settings=self.inputs.es_settings,
                        pressure=self.inputs.pressure,
                        )
        # For testing
        testing = True
        if testing:
            self.ctx._content['optimize'] = load_node(13047)
            return

        print ('optimize workchain: {}'.format(future.pid))

        return ToContext(optimized=future)

    def create_displacement_calculations(self):
        print ('create displacements')
        self.report('create displacements')

        # print self.ctx._get_dict()

        if 'optimized' in self.ctx:
            self.ctx.final_structure = self.ctx.optimized.out.optimized_structure
        else:
            self.ctx.final_structure = self.inputs.structure

        supercells = create_supercells_with_displacements_using_phonopy(self.ctx.final_structure,
                                                                        self.inputs.ph_settings)

        self.ctx.data_sets = supercells.pop('data_sets')
        self.ctx.number_of_displacements = len(supercells)

        calcs = {}

        # Load data from nodes
        testing = True
        if testing:
            from aiida.orm import load_node
            nodes = [13147, 13152, 13157, 13162]  # VASP
            labels = ['structure_1', 'structure_0', 'structure_3', 'structure_2']
            for pk, label in zip(nodes, labels):
                future = load_node(pk)
                self.ctx._content[label] = future

            self.ctx._content['born_charges'] = load_node(13167)
            return

        for label, supercell in supercells.iteritems():
            # print label, structure

            #print self.inputs.es_settings.dict.code

            # plugin = self.inputs.code.get_attr('input_plugin')

            JobCalculation, calculation_input = generate_inputs(supercell,
                                                                self.inputs.machine,
                                                                self.inputs.es_settings,
                                                                #pressure=self.input.pressure,
                                                                type='forces')

            calculation_input._label = label
            future = submit(JobCalculation, **calculation_input)
            print label, future.pid
            self.report('{} pk = {}'.format(label, future.pid))

            calcs[label] = future

        # Born charges
        if 'born_charges' in self.inputs.es_settings.dict.code:
            self.report('calculate born charges')
            JobCalculation, calculation_input = generate_inputs(self.ctx.final_structure,
                                                                self.inputs.machine,
                                                                self.inputs.es_settings,
                                                                #pressure=self.input.pressure,
                                                                type='born_charges')
            future = submit(JobCalculation, **calculation_input)
            print ('born_charges: {}'.format(future.pid))
            calcs['born_charges'] = future

        return ToContext(**calcs)

    def get_force_constants(self):

        print ('calculate force constants')
        self.report('calculate force constants')

        wf_inputs = {}
        for i in range(self.ctx.number_of_displacements):
            wf_inputs['forces_{}'.format(i)] = self.ctx.get('structure_{}'.format(i)).out.output_array
        wf_inputs['data_sets'] = self.ctx.data_sets

        self.ctx.force_sets = create_forces_set(**wf_inputs)['force_sets']

        if 'code' in self.inputs.ph_settings.get_dict():
            print ('remote phonopy FC calculation')
            code_label = self.inputs.ph_settings.get_dict()['code']
            JobCalculation, calculation_input = generate_phonopy_params(code=Code.get_from_string(code_label),
                                                                        structure=self.ctx.final_structure,
                                                                        ph_settings=self.inputs.ph_settings,
                                                                        machine=self.inputs.machine,
                                                                        force_sets=self.ctx.force_sets)
            future = submit(JobCalculation, **calculation_input)
            print 'phonopy FC calc:', future.pid

            return ToContext(phonopy_output=future)
        else:
            print ('local phonopy FC calculation')
            self.ctx.phonopy_output = get_force_constants_from_phonopy(structure=self.ctx.final_structure,
                                                                       ph_settings=self.inputs.ph_settings,
                                                                       force_sets=self.ctx.force_sets)

        return

    def calculate_phonon_properties(self):

        # print self.ctx._get_dict()
        print ('calculate phonon properties')
        self.report('calculate phonon properties')

        try:
            force_constants = self.ctx.phonopy_output['force_constants']
        except TypeError:
            force_constants = self.ctx.phonopy_output.out.force_constants

        if 'born_charges' in self.ctx:
            force_constants = add_nac_to_force_constants(force_constants, self.ctx.born_charges.out.output_array)['force_constants']

        phonon_properties = get_properties_from_phonopy(structure=self.ctx.final_structure,
                                                        ph_settings=self.inputs.ph_settings,
                                                        force_constants=force_constants)

        self.out('force_constants', force_constants)
        self.out('thermal_properties', phonon_properties['thermal_properties'])
        self.out('dos', phonon_properties['dos'])
        self.out('band_structure', phonon_properties['band_structure'])
        self.out('final_structure', self.ctx.final_structure)

        self.report('finish phonon')

        return
