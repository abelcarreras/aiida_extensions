# Works run by the daemon (using submit)

from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida.work.workchain import WorkChain, ToContext
from aiida.work.workfunction import workfunction

from aiida.work.run import run, submit, async
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.upf import UpfData
from aiida.orm.data.base import Str, Float, Bool

#from aiida.orm.calculation.job.quantumespresso.pw import PwCalculation
#from aiida.orm.calculation.job.vasp.vasp import VaspCalculation
from aiida.work.workchain import if_

VaspCalculation = CalculationFactory('vasp.vasp')
PwCalculation = CalculationFactory('quantumespresso.pw')
PhonopyCalculation = CalculationFactory('phonopy')

import numpy as np


# Function obtained from aiida's quantumespresso plugin. Copied here for convinence
def get_pseudos(structure, family_name):
    """
    Set the pseudo to use for all atomic kinds, picking pseudos from the
    family with name family_name.

    :note: The structure must already be set.

    :param family_name: the name of the group containing the pseudos
    """
    from collections import defaultdict
    from aiida.orm.data.upf import get_pseudos_from_structure

    # A dict {kind_name: pseudo_object}
    kind_pseudo_dict = get_pseudos_from_structure(structure, family_name)

    # We have to group the species by pseudo, I use the pseudo PK
    # pseudo_dict will just map PK->pseudo_object
    pseudo_dict = {}
    # Will contain a list of all species of the pseudo with given PK
    pseudo_species = defaultdict(list)

    for kindname, pseudo in kind_pseudo_dict.iteritems():
        pseudo_dict[pseudo.pk] = pseudo
        pseudo_species[pseudo.pk].append(kindname)

    pseudos = {}
    for pseudo_pk in pseudo_dict:
        pseudo = pseudo_dict[pseudo_pk]
        kinds = pseudo_species[pseudo_pk]
        for kind in kinds:
            pseudos[kind] = pseudo

    return pseudos

def generate_qe_params(code, structure, machine, settings, kpoints, pseudo):
    """
    generate the input paramemeters needed to run a calculation for PW (Quantum Espresso)
    :param code: aiida Code object
    :param structure:  aiida StructureData object
    :param machine: aiida ParametersData object containing a dictionary with the computational resources information
    :param settings: aiida ParametersData object containing a dictionary with the input parameters for PW
    :param kpoints: aiida KpointsData object
    :param pseudo: aiida Str object containing the label of the pseudopotentals family to use
    :return: Calculation process object, input dictionary
    """


    # The inputs
    inputs = PwCalculation.process().get_inputs_template()
    inputs.code = code

    # The structure
    inputs.structure = structure

    # Machine
    inputs._options.resources = machine.dict.resources
    inputs._options.max_wallclock_seconds = machine.dict.max_wallclock_seconds

    # Parameters
    inputs.parameters = settings

    # Kpoints
    inputs.kpoints = kpoints

    # Pseudopotentials
    ######## MANUALTEST #########
    manual_pseudo = False
    if manual_pseudo:
        # Pseudopotentials (test)
        pseudo_dir = '/Users/abel/software/espresso/pbe/'
        raw_pseudos = [
           pseudo_dir + "Ga.pbe-dn-rrkjus_psl.1.0.0.UPF",
           pseudo_dir + "N.pbe-n-rrkjus_psl.1.0.0.UPF"]

        pseudos = {}

        for file_path in raw_pseudos:
            pseudo, created = UpfData.get_or_create(file_path, use_first=True)
            pseudos.update({pseudo.element: pseudo})

        inputs.pseudo = pseudos
        return PwCalculation.process(), inputs
    ######## MANUAL TEST #########

    inputs.pseudo = get_pseudos(structure, pseudo)

    return PwCalculation.process(), inputs


def generate_vasp_params(code, structure, machine, settings):
    """
    generate the input paramemeters needed to run a calculation for VASP
    :param structure:  aiida StructureData object
    :param machine: aiida ParametersData object containing a dictionary with the computational resources information
    :param settings: aiida ParametersData object containing a dictionary with the INCAR parameters
    :return: Calculation process object, input dictionary
    """

    #plugin = self.inputs.es_settings.dict.code.get_attr('input_plugin')

    # The inputs
    inputs = VaspCalculation.process().get_inputs_template()

    # code
    inputs.code = code

    # structure
    inputs.structure = structure

    inputs._options.resources = machine.dict.resources
    inputs._options.max_wallclock_seconds = machine.dict.max_wallclock_seconds

    # INCAR (parameters)
    inputs.incar = ParameterData(dict=settings.dict.incar)


    # POTCAR (pseudo)
    inputs.potcar = ParameterData(dict=settings.dict.pseudos)

    settings = {'PARSER_INSTRUCTIONS':[]}
    pinstr = settings['PARSER_INSTRUCTIONS']
    pinstr += [{
        'instr': 'array_data_parser',
        'type': 'data',
        'params': {}},
        {
        'instr': 'output_parameters',
        'type': 'data',
        'params': {}},
        {
        'instr': 'dummy_error_parser',
        'type': 'error',
        'params': {}},
        {
        'instr': 'default_structure_parser',
        'type': 'structure',
        'params': {}}
    ]

    # Kpoints
    from pymatgen.io import vasp as vaspio
    kpoints_pg = vaspio.Kpoints.monkhorst_automatic(kpts=[2, 2, 2],
                                                    shift=[0.0, 0.0, 0.0])
    kpoints = ParameterData(dict=kpoints_pg.as_dict())
    inputs.kpoints = kpoints

    inputs.settings = ParameterData(dict=settings)

    return VaspCalculation.process(), inputs



def generate_phonopy_params(code, structure, parameters, machine, data_sets):
    """
    Generate inputs parameters needed to do a remote phonopy calculation
    :param code: Aiida Code object
    :param structure: Aiida StructureData Object
    :param parameters: Aiida Parametersdata object containing a dictionary with the data neede to run a phonopy
                       calculation: supercell matrix, primitive matrix, displacement, distance and mesh (others may be included in the future)
    :param machine: Aiida Parametersdata object containing a dictioary with the computational resources information
    :param data_sets: Aiida ParametersData object containing the collected forces and displacement onformation of all the supercells
    :return: Calculation process object, input dictionary
    """

    # The inputs
    inputs = PhonopyCalculation.process().get_inputs_template()

    # code
    inputs.code = code

    # structure
    inputs.structure = structure

    # parameters
    inputs.parameters = parameters

    # resources
    inputs._options.resources = machine.dict.resources
    inputs._options.max_wallclock_seconds = machine.dict.max_wallclock_seconds

    # data_sets
    inputs.data_sets = data_sets

    return PhonopyCalculation.process(), inputs




@workfunction
def create_supercells_with_displacements_using_phonopy(structure, phonopy_input):
    """
    Create the supercells with the displacements to use the finite displacements methodology to calculate the
    force constants
    :param structure: Aiida StructureData Object
    :param phonopy_input: Aiida Parametersdata object containing a dictionary with the data needed to run phonopy:
            supercells matrix, primitive matrix and displacement distance.
    :return: dictionary of Aiida StructureData Objects containing the cells with displacements
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    import numpy as np

   # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonopy_input = phonopy_input.get_dict()
    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'])

    phonon.generate_displacements(distance=phonopy_input['distance'])

    cells_with_disp = phonon.get_supercells_with_displacements()

 # Transform cells to StructureData and set them ready to return
    data_sets = phonon.get_displacement_dataset()
    data_sets_object = ArrayData()
    for i, first_atoms in enumerate(data_sets['first_atoms']):
        data_sets_array = np.array([first_atoms['direction'], first_atoms['number'], first_atoms['displacement']])
        data_sets_object.set_array('data_sets_{}'.format(i), data_sets_array)


    disp_cells = {'data_sets':data_sets_object}
    for i, phonopy_supercell in enumerate(cells_with_disp):
        supercell = StructureData(cell=phonopy_supercell.get_cell())
        for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
                                    phonopy_supercell.get_positions()):
            supercell.append_atom(position=position, symbols=symbol)
        disp_cells["structure_{}".format(i)] = supercell

    return disp_cells


@workfunction
def create_forces_set(**kwargs):
    # Build data_sets from forces of supercells with displacments

    import numpy as np

    data_set = kwargs.pop('data_sets')

    force_sets = ArrayData()
    for i in data_set.get_arraynames():
        force_array = kwargs.pop(i.replace('data_sets', 'forces')).get_array('forces')[0]
        data_set_array =  np.array([data_set.get_array(i)[0], data_set.get_array(i)[1], data_set.get_array(i)[2], force_array])
        force_sets.set_array(i, data_set_array)

    return {'force_sets': force_sets}

@workfunction
def get_force_constants_from_phonopy(**kwargs):
    """
    Calculate the force constants using phonopy
    :param kwargs:
    :return:
    """

    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    import numpy as np
 #   print 'function',kwargs

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input')

 # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input.get_dict()['supercell'],
                     primitive_matrix=phonopy_input.get_dict()['primitive'],
                     distance=phonopy_input.get_dict()['distance'])

 # Build data_sets from forces of supercells with displacments

    data_sets = phonon.get_displacement_dataset()
    for i, first_atoms in enumerate(data_sets['first_atoms']):
        forces = kwargs.pop('forces_{}'.format(i)).get_array('forces')[0]
        first_atoms['forces'] = np.array(forces, dtype='double', order='c')

    # LOCAL calculation

    # Calculate and get force constants
    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()

    force_constants = phonon.get_force_constants()

    array_data = ArrayData()
    array_data.set_array('force_constants', force_constants)

    return {'array_data': array_data}


@workfunction
def get_properties_from_phonopy(structure, phonopy_input, force_constants):
    """
    Calculate DOS and thermal properties using phonopy (locally)
    :param structure: Aiida StructureData Object
    :param phonopy_input: Aiida Parametersdata object containing a dictionary with the data needed to run phonopy:
            supercells matrix, primitive matrix and q-points mesh.
    :param force_constants:
    :return:
    """

    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

   # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonopy_input = phonopy_input.get_dict()
    force_constants = force_constants.get_array('force_constants')

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'])

    phonon.set_force_constants(force_constants)

    #Normalization factor primitive to unit cell
    normalization_factor = phonon.unitcell.get_number_of_atoms()/phonon.primitive.get_number_of_atoms()

    phonon.set_mesh(phonopy_input['mesh'], is_eigenvectors=True, is_mesh_symmetry=False)
    phonon.set_total_DOS()
    phonon.set_partial_DOS()

    # get DOS (normalized to unit cell)
    total_dos = phonon.get_total_DOS()*normalization_factor
    partial_dos = phonon.get_partial_DOS()*normalization_factor

    # Stores DOS data in DB as a workflow result
    dos = ArrayData()
    dos.set_array('frequency',total_dos[0])
    dos.set_array('total_dos',total_dos[1])
    dos.set_array('partial_dos',partial_dos[1])


    #THERMAL PROPERTIES (per primtive cell)
    phonon.set_thermal_properties()
    t, free_energy, entropy, cv = phonon.get_thermal_properties()

    # Stores thermal properties (per unit cell) data in DB as a workflow result
    thermal_properties = ArrayData()
    thermal_properties.set_array('temperature', t)
    thermal_properties.set_array('free_energy', free_energy*normalization_factor)
    thermal_properties.set_array('entropy', entropy*normalization_factor)
    thermal_properties.set_array('cv', cv*normalization_factor)

    return {'thermal_properties': thermal_properties, 'dos': dos}



class FrozenPhonon(WorkChain):
    """
    Workflow to calculate the force constants and phonon properties using phonopy
    """

    @classmethod
    def define(cls, spec):
        super(FrozenPhonon, cls).define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("machine", valid_type=ParameterData)
        spec.input("ph_settings", valid_type=ParameterData)
        spec.input("es_settings", valid_type=ParameterData)
        # Should be optional
        spec.input("optimize", valid_type=Bool)
        spec.input("pressure", valid_type=Float)
      #  spec.dynamic_input("optimize")

        #spec.outline(cls.create_displacement_calculations,
        #             if_(cls.remote_phonopy)(cls.get_force_constants_remote,
        #                                     cls.collect_phonopy_data).else_(
        #                 cls.get_force_constants))

        print 'test1!'
        spec.outline(cls.create_displacement_calculations, cls.get_force_constants)
        #spec.outline(cls.create_displacement_calculations, cls.get_force_constants_remote, cls.collect_phonopy_data)

        # spec.dynamic_output()

    def remote_phonopy(self, ctx):
        return 'code' in self.inputs.ph_settings.get_dict()

    def create_displacement_calculations(self, ctx):

        print 'test2!'
        structures = create_supercells_with_displacements_using_phonopy(self.inputs.structure,
                                                                        self.inputs.ph_settings)

        print 'test!'
        ctx.data_sets = structures.pop('data_sets')
        ctx.number_of_displacements = len(structures)

        generate_inputs = { 'quantumespresso.pw' : generate_qe_params,
                            'vasp.vasp': generate_vasp_params}

############### FOR TESTING ###############
# 1) Load data from nodes
        if False: #For test
            from aiida.orm import load_node
            nodes = [4768, 4771, 4774, 4777] # QE
         #   nodes = [4646, 4651, 4656, 4661] # VASP
            nodes = [5507, 5509, 5511, 5513] # VASP
            labels = ['structure_1', 'structure_0', 'structure_3', 'structure_2']
            for pk, label in zip(nodes, labels):
                future = load_node(pk)
                ctx._content[label] = future.get_outputs_dict()
            return

        calcs = {}
        for label, structure in structures.iteritems():
            print label, structure

            # plugin = self.inputs.code.get_attr('input_plugin')
            try:
                plugin = self.inputs.es_settings.dict.code.get_attr('input_plugin')
            except:
                plugin = self.inputs.es_settings.dict.code_forces.get_attr('input_plugin')

            JobCalculation, calculation_input = generate_inputs[plugin](structure,
                                                                        self.inputs.machine,
                                                                        self.inputs.es_settings)

            calculation_input._label = label
            future = submit(JobCalculation, **calculation_input)
            calcs[label] = future

        return ToContext(**calcs)

    def get_force_constants(self, ctx):

  #      print ctx._get_dict()

        wf_inputs = {}
        for key, value in ctx._get_dict().iteritems():
            if key.startswith('structure_'):
                wf_inputs[key.replace('structure', 'forces')] = value['output_array']

        wf_inputs['structure'] = self.inputs.structure
        wf_inputs['phonopy_input'] = self.inputs.ph_settings

        wf_inputs['machine'] = self.inputs.machine

        phonopy_output = get_force_constants_from_phonopy(**wf_inputs)
        force_constants = phonopy_output['array_data']

        phonon_properties = get_properties_from_phonopy(self.inputs.structure,
                                                     self.inputs.ph_settings,
                                                     force_constants)

        self.out('force_constants', force_constants)
        self.out('phonon_properties', phonon_properties['thermal_properties'])
        self.out('dos', phonon_properties['dos'])

    def get_force_constants_remote(self, ctx):
        wf_inputs = {}
        for key, value in ctx._get_dict().iteritems():
            if key.startswith('structure_'):
                wf_inputs[key.replace('structure', 'forces')] = value['output_array']

        wf_inputs['data_sets'] = ctx.data_sets
        force_sets = create_forces_set(**wf_inputs)['force_sets']

        code_label = self.inputs.ph_settings.get_dict()['code']

        JobCalculation, calculation_input = generate_phonopy_params(Code.get_from_string(code_label),
                                                                    self.inputs.structure,
                                                                    self.inputs.ph_settings,
                                                                    self.inputs.machine,
                                                                    force_sets)

        future = submit(JobCalculation, **calculation_input)
        calcs = {'phonopy_results': future}

        return ToContext(**calcs)

    def collect_phonopy_data(self, ctx):

        force_constants = ctx.phonopy_results['array_data']

        phonon_properties = get_properties_from_phonopy(self.inputs.structure,
                                                        self.inputs.ph_settings,
                                                        force_constants)

        self.out('force_constants', force_constants)
        self.out('phonon_properties', phonon_properties['thermal_properties'])
        self.out('dos', phonon_properties['dos'])


################### EXAMPLE INPUT FOR VASP AND QUANTUM ESPRESSO ###################

if __name__ == "__main__":

    # Define structure

    import numpy as np

    cell = [[ 3.1900000572, 0,           0],
            [-1.5950000286, 2.762621076, 0],
            [ 0.0,          0,           5.1890001297]]

    structure = StructureData(cell=cell)

    scaled_positions=[(0.6666669,  0.3333334,  0.0000000),
                      (0.3333331,  0.6666663,  0.5000000),
                      (0.6666669,  0.3333334,  0.3750000),
                      (0.3333331,  0.6666663,  0.8750000)]

    symbols=['Ga', 'Ga', 'N', 'N']

    positions = np.dot(scaled_positions, cell)

    for i, scaled_position in enumerate(scaled_positions):
        structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                              symbols=symbols[i])


# PHONOPY settings
    ph_settings = ParameterData(dict={'supercell': [[2,0,0],
                                                    [0,2,0],
                                                    [0,0,2]],
                                      'primitive': [[1.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 1.0]],
                                      'distance': 0.01,
                                      'mesh': [40, 40, 40],
                                      # 'code': 'phonopy@stern_outside'  # comment to use local phonopy
                                      })

# VASP SPECIFIC
    if True:   # Set TRUE to use VASP or FALSE to use Quantum Espresso
        incar_dict = {
            # 'PREC'   : 'Accurate',
            'EDIFF'  : 1e-08,
            'NELMIN' : 5,
            'NELM'   : 100,
            'ENCUT'  : 400,
            'ALGO'   : 38,
            'ISMEAR' : 0,
            'SIGMA'  : 0.01,
            'GGA'    : 'PS'
        }

        es_settings = ParameterData(dict=incar_dict)


        from pymatgen.io import vasp as vaspio
        #kpoints
        #kpoints_pg = vaspio.Kpoints.monkhorst_automatic(
        #                         kpts=[2, 2, 2],
        #                         shift=[0.0, 0.0, 0.0])
        #kpoints = ParameterData(dict=kpoints_pg.as_dict())

        potcar = vaspio.Potcar(symbols=['Ga', 'N'],
                               functional='PBE')

        settings_dict = {'code': 'vasp541mpi@boston',
                         'parameters': incar_dict,
                         'kpoints_per_atom': 1000,  # k-point density
                         'pseudos': potcar.as_dict()}

        # pseudos = ParameterData(dict=potcar.as_dict())
        es_settings = ParameterData(dict=settings_dict)



    # QE SPECIFIC
    if False:
        parameters_dict = {
            'CONTROL': {'calculation': 'scf',
                        'tstress': True,  #  Important that this stays to get stress
                        'tprnfor': True,},
            'SYSTEM': {'ecutwfc': 30.,
                       'ecutrho': 200.,},
            'ELECTRONS': {'conv_thr': 1.e-6,}
        }

        # Kpoints
        #kpoints_mesh = 2
        #kpoints = KpointsData()
        #kpoints.set_kpoints_mesh([kpoints_mesh, kpoints_mesh, kpoints_mesh])
        #code = Code.get_from_string('pw@stern_outside')

        pseudos = Str('pbe_ps')

        settings_dict = {'code': 'pw@stern_outside',
                         'parameters': parameters_dict,
                         'kpoints_per_atom': 1000,  # k-point density
                         'pseudos_family': 'pbe_ps'}

        es_settings = ParameterData(dict=settings_dict)


    # LAMMPS SPECIFIC
    if False:
        # GaN Tersoff
        tersoff_gan = {
            'Ga Ga Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199',
            'N  N  N': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77',
            'Ga Ga N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
            'Ga N  N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
            'N  Ga Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
            'N  Ga N ': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000',
            'N  N  Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
            'Ga N  Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000'}

        # Silicon(C) Tersoff
        # tersoff_si = {'Si  Si  Si ': '3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8'}


        potential = {'pair_style': 'tersoff',
                     'data': tersoff_gan}

        parameters = {'relaxation': 'tri',  # iso/aniso/tri
                      'pressure': 0.0,  # kbars
                      'vmax': 0.000001,  # Angstrom^3
                      'energy_tolerance': 1.0e-25,  # eV
                      'force_tolerance': 1.0e-25,  # eV angstrom
                      'max_evaluations': 1000000,
                      'max_iterations': 500000}

        settings_dict = {'code_forces': 'lammps_force@stern',
                         'code_optimize': 'lammps_optimize@stern',
                         'parameters': parameters,
                         'potential': potential}

        es_settings = ParameterData(dict=settings_dict)



    # CODE INDEPENDENT
    machine_dict = {'resources': {'num_machines': 1,
                                  'parallel_env': 'mpi*',
                                  'tot_num_mpiprocs': 16},
                    'max_wallclock_seconds': 30 * 60,
                    }

    machine = ParameterData(dict=machine_dict)

    results = run(FrozenPhonon,
                  structure=structure,
                  machine=machine,
                  es_settings=es_settings,
                  ph_settings=ph_settings,
                  # Optional settings
                  pressure=Float(10),
                  optimize=Bool(0)
                  )

    # Check results

    print results['force_constants'].get_array('force_constants')

    print results['force_constants'].pk
    print results['phonon_properties'].pk
    print results['dos'].pk
