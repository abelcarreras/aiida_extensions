from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()


from aiida.work.run import run, submit, async
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData

PhonopyCalculation = CalculationFactory('phonopy')


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

    PwCalculation = CalculationFactory('quantumespresso.pw')
    from aiida.orm.data.upf import UpfData


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


def generate_lammps_params(structure, machine, settings, pressure=0.0, type=None):
    """
    generate the input paramemeters needed to run a calculation for LAMMPS
    :param structure:  aiida StructureData object
    :param machine: aiida ParametersData object containing a dictionary with the computational resources information
    :param settings: aiida ParametersData object containing a dictionary with the LAMMPS parameters
    :return: Calculation process object, input dictionary
    """

    if type is None:
        code = settings.dict.code
    else:
        code = settings.dict.code[type]

    plugin = Code.get_from_string(code).get_attr('input_plugin')
    LammpsCalculation = CalculationFactory(plugin)
    inputs = LammpsCalculation.process().get_inputs_template()
    inputs.code = Code.get_from_string(code)

    inputs._options.resources = machine.dict.resources
    inputs._options.max_wallclock_seconds = machine.dict.max_wallclock_seconds

    inputs.structure = structure
    inputs.potential = ParameterData(dict=settings.dict.potential)

    # if code.get_input_plugin_name() == 'lammps.optimize':
    if type == 'optimize':
        lammps_parameters = dict(settings.dict.parameters)
        lammps_parameters.update({'pressure': pressure})  # pressure kb
        inputs.parameters = ParameterData(dict=lammps_parameters)

    return LammpsCalculation.process(), inputs


def generate_vasp_params(structure, machine, settings, type=None, pressure=0.0):
    """
    generate the input paramemeters needed to run a calculation for VASP
    :param structure:  aiida StructureData object
    :param machine: aiida ParametersData object containing a dictionary with the computational resources information
    :param settings: aiida ParametersData object containing a dictionary with the INCAR parameters
    :return: Calculation process object, input dictionary
    """

    if type is None:
        code = settings.dict.code
    else:
        code = settings.dict.code[type]

    plugin = Code.get_from_string(code).get_attr('input_plugin')

    VaspCalculation = CalculationFactory(plugin)

    inputs = VaspCalculation.process().get_inputs_template()

    # code
    inputs.code = Code.get_from_string(code)


    # structure
    inputs.structure = structure

    inputs._options.resources = machine.dict.resources
    inputs._options.max_wallclock_seconds = machine.dict.max_wallclock_seconds

    # INCAR (parameters)
    incar = dict(settings.dict.parameters)

    if type == 'optimize':
        incar.update({
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
            'PSTRESS': pressure})  # unit: kb -> kB

    elif type == 'optimize_constant_volume':
        incar.update({
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

    elif type == 'forces':
        incar.update({
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

    elif type == 'born_charges':
        incar.update({
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

    inputs.incar = ParameterData(dict=incar)

    # POTCAR (pseudo potentials)
    inputs.potcar = ParameterData(dict=settings.dict.pseudos)

    settings_parse = {'PARSER_INSTRUCTIONS': []}
    pinstr = settings_parse['PARSER_INSTRUCTIONS']
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

    if 'kpoints_per_atom' in settings.get_dict():
        kpoints_pg = vaspio.Kpoints.automatic_density(structure.get_pymatgen_structure(), settings.dict.kpoints_per_atom)
    else:
        print settings.dict.kpoints
        kpoints_pg = vaspio.Kpoints(comment='aiida generated',
                                    style=settings.dict.kpoints['type'],
                                    kpts=(settings.dict.kpoints['points'],),
                                    kpts_shift=settings.dict.kpoints['shift'])

    inputs.kpoints = ParameterData(dict=kpoints_pg.as_dict())

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

    inputs.settings = ParameterData(dict=settings)


    return VaspCalculation.process(), inputs


def generate_inputs(structure, machine, es_settings, type=None, pressure=0.0):

    print 'generate inputs start, type', type

    if type is None:
        plugin = Code.get_from_string(es_settings.dict.code).get_attr('input_plugin')

    else:
        print es_settings.dict.code
        plugin = Code.get_from_string(es_settings.dict.code[type]).get_attr('input_plugin')

    print 'plugin', plugin
    if plugin == 'vasp.vasp':
        return generate_vasp_params(structure, machine, es_settings, type=type)

    # elif plugin == 'quantumespresso.pw':
    #     return generate_qe_params(structure, machine, es_settings)

    elif plugin in ['lammps.force', 'lammps.optimize', 'lammps.md']:
        return generate_lammps_params(structure, machine, es_settings, type=type, pressure=pressure)
    else:
        print 'No supported plugin'
        exit()
