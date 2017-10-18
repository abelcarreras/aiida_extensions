# Works run by the daemon (using submit)

from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida.work.workchain import WorkChain, ToContext
from aiida.work.workfunction import workfunction

from aiida.work.run import run, submit, async
from aiida.orm import Code, CalculationFactory, load_node
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.upf import UpfData
from aiida.orm.data.base import Str, Float, Bool
from aiida.orm.data.force_sets import ForceSets
from aiida.orm.data.force_constants import ForceConstants


#from aiida.orm.calculation.job.quantumespresso.pw import PwCalculation
#from aiida.orm.calculation.job.vasp.vasp import VaspCalculation
from aiida.work.workchain import if_

PwCalculation = CalculationFactory('quantumespresso.pw')
PhonopyCalculation = CalculationFactory('phonopy')

from aiida.workflows.wc_optimize import OptimizeStructure

import numpy as np
from generate_inputs import *

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
    data_sets_object = ForceSets(data_sets=data_sets)

    #data_sets_object = ArrayData()
    #for i, first_atoms in enumerate(data_sets['first_atoms']):
    #    data_sets_array = np.array([first_atoms['direction'], first_atoms['number'], first_atoms['displacement']])
    #    data_sets_object.set_array('data_sets_{}'.format(i), data_sets_array)

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
    # Build data_sets from forces of supercells with displacments

    data_sets = kwargs.pop('data_sets')

    force_sets = ForceSets(data_sets=data_sets.get_data_sets())

    forces = []
    for i in range(data_sets.get_number_of_displacements()):
        forces.append(kwargs.pop('forces_{}'.format(i)).get_array('forces')[0])

    force_sets.set_forces(forces)

    return {'force_sets': force_sets}


    force_sets = ArrayData()
    for i in data_set.get_arraynames():
        force_array = kwargs.pop(i.replace('data_sets', 'forces')).get_array('forces')[0]
        data_set_array = np.array([data_set.get_array(i)[0], data_set.get_array(i)[1], data_set.get_array(i)[2], force_array])
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
    # print 'function',kwargs

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_sets = kwargs.pop('force_sets')

 # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'])

    phonon.generate_displacements(distance=phonopy_input['distance'])

 # Build data_sets from forces of supercells with displacments
    phonon.set_displacement_dataset(force_sets.get_force_sets())
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
    thermal_properties.set_array('free_energy', free_energy * normalization_factor)
    thermal_properties.set_array('entropy', entropy * normalization_factor)
    thermal_properties.set_array('cv', cv * normalization_factor)

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
        spec.input("optimize", valid_type=Bool, required=False, default=Bool(True))
        spec.input("pressure", valid_type=Float, required=False, default=Float(0.0))
      #  spec.dynamic_input("optimize")

        #spec.outline(cls.create_displacement_calculations,
        #             if_(cls.remote_phonopy)(cls.get_force_constants_remote,
        #                                     cls.collect_phonopy_data).else_(
        #                 cls.get_force_constants))

        spec.outline(cls.optimize, cls.create_displacement_calculations, cls.get_force_constants)
        #spec.outline(cls.create_displacement_calculations, cls.get_force_constants_remote, cls.collect_phonopy_data)

    def optimize(self):
        print 'start optimize'
        future = submit(OptimizeStructure,
                        structure=self.inputs.structure,
                        machine=self.inputs.machine,
                        es_settings=self.inputs.es_settings,
                        pressure=self.inputs.pressure,
                        )
        self.ctx._content['optimize'] = load_node(481308)

        #return ToContext(optimized=future)

    def remote_phonopy(self):
        return 'code' in self.inputs.ph_settings.get_dict()

    def create_displacement_calculations(self):
        print 'start displacements'
        print 'test2!', self.ctx._get_dict()

        if 'optimized' in self.ctx:
            structure = self.ctx.optimized.out.optimized_structure
        else:
            structure = self.inputs.structure

        structures = create_supercells_with_displacements_using_phonopy(structure, self.inputs.ph_settings)

        self.ctx.data_sets = structures.pop('data_sets')
        self.ctx.number_of_displacements = len(structures)

############### FOR TESTING ###############
# 1) Load data from nodes
        if True: #For test
            from aiida.orm import load_node
            nodes = [482152, 482154, 482156, 482158]  # LAMMPS
            labels = ['structure_1', 'structure_0', 'structure_3', 'structure_2']
            for pk, label in zip(nodes, labels):
                future = load_node(pk)
                self.ctx._content[label] = future
            return

        calcs = {}
        for label, structure in structures.iteritems():
            print label, structure

            print self.inputs.es_settings.dict.code

            # plugin = self.inputs.code.get_attr('input_plugin')

            JobCalculation, calculation_input = generate_inputs(structure,
                                                                self.inputs.machine,
                                                                self.inputs.es_settings,
                                                                #pressure=self.input.pressure,
                                                                type='forces')

            calculation_input._label = label
            future = submit(JobCalculation, **calculation_input)
            print label, future.pid
            calcs[label] = future

        return ToContext(**calcs)

    def get_force_constants(self):


        #print 'dict', self.ctx._get_dict()
        # wf_inputs = {}
        #for key, calc in self.ctx._get_dict():
        #    if key.startswith('structure_'):
        #        print 'key', key
        #        wf_inputs[key.replace('structure', 'forces')] = calc.out.output_array

        #print wf_inputs

        wf_inputs = {}
        print 'DISP', self.ctx.number_of_displacements
        for i in range(self.ctx.number_of_displacements):
            print 'forces_{}'.format(i), self.ctx.get('structure_{}'.format(i))
            wf_inputs['forces_{}'.format(i)] = self.ctx.get('structure_{}'.format(i)).out.output_array

        wf_inputs['data_sets'] = self.ctx.data_sets
        force_sets = create_forces_set(**wf_inputs)['force_sets']

        print force_sets


        wf_inputs = {}

        wf_inputs['structure'] = self.inputs.structure
        wf_inputs['phonopy_input'] = self.inputs.ph_settings

        wf_inputs['machine'] = self.inputs.machine
        wf_inputs['force_sets'] = self.ctx.data_sets

        phonopy_output = get_force_constants_from_phonopy(**wf_inputs)
        force_constants = phonopy_output['array_data']

        phonon_properties = get_properties_from_phonopy(self.inputs.structure,
                                                       self.inputs.ph_settings,
                                                       force_constants)

        self.out('force_constants', force_constants)
        self.out('phonon_properties', phonon_properties['thermal_properties'])
        self.out('dos', phonon_properties['dos'])

        return

    def get_force_constants_remote(self):
        wf_inputs = {}
        for key, value in self.ctx._get_dict().iteritems():
            if key.startswith('structure_'):
                wf_inputs[key.replace('structure', 'forces')] = value['output_array']

        wf_inputs['data_sets'] = self.ctx.data_sets
        force_sets = create_forces_set(**wf_inputs)['force_sets']

        print force_sets.get_data_sets()
        exit()

        code_label = self.inputs.ph_settings.get_dict()['code']

        JobCalculation, calculation_input = generate_phonopy_params(Code.get_from_string(code_label),
                                                                    self.inputs.structure,
                                                                    self.inputs.ph_settings,
                                                                    self.inputs.machine,
                                                                    force_sets)

        future = submit(JobCalculation, **calculation_input)
        calcs = {'phonopy_results': future}

        return ToContext(**calcs)

    def collect_phonopy_data(self):

        force_constants = self.ctx.phonopy_results['array_data']

        phonon_properties = get_properties_from_phonopy(self.inputs.structure,
                                                        self.inputs.ph_settings,
                                                        force_constants)

        self.out('force_constants', force_constants)
        self.out('phonon_properties', phonon_properties['thermal_properties'])
        self.out('dos', phonon_properties['dos'])

        return


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
    print results

    print results['force_constants'].get_array('force_constants')

    print results['force_constants'].pk
    print results['phonon_properties'].pk
    print results['dos'].pk
