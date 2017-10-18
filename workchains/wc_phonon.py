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

from aiida.workflows.wc_optimize import OptimizeStructure

import numpy as np
from generate_inputs import *

PwCalculation = CalculationFactory('quantumespresso.pw')
PhonopyCalculation = CalculationFactory('phonopy')


def generate_phonopy_params(code, structure, parameters, machine, force_sets):
    """
    Generate inputs parameters needed to do a remote phonopy calculation
    :param code: AiiDA Code object
    :param structure: AiiDA StructureData Object
    :param parameters: AiiDA ParametersData object containing a dictionary with the data neede to run a phonopy
                       calculation: supercell matrix, primitive matrix, displacement, distance and mesh
    :param machine: AiiDA ParametersData object containing a dictionary with the computational resources information
    :param force_sets: AiiDA ParametersData object containing the collected forces and displacement information
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
    inputs.data_sets = force_sets

    return PhonopyCalculation.process(), inputs


@workfunction
def create_supercells_with_displacements_using_phonopy(structure, phonopy_input):
    """
    Create the supercells with the displacements to use the finite displacements methodology to calculate the
    force constants
    :param structure: AiiDa StructureData object
    :param phonopy_input: AiiDa ParametersData object containing a dictionary with the data needed to run phonopy:
            supercells matrix, primitive matrix and displacement distance.
    :return: dictionary of AiiDa StructureData Objects containing the cells with displacements
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


@workfunction
def get_force_constants_from_phonopy(**kwargs):
    """
    Calculate the force constants using phonopy
    :param kwargs:
    :return: phonopy force constants
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
    :return: phonopy thermal properties and DOS
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

        spec.outline(cls.optimize, cls.create_displacement_calculations, cls.get_force_constants, cls.collect_phonopy_data)
        #spec.outline(cls.create_displacement_calculations, cls.get_force_constants_remote, cls.collect_phonopy_data)

    def optimize(self):
        print 'start optimize'
        future = submit(OptimizeStructure,
                        structure=self.inputs.structure,
                        machine=self.inputs.machine,
                        es_settings=self.inputs.es_settings,
                        pressure=self.inputs.pressure,
                        )
        # For testing
        testing = True
        if testing:
            self.ctx._content['optimize'] = load_node(481308)
            return

        return ToContext(optimized=future)

    def create_displacement_calculations(self):
        print 'create displacements'
        print self.ctx._get_dict()


        if 'optimized' in self.ctx:
            structure = self.ctx.optimized.out.optimized_structure
        else:
            structure = self.inputs.structure

        structures = create_supercells_with_displacements_using_phonopy(structure, self.inputs.ph_settings)

        self.ctx.data_sets = structures.pop('data_sets')
        self.ctx.number_of_displacements = len(structures)

        # Load data from nodes
        testing = False
        if testing:
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

        print 'calculate force constants'

        wf_inputs = {}
        for i in range(self.ctx.number_of_displacements):
            wf_inputs['forces_{}'.format(i)] = self.ctx.get('structure_{}'.format(i)).out.output_array
        wf_inputs['data_sets'] = self.ctx.data_sets

        self.ctx.force_sets = create_forces_set(**wf_inputs)['force_sets']

        if 'code' in self.inputs.ph_settings.get_dict():
            print ('remote phonopy calculation')
            code_label = self.inputs.ph_settings.get_dict()['code']
            JobCalculation, calculation_input = generate_phonopy_params(code=Code.get_from_string(code_label),
                                                                        structure=self.inputs.structure,
                                                                        parameters=self.inputs.ph_settings,
                                                                        machine=self.inputs.machine,
                                                                        force_sets=self.ctx.force_sets)
            future = submit(JobCalculation, **calculation_input)
            print 'phonopy FC calc:', future.pid

            #phonopy_input = {'phonopy_output': future}
            #return ToContext(**phonopy_input)

            return ToContext(phonopy_output=future)
        else:
            print ('local phonopy calculation')
            self.ctx.phonopy_output = get_force_constants_from_phonopy(structure=self.inputs.structure,
                                                                       phonopy_input=self.inputs.ph_settings,
                                                                       machine=self.inputs.machine,
                                                                       force_sets=self.ctx.force_sets)
        return

    def collect_phonopy_data(self):

        # print self.ctx._get_dict()
        print 'collect phonopy data'

        try:
            force_constants = self.ctx.phonopy_output['array_data']
        except TypeError:
            force_constants = self.ctx.phonopy_output.out.array_data

        phonon_properties = get_properties_from_phonopy(self.inputs.structure,
                                                        self.inputs.ph_settings,
                                                        force_constants)

        self.out('force_constants', force_constants)
        self.out('phonon_properties', phonon_properties['thermal_properties'])
        self.out('dos', phonon_properties['dos'])

        return
