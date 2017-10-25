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

from aiida.workflows.wc_phonon import PhononPhonopy, get_path_using_seekpath
from aiida.work.workchain import _If, _While

import numpy as np
from generate_inputs import *


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


@workfunction
def phonopy_gruneisen(phonon_plus, phonon_minus, phonon_origin, ph_settings):
    from phonopy import PhonopyGruneisen

    phonon_plus2 = get_phonon(phonon_plus['final_structure'],
                             phonon_plus['force_constants'],
                             ph_settings)

    phonon_minus2 = get_phonon(phonon_minus['final_structure'],
                              phonon_minus['force_constants'],
                              ph_settings)

    phonon_origin2 = get_phonon(phonon_origin['final_structure'],
                               phonon_origin['force_constants'],
                               ph_settings)

    gruneisen = PhonopyGruneisen(phonon_origin2,  # equilibrium
                                 phonon_plus2,  # plus
                                 phonon_minus2)  # minus

    gruneisen.set_mesh(ph_settings.dict.mesh, is_gamma_center=False, is_mesh_symmetry=True)

    # BAND STRUCTURE
    band_structure = get_path_using_seekpath(phonon_origin['final_structure'])
    gruneisen.set_band_structure(band_structure.get_bands(),
                                 band_structure.get_number_of_points())
    #gruneisen.set_band_structure(band_structure.get_bands(), 51)

    band_structure.set_band_structure_gruneisen(gruneisen.get_band_structure())

    # mesh
    mesh = gruneisen.get_mesh()
    frequencies_mesh = np.array(mesh.get_frequencies())
    gruneisen_mesh = np.array(mesh.get_gruneisen())

    # build mesh
    mesh_array = ArrayData()
    mesh_array.set_array('frequencies', frequencies_mesh)
    mesh_array.set_array('gruneisen', gruneisen_mesh)

    return {'band_structure': band_structure, 'mesh': mesh_array}


class GruneisenPhonopy(WorkChain):
    """
    Workchain to calculate the mode Gruneisen parameters

    """
    @classmethod
    def define(cls, spec):
        super(GruneisenPhonopy, cls).define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("machine", valid_type=ParameterData)
        spec.input("ph_settings", valid_type=ParameterData)
        spec.input("es_settings", valid_type=ParameterData)
        # Optional arguments
        spec.input("optimize", valid_type=Bool, required=False, default=Bool(True))
        #spec.input("pressure", valid_type=Float, required=False, default=Float(0.0))
        spec.input("stress_displacement", valid_type=Float, required=False, default=Float(1e-2))

        spec.outline(cls.create_unit_cell_expansions, cls.calculate_gruneisen)

    def create_unit_cell_expansions(self):

        print 'start create cell expansions'

        # For testing
        testing = False
        if testing:
            self.ctx._content['plus'] = load_node(2381)
            self.ctx._content['origin'] = load_node(2378)
            self.ctx._content['minus'] = load_node(2384)
            return

        calcs = {}
        for expansions in {'plus': float(self.inputs.stress_displacement),
                           'origin': 0.0,
                           'minus': -float(self.inputs.stress_displacement)}.items():

            future = submit(PhononPhonopy,
                            structure=self.inputs.structure,
                            machine=self.inputs.machine,
                            ph_settings=self.inputs.ph_settings,
                            es_settings=self.inputs.es_settings,
                            pressure=Float(expansions[1]),
                            optimize=Bool(True)
                            )

            calcs[expansions[0]] = future
            print ('phonon workchain: {} {}'.format(expansions[0], future.pid))

        return ToContext(**calcs)

    def calculate_gruneisen(self):

        print 'calculate gruneisen'
        print self.ctx.plus, self.ctx.minus, self.ctx.origin

        gruneisen_results = phonopy_gruneisen(phonon_plus=self.ctx.plus,
                                              phonon_minus=self.ctx.minus,
                                              phonon_origin=self.ctx.origin,
                                              ph_settings=self.inputs.ph_settings)

        self.out('band_structure', gruneisen_results['band_structure'])
        self.out('mesh', gruneisen_results['mesh'])