from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.force_constants import ForceConstantsData

from aiida.common.exceptions import InputValidationError
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty

import numpy as np

# for future use
def get_BORN_txt(structure, parameters, nac_data, symprec=1.e-5):

    from phonopy.structure.cells import get_primitive, get_supercell
    from phonopy.structure.symmetry import Symmetry
    from phonopy.interface import get_default_physical_units
    from phonopy.structure.atoms import Atoms as PhonopyAtoms

    born_charges = nac_data.get_array('born_charges')
    epsilon = nac_data.get_array('epsilon')

    print ('inside born parameters')
    pmat = parameters['primitive']
    smat = parameters['supercell']

    ucell = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                         positions=[site.position for site in structure.sites],
                         cell=structure.cell)

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

    born_txt = ('{}\n'.format(factor))
    for num in epsilon.flatten():
        born_txt += ('{0:4.8f}'.format(num))
    born_txt += ('\n')

    for atom in reduced_borns:
        for num in atom:
            born_txt += ('{0:4.8f}'.format(num))
        born_txt += ('\n')
    born_txt += ('{}\n'.format(factor))

    return born_txt

def structure_to_poscar(structure):

    types = [site.kind_name for site in structure.sites]
    atom_type_unique = np.unique(types, return_index=True)
    sort_index = np.argsort(atom_type_unique[1])
    elements = np.array(atom_type_unique[0])[sort_index]
    elements_count= np.diff(np.append(np.array(atom_type_unique[1])[sort_index], [len(types)]))

    poscar = '# VASP POSCAR generated using aiida workflow '
    poscar += '\n1.0\n'
    cell = structure.cell
    for row in cell:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*row)
    poscar += ' '.join([str(e) for e in elements]) + '\n'
    poscar += ' '.join([str(e) for e in elements_count]) + '\n'
    poscar += 'Cartesian\n'
    for site in structure.sites:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*site.position)

    return poscar


def parameters_to_input_file(parameters_object):
    parameters = parameters_object.get_dict()
    input_file = 'DIM = {} {} {}\n'.format(*np.diag(parameters['supercell']))
    input_file += 'PRIMITIVE_AXIS = {} {} {}  {} {} {}  {} {} {}'.format(
        *np.array(parameters['primitive']).reshape((1, 9))[0])
    input_file += 'MESH = {} {} {}\n'.format(*np.diag(parameters['mesh']))
    return input_file


class PhonopyCalculation(JobCalculation):
    """
    A basic plugin for calculating force constants using Phonopy.

    Requirement: the node should be able to import phonopy
    """

    def _init_internal_params(self):
        super(PhonopyCalculation, self)._init_internal_params()

        self._INPUT_FILE_NAME = 'phonopy.conf'
        self._INPUT_CELL = 'POSCAR'
        self._INPUT_FORCE_SETS = 'FORCE_SETS'
        self._INPUT_NAC = 'BORN'

        self._OUTPUT_FILE_NAME = 'FORCE_CONSTANTS'
        self._default_parser = "phonopy"

    @classproperty
    def _use_methods(cls):
        """
        Additional use_* methods for the namelists class.
        """
        retdict = JobCalculation._use_methods
        retdict.update({
            "parameters": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'parameters',
                'docstring': ("Use a node that specifies the phonopy parameters "
                              "for the namelists"),
            },
            "force_constants": {
                'valid_types': ForceConstantsData,
                'additional_parameter': None,
                'linkname': 'data_sets',
                'docstring': ("Use a node that specifies the data_sets "
                              "for the namelists"),
            },
            "structure": {
                'valid_types': StructureData,
                'additional_parameter': None,
                'linkname': 'structure',
                'docstring': "Use a node for the structure",
            },
            "nac_data": {
                'valid_types': StructureData,
                'additional_parameter': None,
                'linkname': 'nac_data',
                'docstring': "Use a node for the Non-analitical corrections data",
            },
        })
        return retdict

    def _prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """

        try:
            parameters_data = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            pass
            # raise InputValidationError("No parameters specified for this "
            #                           "calculation")
        if not isinstance(parameters_data, ParameterData):
            raise InputValidationError("parameters is not of type "
                                       "ParameterData")

        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError("no structure is specified for this calculation")

        try:
            force_constants = inputdict.pop(self.get_linkname('force_constants'))
        except KeyError:
            raise InputValidationError("no data_set is specified for this calculation")

        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError("no code is specified for this calculation")

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        # =================== prepare the python input files =====================

        cell_txt = structure_to_poscar(structure)
        input_txt = parameters_to_input_file(parameters_data)
        force_sets_txt = force_constants.get_phonopy_formatted_txt()

        # =========================== dump to file =============================

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        cell_filename = tempfolder.get_abs_path(self._INPUT_CELL)
        with open(cell_filename, 'w') as infile:
            infile.write(cell_txt)

        force_sets_filename = tempfolder.get_abs_path(self._INPUT_FORCE_CONSTANTS)
        with open(force_sets_filename, 'w') as infile:
            infile.write(force_sets_txt)

        # ============================ calcinfo ================================

        local_copy_list = []
        remote_copy_list = []
        #    additional_retrieve_list = settings_dict.pop("ADDITIONAL_RETRIEVE_LIST",[])

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        # Empty command line by default
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        # Retrieve files
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self._OUTPUT_FILE_NAME)

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = [self._INPUT_FILE_NAME, '--readfc', '-dos']
        codeinfo.code_uuid = code.uuid
        codeinfo.withmpi = False
        calcinfo.codes_info = [codeinfo]
        return calcinfo