# -*- coding: utf-8 -*-
from shutil import copyfile
import numpy as np
#
#from aiida import get_file_header
from aiida.common.utils import classproperty
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError
#
from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.array.kpoints import ArrayData
#
from pymatgen.io import vasp as vaspio

__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'


def write_poscar(poscar, file='POSCAR'):
    poscar_dict = poscar.as_dict()

    types = [site['species'][0]['element'] for site in poscar_dict['structure']['sites']]
    atom_type_unique = np.unique(types, return_index=True)
    sort_index = np.argsort(atom_type_unique[1])
    elements_count= np.diff(np.append(np.array(atom_type_unique[1])[sort_index], [len(types)]))

    print poscar_dict
    poscar_txt = poscar_dict['comment']
    poscar_txt += '\n1.0\n'
    cell = poscar_dict['structure']['lattice']['matrix']
    for row in cell:
        poscar_txt += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*row)

    poscar_txt += ' '.join(np.unique([site['species'][0]['element'] for site in poscar_dict['structure']['sites']])) + '\n'
    print  [site['species'][0]['element'] for site in poscar_dict['structure']['sites']]
    poscar_txt += ' '.join([str(e) for e in elements_count]) + '\n'

    poscar_txt += str(len(poscar_dict['structure']['sites'])) + '\n'
    poscar_txt += 'Cartesian\n'
    for site in poscar_dict['structure']['sites']:
        poscar_txt += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*site['xyz'])

    print poscar_txt

def errmsg(key):
    """
    Returns an error message string which can then be easily formated.

    :key: dictionary key correspnding to the error message we want returned
    """
    d = {
        'not_specified': (
            "No {} parameters specified for this calculation;"
        ),
        'wrong_class': (
            "{} object is not of type {};"
        ),
        'aiida2vasp': (
            "Failed to instantiate a pymatgen {} object "
            "from the aiida representation; "
            "Error message: {}"
        ),
        'unexpected': (
            "Unespected crash while {}; "
            "Error message: {}"
        )
    }

    try:
        return d[key]
    except:
        raise KeyError(
            'Requested error message key ({}) does not exist.'
        ).format(key)


def ParserInstructionFactory(module):
    """
    Return a suitable VASP parser utility function.
    """
    from aiida.common.pluginloader import BaseFactory
    from aiida.parsers.plugins.vasp.instruction import BaseInstruction
    try:
        return BaseFactory(
            module,
            BaseInstruction,
            'aiida.parsers.plugins.vasp.instruction',
            suffix='Instruction'
        )
    except:
        raise ValueError()


# simple utility functions, should facilitate (de)serialization
# of POSCAR
def assemble_poscar(
        structure,
        structure_extras=None
        ):

    try:

        try:
            structure = structure.get_pymatgen_structure()
            print structure.lattice.matrix
        except Exception as e:
            msg = errmsg('aiida2vasp').format('structure', e)
            raise ValueError(msg)

        if structure_extras:
            # predictor_corrector
            try:
                predictor_corrector = structure_extras.get_array(
                    'predictor_corrector'
                )
            except KeyError:
                predictor_corrector = None
            # velocities
            try:
                velocities = structure_extras.get_array(
                    'velocities'
                )
            except KeyError:
                velocities = None
            # selective_dynamics
            try:
                selective_dynamics = structure_extras.get_array(
                    'selective_dynamics'
                )
            except KeyError:
                selective_dynamics = None
        else:
            predictor_corrector = None
            velocities = None
            selective_dynamics = None

        # creating new POSCAR object
        poscar = vaspio.Poscar(
            structure,
            selective_dynamics=selective_dynamics,
            velocities=velocities,
            predictor_corrector=predictor_corrector)
        # set comment line

        tmp = poscar.comment
        poscar.comment = (
            "# {}, Automatically Generated by AiiDA VASP Plugin".format(tmp))

    except Exception as e:
        msg = "Failed to assemble POSCAR with error message: {}".format(e)
        raise ValueError(msg)

    return poscar


def disassemble_poscar(poscar):

    try:
        poscar_struct = StructureData(pymatgen_structure=poscar.structure)
        structure_extras = ArrayData()

        opt = False
        # optional parameters
        if poscar.predictor_corrector:
            opt = True
            structure_extras.set_array(
                'predictor_corrector', np.array(poscar.predictor_corrector))

        if poscar.selective_dynamics:
            opt = True
            structure_extras.set_array(
                'selective_dynamics', np.array(poscar.selective_dynamics))

        if poscar.velocities:
            opt = True
            structure_extras.set_array(
                'velocities', np.array(poscar.velocities))

    except Exception as e:
        msg = (
            "Failed to disassemble the POSCAR object "
            "with error message: {}".format(e)
        )
        raise ValueError(msg)

    return {
        'structure': poscar_struct,
        'structure_extras': structure_extras if opt else None
    }


class VaspCalculation(JobCalculation):
    """
    Plugin for performing VASP calculations.

    Requirements: pymatgen
    """

    def _init_internal_params(self):
        super(VaspCalculation, self)._init_internal_params()

        # input files -- currently supported
        self._incar = 'INCAR'
        self._poscar = 'POSCAR'
        self._potcar = 'POTCAR'  # partial -- pot_map not tested
        self._kpoints = 'KPOINTS'  # partial -- only autom. generation

        # output files
        self._output_structure = "CONTCAR"
        self._default_output = 'vasprun.xml'

        # parser -- only one parser needed --
        # customization of the parsing proccess
        # should be done using the output parser instructions,
        # specified via the settings node
        self._default_parser = 'vasp'

    @classproperty
    def _use_methods(cls):
        """
        Additional use_* methods for the namelists class.
        """
        retdict = JobCalculation._use_methods
        retdict.update({
            "incar": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'incar',
                'docstring': (
                    "Use a node that specifies the pymatgen incar "
                    "file"
                ),
            },
            "structure": {
                'valid_types': StructureData,
                'additional_parameter': None,
                'linkname': 'structure',
                'docstring': "Use a node for the structure.",
            },
            "potcar": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'potcar',
                'docstring': (
                    "Use a node that specifies the pymatgen potcar "
                    "file"
                ),
            },
            # NOTE: see what happens in pymatgen when we specify kpoints
            # by hand ?!!
            # Do we need to forsee an additonal raw_kpoints object ?
            ## TODO: switch to AiiDA kpoints format
            "kpoints": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'kpoints',
                'docstring': ("Use a node that specifies the kpoints"),
            },
            "settings": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'settings',
                'docstring': (
                    "Use a node that specifies the extra information "
                    "to be used by the calculation"
                ),
            },
            # optional data
            "structure_extras": {
                'valid_types': ArrayData,
                'additional_parameter': None,
                'linkname': 'structure_extras',
                'docstring': (
                    "Node for additional arrays that can be found in "
                    "the CONTCAR file.")
            },
            "chgcar": {
                'valid_types': SinglefileData,
                'additional_parameter': None,
                'linkname': 'chgcar',
                'docstring': (
                    "Node for CHGCAR file.")
            },
            "wavecar": {
                'valid_types': SinglefileData,
                'additional_parameter': None,
                'linkname': 'wavecar',
                'docstring': (
                    "Node for WAVECAR file.")
            }
            })
        return retdict

    ### Methods for importing Pymatgen data
    def import_poscar(self, poscar):
        """
            **Method:** Takes Pymatgen Poscar object and internally sets:
            structure and structure_extras nodes.

            *Input:*
            :poscar: Pymatgen Poscar object.
        """
        try:
            assert type(poscar) == vaspio.Poscar

            poscar_parts = disassemble_poscar(poscar)
            struct = poscar_parts['structure']
            extras = poscar_parts['structure_extras']

            self.use_structure(struct)
            if extras:
                self.use_structure_extras(extras)
        except:
            print(
                "Invalid input type ({})!"
                "Please provide a valid Pymatgen Poscar object.").format(
                type(poscar))

    def import_incar(self, incar):
        """
            **Method:** Takes Pymatgen Incar object and internally sets
            incar node.

            *Input:*
            :incar: Pymatgen Incar object.
        """
        try:
            assert type(incar) == vaspio.Incar
            incar = ParameterData(dict=incar.as_dict())
            self.use_incar(incar)
        except:
            print(
                "Invalid input type ({})!"
                "Please provide a valid Pymatgen Incar object.").format(
                type(incar))

    def import_kpoints(self, kpoints):
        """
            **Method:** Takes Pymatgen Kpoints object and internally sets
            kpoints node.

            *Input:*
            :kpoints: Pymatgen Kpoints object.
        """
        try:
            assert type(kpoints) == vaspio.Kpoints
            kpoints = ParameterData(dict=kpoints.as_dict())
            self.use_kpoints(kpoints)
        except:
            print(
                "Invalid input type ({})!"
                "Please provide a valid Pymatgen Kpoints object.").format(
                type(kpoints))

    def import_potcar(self, potcar):
        """
            **Method:** Takes Pymatgen Potcar object and internally sets
            potcar node.

            *Input:*
            :kpoints: Pymatgen Potcar object.
        """
        try:
            assert type(potcar) == vaspio.Potcar
            potcar = ParameterData(dict=potcar.as_dict())
            self.use_potcar(potcar)
        except:
            print(
                "Invalid input type ({})!"
                "Please provide a valid Pymatgen Potcar object.").format(
                type(potcar))

    ### Submission
    def _prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """

        ## TODO:
        # - still need to see what happens with the kpoints object if
        #   specified by hand

        # === code ===
        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError(errmsg('not_specified').format('CODE'))

        # === mandatory input files ===
        # INCAR
        try:
            incar = inputdict.pop(self.get_linkname('incar'))
        except KeyError:
            raise InputValidationError(
                errmsg('not_specified').format('INCAR')
            )
        if not isinstance(incar, ParameterData):
            raise InputValidationError(
                errmsg('wrong_class').format('INCAR', 'ParameterData')
            )

        try:
            incar = vaspio.Incar.from_dict(incar.get_dict())
        except Exception as e:
            msg = errmsg('aiida2vasp').format('INCAR', e)
            raise InputValidationError(msg)

        # KPOINTS -- for now just supports automatically genrated mesh
        try:
            kpoints = inputdict.pop(self.get_linkname('kpoints'))
        except KeyError:
            raise InputValidationError(
                errmsg('not_specified').format('KPOINTS')
            )
        if not isinstance(kpoints, ParameterData):
            raise InputValidationError(
                errmsg('wrong_class').format('KPOINTS', 'ParameterData')
            )
        try:
            kpoints = vaspio.Kpoints.from_dict(kpoints.get_dict())
        except Exception as e:
            msg = errmsg('aiida2vasp').format('KPOINTS', e)
            raise InputValidationError(msg)

        # POTCAR -- pot_map not tested
        try:
            potcar = inputdict.pop(self.get_linkname('potcar'))
        except KeyError:
            raise InputValidationError(
                errmsg('not_specified').format('POTCAR')
            )
        if not isinstance(potcar, ParameterData):
            raise InputValidationError(
                errmsg('wrong_class').format('POTCAR', 'ParameterData')
            )
        try:
            potcar = vaspio.Potcar.from_dict(potcar.get_dict())
        except Exception as e:
            msg = errmsg('aiida2vasp').format('POTCAR', e)
            raise InputValidationError(msg)

        # SETTINGS
        try:
            settings = inputdict.pop(self.get_linkname('settings'))
        except KeyError:
            raise InputValidationError(
                errmsg('not_specified').format('settings')
            )
        if not isinstance(settings, ParameterData):
            raise InputValidationError(
                errmsg('wrong_class').format('settings', 'ParameterData')
            )
        # SETTINGS - PARSER INSTRUCTIONS
        instr_deps = set()  # instructon dependencies
        instruct = settings.get_dict().get(u'PARSER_INSTRUCTIONS', None)
        if instruct:
            for i in instruct:
                try:
                    assert isinstance(i, dict)
                except:
                    raise InputValidationError(
                        errmsg('wrong_class').format(
                            u'PARSER_INSTRUCTIONS', 'Dict')
                    )
                # instruction name
                try:
                    assert 'instr' in i
                except:
                    raise InputValidationError(
                        errmsg('not_specified').format(
                            "instruction name ('instr')")
                    )
                # instruction type
                try:
                    assert 'type' in i
                except:
                    raise InputValidationError(
                        errmsg('not_specified').format(
                            "instruction type ('type')")
                    )
                try:
                    assert i['type'].lower() in ('data', 'error', 'structure')
                except:
                    raise InputValidationError(
                        'Bad instruction type ({}) found!'.format(
                            i['type'])
                    )
                # parameters
                try:
                    assert 'params' in i
                except:
                    raise InputValidationError(
                        errmsg('not_specified').format(
                            "instruction parameters ('params')")
                    )
                try:
                    assert isinstance(i['params'], dict)
                except:
                    raise InputValidationError(
                        errmsg('wrong_class').format(
                            "Instruction parameters ('params')", 'Dict')
                    )
                # get the input files list (static instruction dependencies)
                try:
                    itype = i['type']
                    iname = i['instr']
                    ifull_name = "{}.{}".format(itype, iname)
                    instr = ParserInstructionFactory(ifull_name)

                    if not instr._dynamic_file_list_:  # is static
                        for file in instr._input_file_list_:
                            instr_deps.add(file)
                except Exception as e:
                    raise InputValidationError((
                        "Failed to get instruction dependencies for {}."
                        "Error: {}.").format(i['instr'], e)
                    )

        else:
            # output parser default behaviour will be applied
            pass

        # POSCAR & STRUCTURE
        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError(
                errmsg('not_specified').format('structure')
            )
        if not isinstance(structure, StructureData):
            raise InputValidationError(
                errmsg('wrong_class').format('structure', 'StructureData')
            )
        # here we look for the optional data associated with the POSCAR
        # predictor_corrector, selective_dynamics, velocities
        try:
            structure_extras = inputdict.pop(
                self.get_linkname('structure_extras'), None
            )
        except KeyError:
            structure_extras = None  # optional parameters
        if structure_extras:
            if not isinstance(structure_extras, ArrayData):
                raise InputValidationError(
                    errmsg('wrong_class').format(
                        'structure_extras', 'ArrayData')
                )
        #
        try:
            poscar = assemble_poscar(
                structure,
                structure_extras=structure_extras
            )
        except Exception as e:
            msg = errmsg('aiida2vasp').format('POSCAR', e)
            raise InputValidationError(msg)
        #
        # CHGCAR
        try:
            chgcar = inputdict.pop(
                self.get_linkname('chgcar'), None
            )
        except KeyError:
            chgcar = None  # optional parameters
        if chgcar:
            if not isinstance(chgcar, SinglefileData):
                raise InputValidationError(
                    errmsg('wrong_class').format(
                        'chgcar', 'SinglefileData')
                )
            try:
                chgcar = chgcar.get_file_abs_path()
                chgcar = vaspio.Chgcar.from_file(chgcar)
            except Exception as e:
                msg = errmsg('aiida2vasp').format('CHGCAR', e)
                raise InputValidationError(msg)
        #
        # WAVECAR
        try:
            wavecar = inputdict.pop(
                self.get_linkname('wavecar'), None
            )
        except KeyError:
            wavecar = None  # optional parameters
        if wavecar:
            if not isinstance(wavecar, SinglefileData):
                raise InputValidationError(
                    errmsg('wrong_class').format(
                        'wavecar', 'SinglefileData')
                )
            # wavecar doesn't have a pymatgen object

        # === set up calculation dir ===
        try:
            write_poscar(poscar, file=tempfolder.get_abs_path('POSCAR'))
            incar.write_file(tempfolder.get_abs_path('INCAR'))
            poscar.write_file(tempfolder.get_abs_path('POSCAR'))
            potcar.write_file(tempfolder.get_abs_path('POTCAR'))
            kpoints.write_file(tempfolder.get_abs_path('KPOINTS'))
            if chgcar:
                chgcar.write_file(tempfolder.get_abs_path('CHGCAR'))
            if wavecar:
                # wavecar is a SinglefileData object
                dest = tempfolder.get_abs_path('WAVECAR')
                src = wavecar.get_file_abs_path()
                copyfile(src, dest)
        except Exception as e:
            msg = (
                "Failed to write the input files to the sandbox, "
                "with error message {}".format(e)
            )
            raise IOError(msg)

        # === calc & code info ===
        settings_dict = settings.get_dict() if settings else {}

        local_copy_list = []  # see what are these two ?!!
        remote_copy_list = []
        additional_retrieve_list = settings_dict.pop(
            "ADDITIONAL_RETRIEVE_LIST", []
        )
        # append instruction dependencies
        for item in instr_deps:
            if not item in additional_retrieve_list:
                additional_retrieve_list.append(item)

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        codeinfo = CodeInfo()
        # Empty command line by default
        codeinfo.cmdline_params = settings_dict.pop('CMDLINE', [])
        codeinfo.stdout_name = None  # self._default_output
        codeinfo.code_uuid = code.uuid
        #
        calcinfo.codes_info = [codeinfo]

        # Retrieve files
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self._default_output)
        calcinfo.retrieve_list.append(self._output_structure)
        calcinfo.retrieve_list += additional_retrieve_list

        return calcinfo
