# -*- coding: utf-8 -*-

# imports here
from aiida.parsers.exceptions import OutputParsingError
#
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
#
from aiida.orm.calculation.job.vasp.vasp import disassemble_poscar  # fnc
#
from aiida.parsers.plugins.vasp.instruction import BaseInstruction
#
from pymatgen.io import vasp

__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'


# main body below
# this class name intentionally breaks the CamelCaseConvention
# TODO check if calling capitalize() in the BaseFactory is an intended behaviour
class Default_structure_parserInstruction(BaseInstruction):

    _dynamic_file_list_ = True

    def _parser_function(self):
        """
        Parses CONTCAR/POSCAR file.

        :output_structure: name of the structure file; must be specified in
        parameters list.
        """
        parser_warnings = {}  # return non-critical errors
        nodes_list = []

        try:
            ofile = self._params['OUTPUT_STRUCTURE']
            out_structure = self._out_folder.get_abs_path(ofile)
            out_structure = vasp.Poscar.from_file(out_structure)
            out_structure = disassemble_poscar(out_structure)
        except KeyError:
            raise OutputParsingError(
                "Output structure file was not specified in params!"
            )
        except OSError:
            raise OutputParsingError(
                "Output structure file ({}/{}) not found!".format(
                    self.out_folder.get_abs_path(),
                    ofile
                )
            )
        except Exception as e:
            raise OutputParsingError(
                "Parsing of output structure failed! "
                "Error: {}".format(e)
            )

        for item in out_structure:
            nodes_list.append((item, out_structure[item]))

        if not nodes_list:
            parser_warnings.setdefault(
                'Returning empty node list.',
                'Parsing of the output structure may have quietly failed!'
            )

        if not parser_warnings:
            parser_warnings = None

        return nodes_list, parser_warnings
