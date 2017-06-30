# -*- coding: utf-8 -*-

# imports here
from aiida.parsers.exceptions import OutputParsingError
#
from aiida.orm.data.parameter import ParameterData
#
from aiida.parsers.plugins.vasp.instruction import BaseInstruction

__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'


# main body below
# this class name intentionally breaks the CamelCaseConvention
# TODO check if calling capitalize() in the BaseFactory is an intended behaviour

class Default_error_parserInstruction(BaseInstruction):

    _dynamic_file_list_ = True

    def _parser_function(self):
        parser_warnings = {}  # for compatibility

        try:
            errfile = self._params['SCHED_ERROR_FILE']
            errfile = self._out_folder.get_abs_path(errfile)
        except KeyError:
            raise OutputParsingError(
                "{} expects the SCHED_ERROR_FILE to "
                "be provided as a parameter.".format(
                    self.__class__.__name__)
            )
        except OSError:
            raise OutputParsingError(
                "SCHED_ERROR_FILE ({}/{}) not found !".format(
                    self._out_folder.get_abs_path(),
                    self._params['SCHED_ERROR_FILE']
                )
            )

        # === parse errors & warnings ===
        # just a text blob --> no way to parse things more cleanly ?!!
        with open(errfile, 'r') as f:
            errors = f.read()
        # use if/else to make things more explicit
        if errors:
            errors = ParameterData(dict={'runtime_errors': errors})
        else:
            errors = ParameterData(dict={'runtime_errors': None})

        # return [('runtime_errors', errors), ('bad.key', errors)], parser_warnings  # for debug
        return [('runtime_errors', errors)], parser_warnings
