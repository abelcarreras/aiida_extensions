# -*- coding: utf-8 -*-

# imports here
from aiida.common.exceptions import ValidationError
from aiida.common.exceptions import InputValidationError
#
from aiida.orm.data.parameter import ParameterData

__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'


# main body below
class BaseInstruction(object):
    """
    Base class for VASP parser instructions.

    What does it provide/ensure ?!

    What needs to be extended ?!
    """
    _input_file_list_ = []    # used to automatically update the retrieve list
                              # in the VaspCalculation plugin
    _dynamic_file_list_ = False

    def __init__(self, out_folder, params=None):
        super(BaseInstruction, self).__init__()

        # check that the file to parse is well defined
        if not self._dynamic_file_list_:
            if not self._input_file_list_:
                raise NotImplementedError(
                    '_input_file_list_ must be defined or '
                    '_dynamic_file_list set to True!'
                )

        self._out_folder = out_folder
        # check if the output folder exists on execution

        # will be set later by the execute method
        self._data = []  # parsed data container (node list)
        self._errors = {}  # a single ParameterData object
        if params:
            try:
                assert isinstance(params, dict)
                self._params = params
            except:
                raise ValueError(
                    "Parameters need to be a dictionary instance! "
                    "Found: {}".format(type(params))
                )
        else:
            self._params = {}

    def _parser_function(self):
        """
        This is the ONLY function that needs to be overriden by the user.

        Expected output: `[('node_name', ValidAiidaDataStructure), ...]`
        """
        raise NotImplementedError()

    def execute(self):
        # raise appropriate errors if self._out_folder doesn't exist or
        # the required input files are not present
        errors = self._check_input()  # if found, stored in self._errors
        parser_warnings = None

        # try parsing the output
        if not errors:
            # get the data
            try:
                self._data, parser_warnings = self._parser_function()
            except Exception as e:
                self._errors.setdefault('parser_function', "{}".format(e))

            # check parser warnings
            if parser_warnings:
#                print '#2 {}'.format(parser_warnings)
                try:
                    for k in parser_warnings:
                        assert type(parser_warnings[k]) == str
                except:
                    raise ValueError(
                        'Parser function returned invalid parser_warning '
                        'data format. '
                        '(For developers - probably a bad implementation of '
                        'the _parser_function.)'
                    )

                # if ok
                self._errors.setdefault(
                    'parser_function_warnings', parser_warnings)

            # TODO: check for remaining file locks
            # better to deal with this through the documentation
            # Assuming the _parser_function is properly implemented,
            # if sth unexpected occurs during the runtime it will
            # probably not be reproducible anyway

            if self._data:
                self._validate_data()  # enforce the data format
        else:
            self._errors.setdefault(
                "parser_function",
                "Data parsing skiped - input errors found!"
            )

        # === return the results ===
        # attach errors
        errors = ParameterData(dict=self._errors)
        errors = (
            'errors@{}'.format(self.__class__.__name__),
            errors
        )
        self._data.append(errors)

        return self._data

    def _check_input(self):
        # check the whole input in a single scan
        from os import path as osp

        # check if out_folder exists
        if not osp.isdir(self._out_folder.get_abs_path()):
            self._errors.setdefault(
                "input_check_outdir",
                "VASP output folder ({}) not found!".format(
                    self._out_folder.get_abs_path())
            )

        # check if all input files exist
        for file in self._input_file_list_:
            if not osp.isfile(self._out_folder.get_abs_path(file)):
                self._errors.setdefault(
                    "input_check_file_{}".format(file),
                    "Input file not found!"
                )

        # return if any errors were encountered
        if self._errors:
            return True
        else:
            return False

    def _validate_data(self):
        valid_data_types = set([
            'ParameterData', 'StructureData', 'KpointsData', 'ArrayData'])

        for item in self._data:
                k, v = item  # unpack the tuple
                vtype = v.__class__.__name__
                if not vtype in valid_data_types:
                    raise ValueError(
                        '{}'.format(k),
                        'Bad data type found ({}) for {}.'
                        '(For developers - check the _parser_function '
                        'implementation. '
                        'This should NOT have happened!)'.format(vtype, k)
                    )
                # append instruction class name to the item
                idx = self._data.index(item)
                k = self._data[idx][0]
                k += '@{}'.format(self.__class__.__name__)
