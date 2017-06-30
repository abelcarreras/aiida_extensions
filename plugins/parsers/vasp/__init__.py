# -*- coding: utf-8 -*-
#import sys  # for devel only
#
#import numpy
#import json
#
from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.common.datastructures import calc_states
#
from aiida.orm.calculation.job.vasp.vasp import VaspCalculation
from aiida.orm.calculation.job.vasp.vasp import ParserInstructionFactory
from aiida.orm.data.parameter import ParameterData

__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'


class VaspParser(Parser):
    """
    This class is the implementation of the Parser class
    for the VASP calculator.
    """
    _outstruct_name = 'output_structure'

    def __init__(self, calculation):
        """
        Initialize the instance of VaspParser
        """
        # check for valid input
        if not isinstance(calculation, VaspCalculation):
            raise OutputParsingError(
                "Input calculation must be a VaspCalculation"
            )
        self._calc = calculation

    def parse_from_calc(self, manual=True, custom_instruct=None):
        """
        Parses the datafolder, stores results.
        """
        from aiida.common.exceptions import InvalidOperation
        from aiida.common import aiidalogger
        from aiida.utils.logger import get_dblogger_extra

        parserlogger = aiidalogger.getChild('vaspparser')
        logger_extra = get_dblogger_extra(self._calc)

        # suppose at the start that the job is successful
        successful = True
        parser_warnings = {}  # for logging non-critical events

        # check that calculation is in the right state
        if not manual:
            state = self._calc.get_state()
            if state != calc_states.PARSING:
                raise InvalidOperation(
                    "Calculation not in {} state".format(calc_states.PARSING)
                )

        # get parser instructions
        # TODO: output parser should NOT interpret the input !!!
        try:
            instruct = self._calc.get_inputs_dict().pop(
                self._calc.get_linkname('settings'))
            instruct = instruct.get_dict()
            instruct = instruct[u'PARSER_INSTRUCTIONS']

##########   Abel Modification to test custom parsers


            if not(isinstance(custom_instruct, type(None))):
                instruct = custom_instruct 
 
##########
            
            # check if structure, data, and error parsers are specified
            # if not append defaults
            itypes = [i['type'] for i in instruct]
            # structure
            if not 'structure' in itypes:
                instruct.append({
                    'instr': 'default_structure_parser',
                    'type': 'structure',
                    'params': {}}
                )
                parser_warnings.setdefault(
                    'Structure parser instruction not found!',
                    'default_structure_parser loaded.'
                )
            # error
            if not 'error' in itypes:
                instruct.append({
                    'instr': 'default_error_parser',
                    'type': 'error',
                    'params': {}}
                )
                parser_warnings.setdefault(
                    'Error parser instruction not found!',
                    'default_error_parser loaded.'
                )
            # output
            if not 'data' in itypes:
                instruct.append({
                    'instr': 'default_vasprun_parser',
                    'type': 'data',
                    'params': {}}
                )
                parser_warnings.setdefault(
                    'Data parser instruction not found!',
                    'default_data_parser_parser loaded.'
                )
        except:
            parser_warnings.setdefault(
                'Parser instructions not found',
                'Default instructions were loaded.'
            )
            # don't crash, load default instructions instead
            instruct = [
                # output
                {
                    'instr': 'default_vasprun_parser',
                    'type': 'data',
                    'params': {}
                },
                # error
                {
                    'instr': 'default_error_parser',
                    'type': 'error',
                    'params': {}
                },
                # structure
                {
                    'instr': 'default_structure_parser',
                    'type': 'structure',
                    'params': {}
                }
            ]

        # select the folder object
        out_folder = self._calc.get_retrieved_node()

        # check what is inside the folder
        list_of_files = out_folder.get_folder_list()

        # === check if mandatory files exist ===
        # default output file should exist
        if not self._calc._default_output in list_of_files:
            successful = False
            parserlogger.error(
                "Standard output file ({}) not found".format(
                    self._calc._default_output
                ),
                extra=logger_extra
            )
            return successful, ()
        # output structure file should exist
        if not self._calc._output_structure in list_of_files:
            successful = False
            parserlogger.error(
                "Output structure file ({}) not found".format(
                    self._calc._output_structure
                ),
                extra=logger_extra
            )
            return successful, ()
        # stderr file should exist
        if not self._calc._SCHED_ERROR_FILE in list_of_files:
            successful = False
            parserlogger.error(
                "STDERR file ({}) not found".format(
                    self._calc._SCHED_ERROR_FILE
                ),
                extra=logger_extra
            )
            return successful, ()

        instr_node_list = []
        errors_node_list = []

        # === execute instructions ===
    #    print instruct
        for instr in instruct:
            # create an executable instruction
            try:
                # load instruction
                itype = instr['type'].lower()
                iname = instr['instr']
                iparams = instr['params']
                ifull_name = "{}.{}".format(itype, iname)

                # append parameters
                if itype == 'error':
                    iparams.setdefault(
                        'SCHED_ERROR_FILE', self._calc._SCHED_ERROR_FILE)
                elif itype == 'structure':
                    iparams.setdefault(
                        'OUTPUT_STRUCTURE', self._calc._output_structure)

                # instantiate
                instr = ParserInstructionFactory(ifull_name)
                instr_exe = instr(
                    out_folder,
                    params=iparams if iparams else None
                )
            except ValueError:
                parser_warnings.setdefault(
                    '{}_instruction'.format(instr),
                    'Invalid parser instruction - could not be instantiated!'
                )
                instr_exe = None

            # execute
            
            if instr_exe:
                try:
                    for item in instr_exe.execute():  # store the results
                        instr_node_list.append(item)
                except Exception as e:
                    print instr, e
            #        parser_warnings['output'].setdefault(  Modified by Abel
                    parser_warnings.setdefault('output',{
                        '{}_instruction'.format(instr),
                        'Failed to execute. Errors: {}'.format(e)
                    })

        # add all parser warnings to the error list
        parser_warnings = ParameterData(dict=parser_warnings)
        errors_node_list.append((
            'parser_warnings', parser_warnings
        ))

        # === save the outputs ===
        new_nodes_list = []

        # save the errors/warrnings
        for item in errors_node_list:
            new_nodes_list.append(item)

        # save vasp data
        if instr_node_list:
            for item in instr_node_list:
                new_nodes_list.append(item)

        return successful, new_nodes_list
