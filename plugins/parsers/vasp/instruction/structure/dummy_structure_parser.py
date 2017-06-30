# -*- coding: utf-8 -*-

# imports here
from aiida.parsers.plugins.vasp.instruction import BaseInstruction

__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'


# main body below
# this class name intentionally breaks the CamelCaseConvention
# TODO check if calling capitalize() in the BaseFactory is an intended behaviour
class Dummy_structure_parserInstruction(BaseInstruction):

    _dynamic_file_list_ = True

    def _parser_function(self):
        """
        Use to skip parsing (explicitly).
        """

        nodes_list = []
        parser_warnings = {}  # return non-critical errors

        if not parser_warnings:
            parser_warnings = None

        return nodes_list, parser_warnings
