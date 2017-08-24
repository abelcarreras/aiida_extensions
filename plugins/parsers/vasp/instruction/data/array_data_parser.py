# -*- coding: utf-8 -*-

# imports here
from aiida.parsers.exceptions import OutputParsingError
#
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
#
from aiida.parsers.plugins.vasp.instruction import BaseInstruction
#
from pymatgen.io.vasp import Vasprun
import numpy as np

class Array_data_parserInstruction(BaseInstruction):

    _input_file_list_ = ['vasprun.xml', 'POTCAR']

    def _parser_function(self):
        """
        Parses the vasprun.xml using the Pymatgen Vasprun function.
        """

        parser_warnings = {}  # return non-critical errors

        vspr = Vasprun(self._out_folder.get_abs_path('vasprun.xml'), exception_on_bad_xml=False)

        # Get forces using pymatgen
        try:
            forces = np.array([vspr.ionic_steps[-1]['forces']])

        except Exception, e:
            msg = (
                "Processing forces, "
                "with error Message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        try:
            stress = np.array(vspr.ionic_steps[-1]['stress'])

        except Exception, e:
            msg = (
                "Processing stress, "
                "with error Message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        try:
            nodes_list = []
            array_data = ArrayData()
            array_data.set_array('forces', forces)
            array_data.set_array('stress', stress)

            nodes_list.append((
                'output_array', array_data
            ))
        except Exception, e:
            msg = (
                "Failed to create AiiDA data structures "
                "(ParameterData/ArrrayData) from parsed data, "
                "with error message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        if not parser_warnings:
            parser_warnings = None

#        print nodes_list
        return nodes_list, parser_warnings
