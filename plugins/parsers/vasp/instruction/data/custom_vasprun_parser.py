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
from pymatgen.io import vasp
#
import phonopy.interface.vasp as vasp_phonopy 


__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'

# This file has been modified respect to the original code by Mario Zic

class Custom_vasprun_parserInstruction(BaseInstruction):

    _input_file_list_ = ['vasprun.xml', 'OUTCAR']

    def _parser_function(self):
        """
        Parses the vasprun.xml using the Pymatgen Vasprun function.
        """
        vasp_param = {}  # ParameterData

        parser_warnings = {}  # return non-critical errors

        # extract data
        try: 
            with open(self._out_folder.get_abs_path('OUTCAR'), 'r') as f:
                text = f.readlines()
        except:
            print ('Error opening')   

        try:
            vspr = vasp.Vasprun(self._out_folder.get_abs_path('vasprun.xml'))
            vasp_param['energy'] = vspr.final_energy
#            vasp_param['volume'] = vspr.final_structure.lattice.volume  #Not here!, not necessary

        except Exception, e:
            msg = (
                "Parsing vasprun file in pymatgen failed, "
                "with error message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        # Get forces using phonopy functions
        try:
            force = vasp_phonopy._get_forces_vasprun_xml(
                                 vasp_phonopy._iterparse(self._out_folder.get_abs_path('vasprun.xml'), tag='varray')
                                 )

            vasp_param['atomic_force'] = force.tolist()


        except Exception, e:
            msg = (
                "Processing of extracted data failed, "
                "with error Message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)


        # construct proper output format
        try:
            nodes_list = []
            parameter_data = ParameterData(dict=vasp_param)
            nodes_list.append((
                'vasp_parameters@{}'.format(self.__class__.__name__),
                parameter_data
            ))
        except Exception, e:
            msg = (
                "Failed to create AiiDA data structures "
                "(ParameterData/ArrrayData) from parsed data, "
                "with error message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

#        parser_warnings.setdefault('custom_parser', 'OK') # test

        if not parser_warnings:
            parser_warnings = None

#        print nodes_list
        return nodes_list, parser_warnings
