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


class Array_data_parserInstruction(BaseInstruction):

    _input_file_list_ = ['vasprun.xml', 'OUTCAR']

    def _parser_function(self):
        """
        Parses the vasprun.xml using the Pymatgen Vasprun function.
        """
        print ('INFUNCTION')
        vasp_param = {}  # ParameterData

        parser_warnings = {}  # return non-critical errors

        print ('Opening')
        # extract data
        try: 
            with open(self._out_folder.get_abs_path('OUTCAR'), 'r') as f:
                text = f.readlines()
        except:
            print ('Error opening')   

#        try:
#            vspr = vasp.Vasprun(self._out_folder.get_abs_path('vasprun.xml'))
#            vasp_param['energy'] = vspr.final_energy
#            vasp_param['volume'] = vspr.final_structure.lattice.volume  #Not here!, not necessary

#        except Exception, e:
#            msg = (
#                "Parsing vasprun file in pymatgen failed, "
#                "with error message:\n>> {}".format(e)
#            )
#            raise OutputParsingError(msg)


        #Get forces using phonopy functions        
        try:
            forces = vasp_phonopy._get_forces_vasprun_xml(
                                 vasp_phonopy._iterparse(self._out_folder.get_abs_path('vasprun.xml'), tag='varray')
                                 )
            import numpy as np
            forces = np.array([forces]) 
  #          vasp_param['atomic_force'] = force.tolist()


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
            array_data = ArrayData()
            array_data.set_array('forces', forces)

    #        nodes_list.append((
    #            'output_parameters',
    #            array_data
    #        ))

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

#        parser_warnings.setdefault('custom_parser', 'OK') # test

        if not parser_warnings:
            parser_warnings = None

#        print nodes_list
        return nodes_list, parser_warnings
