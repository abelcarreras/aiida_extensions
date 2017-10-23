# -*- coding: utf-8 -*-

# imports here
from aiida.parsers.exceptions import OutputParsingError
#
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.born_charges import BornChargesData
#
from aiida.parsers.plugins.vasp.instruction import BaseInstruction
#
from pymatgen.io.vasp import Vasprun
import numpy as np

class Born_data_parserInstruction(BaseInstruction):

    _input_file_list_ = ['vasprun.xml']

    def _parser_function(self):
        """
        Parses the vasprun.xml.
        """

        # Get born charges and epsilon
        nodes_list = []

        born_data = BornChargesData()
        try:
            import xml.etree.cElementTree as ET

            tree = ET.parse(self._out_folder.get_abs_path('vasprun.xml'))
            root = tree.getroot()

            for elements in root.iter('varray'):
                if elements.attrib['name'] == 'epsilon':
                    epsilon = []
                    for row in elements:
                        epsilon.append(np.array(row.text.split(), dtype=float))

                    epsilon = np.array(epsilon)
                    born_data.set_epsilon(epsilon)
                    break

            for elements in root.iter('array'):
                try:
                    if elements.attrib['name'] == 'born_charges':
                        born_charges = []
                        for atom in elements[1:]:
                            atom_array = []
                            for c in atom:
                                atom_array.append(np.array(c.text.split(), dtype=float))
                            born_charges.append(atom_array)

                        born_data.set_born_charges(born_charges)

                        break
                except KeyError:
                    pass
        except:
            pass

        try:
            nodes_list.append((
                'born_charges', born_data
            ))

        except Exception, e:
            msg = (
                "Failed to create AiiDA data structures "
                "(Born charges) from parsed data, "
                "with error message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        parser_warnings = None

        return nodes_list, parser_warnings
