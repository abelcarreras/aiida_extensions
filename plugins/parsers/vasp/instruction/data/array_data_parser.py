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
        Parses the vasprun.xml.
        """

        # Get born charges and epsilon
        nodes_list = []
        array_data = ArrayData()

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
                    array_data.set_array('epsilon', epsilon)
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

                        born_charges = np.array(born_charges)
                        array_data.set_array('born_charges', born_charges)

                        break
                except KeyError:
                    pass
        except:
            pass

        # Use pymatgen vasp parser to get atomic forces and stress tensor

        vspr = Vasprun(self._out_folder.get_abs_path('vasprun.xml'), exception_on_bad_xml=False)

        # Get forces using pymatgen
        try:
            forces = np.array([vspr.ionic_steps[-1]['forces']])
            array_data.set_array('forces', forces)

        except Exception, e:
            msg = (
                "Processing forces, "
                "with error Message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        try:
            stress = np.array(vspr.ionic_steps[-1]['stress'])
            array_data.set_array('stress', stress)

        except Exception, e:
            msg = (
                "Processing stress, "
                "with error Message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        try:
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

        parser_warnings = None


        ########
        # BORN DATA
        ########
        # Get born charges and epsilon
        from aiida.orm.data.born_charges import BornChargesData

        born_data = BornChargesData()
        print 'Object created'
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
                    print epsilon
                    born_data.set_epsilon(epsilon)
                    print 'born_charges OK'

                    break

            print born_data
            for elements in root.iter('array'):
                try:
                    if elements.attrib['name'] == 'born_charges':
                        born_charges = []
                        for atom in elements[1:]:
                            atom_array = []
                            for c in atom:
                                atom_array.append(np.array(c.text.split(), dtype=float))
                            born_charges.append(atom_array)

                        print born_charges
                        born_data.set_born_charges(born_charges)
                        print 'born_charges OK'
                        break
                except KeyError:
                    pass
        except:
            pass

        print 'complete born object', born_data

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

        return nodes_list, parser_warnings
