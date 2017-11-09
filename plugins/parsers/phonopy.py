from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm.data.array import ArrayData
from aiida.orm.data.force_constants import ForceConstantsData
from aiida.orm.data.phonon_dos import PhononDosData

from aiida.orm.data.parameter import ParameterData

import numpy as np


def parse_FORCE_CONSTANTS(filename):
    fcfile = open(filename)
    num = int((fcfile.readline().strip().split())[0])
    force_constants = np.zeros((num, num, 3, 3), dtype=float)
    for i in range(num):
        for j in range(num):
            fcfile.readline()
            tensor = []
            for k in range(3):
                tensor.append([float(x) for x in fcfile.readline().strip().split()])
            force_constants[i, j] = np.array(tensor)
    return force_constants


def parse_partial_DOS(filename, structure):
    partial_dos = np.loadtxt(filename)

    dos = PhononDosData(frequencies=partial_dos[0],
                        dos=np.sum(partial_dos[:, 1:], axis=1),
                        partial_dos=partial_dos[:, 1:].T,
                        atom_labels=[site.kind_name for site in structure.sites])

    return dos


def parse_thermal_properties(filename):
    import yaml
    temperature = []
    free_energy = []
    entropy = []
    cv = []

    with open(filename, 'r') as stream:
        thermal_properties = dict(yaml.load(stream))
        for tp in thermal_properties['thermal_properties']:
            temperature.append(tp['temperature'])
            entropy.append(tp['entropy'])
            free_energy.append(tp['free_energy'])
            cv.append(tp['heat_capacity'])

    tp_object = ArrayData()
    tp_object.set_array('temperature', np.array(temperature))
    tp_object.set_array('free_energy', np.array(free_energy))
    tp_object.set_array('entropy', np.array(entropy))
    tp_object.set_array('heat_capacity', np.array(cv))

    return tp_object

class PhonopyParser(Parser):
    """
    Parser the FORCE_CONSTANTS file of phonopy.
    """

    def __init__(self, calc):
        """
        Initialize the instance of PhonopyParser
        """
        super(PhonopyParser, self).__init__(calc)

    def parse_with_retrieved(self, retrieved):
        """
        Parses the datafolder, stores results.
        """

        # suppose at the start that the job is successful
        successful = True

        # select the folder object
        # Check that the retrieved folder is there
        try:
            out_folder = retrieved[self._calc._get_linkname_retrieved()]
        except KeyError:
            self.logger.error("No retrieved folder found")
            return False, ()

        # check what is inside the folder
        list_of_files = out_folder.get_folder_list()

        # OUTPUT file should exist
        if not self._calc._OUTPUT_FILE_NAME in list_of_files:
            successful = False
            self.logger.error("Output file not found")
            return successful, ()

        # Get file and do the parsing
        outfile = out_folder.get_abs_path(self._calc._OUTPUT_FILE_NAME)
        print self._calc

        force_constants = parse_FORCE_CONSTANTS(outfile)

        if self._calc._OUTPUT_DOS in list_of_files:
            outfile = out_folder.get_abs_path(self._calc._OUTPUT_DOS)
            dos_object = parse_partial_DOS(outfile, self._calc.inp.structure)

        if self._calc._OUTPUT_TP in list_of_files:
            outfile = out_folder.get_abs_path(self._calc._OUTPUT_TP)
            tp_object = parse_thermal_properties(outfile)


        # look at warnings
        warnings = []
        with open(out_folder.get_abs_path(self._calc._SCHED_ERROR_FILE)) as f:
            errors = f.read()
        if errors:
            warnings = [errors]



        # ====================== prepare the output node ======================

        # save the outputs
        new_nodes_list = []

        # save force constants into node
        try:
            new_nodes_list.append(('force_constants', ForceConstantsData(data=force_constants)))
        except KeyError:  # keys not found in json
            pass
        try:
            new_nodes_list.append(('dos', dos_object))
        except:
            pass

        try:
            new_nodes_list.append(('thermal_properties', tp_object))
        except:
            pass



        # add the dictionary with warnings
        new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict={'warnings': warnings})))

        return successful, new_nodes_list
