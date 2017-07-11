from aiida.parsers.exceptions import OutputParsingError
from aiida.orm.data.array.trajectory import TrajectoryData

from aiida.parsers.plugins.vasp.instruction import BaseInstruction


import numpy as np
import mmap


def read_VASP_XDATCAR(file_name, time_step,
                      limit_number_steps=10000000,
                      initial_cut=1,
                      end_cut=None):

    # Dimensionality of VASP calculation
    number_of_dimensions = 3

    configuration_number = []
    positions = []
    counter = 0

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        # Read cell
        for i in range(2): file_map.readline()
        a = file_map.readline().split()
        b = file_map.readline().split()
        c = file_map.readline().split()
        cells = np.array([a, b, c], dtype='double').T

        atomic_symbols_line = file_map.readline().split()

        # for i in range(1): file_map.readline()
        number_of_types = np.array(file_map.readline().split(), dtype=int)
        # print number_of_types
        symbols = []
        for i, j in enumerate(atomic_symbols_line):
            symbols.append([j] * number_of_types[i])
            print symbols
        symbols = [item for sublist in symbols for item in sublist]

        number_of_atoms = number_of_types.sum()

        while True:

            counter += 1

            # Read time steps
            position_number = file_map.find('Direct configuration')
            if position_number < 0: break

            file_map.seek(position_number)
            configuration_number.append(int(file_map.readline().split('=')[1])-1)

            # Initial cut control
            if initial_cut > counter:
                continue

            # Reading coordinates
            read_coordinates = []
            for i in range(number_of_atoms):
                read_coordinates.append(file_map.readline().split()[0:number_of_dimensions])

            try:
                positions.append(np.array(read_coordinates, dtype=float))  # in angstroms
            except ValueError:
                print("Error reading step {0}".format(counter))
                break
                # print(read_coordinates)

            # security routine to limit maximum of steps to read and put in memory
            if limit_number_steps + initial_cut < counter:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break

    file_map.close()

    positions = np.array(positions)

    n_steps = len(positions)

    np.array([cells.tolist() * n_steps])
    step_ids = range(n_steps)
    time = np.array(configuration_number) * time_step

    return step_ids, positions, time, cells, symbols


class Trajectory_parametersInstruction(BaseInstruction):

    print ('INCLASS')
    _input_file_list_ = ['POSCAR', 'XDATCAR']

    def _parser_function(self):
        """
        Parses the XDATCAR using custom function.
        """

        parser_warnings = {}  # return non-critical errors

        timestep = self._calc.inp.incar.dict.NSW * 1e-3  # In picoseconds

        # extract data
        try:
            step_ids, positions, time, cells, symbols = read_VASP_XDATCAR(self._out_folder.get_abs_path('XDATCAR'), timestep)
        except:
            print ('Error parsing XDATCAR')

        #  construct proper trajectory data format
        trajectory_data = TrajectoryData()
        try:
            nodes_list = []
            trajectory_data.set_trajectory(step_ids, cells, symbols, positions, times=time)
            nodes_list.append((
                'trajectory_data',
                trajectory_data
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

        return nodes_list, parser_warnings
