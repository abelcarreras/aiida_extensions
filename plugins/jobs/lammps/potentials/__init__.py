import importlib


class LammpsPotential:
    def __init__(self, data, structure, potential_filename='None'):

        self._names = [site.name for site in structure.kinds]

        self._type = data.dict.pair_style
        self._data = data.dict.data
        self._potential_filename = potential_filename

        try:
            potential_module=importlib.import_module(self._type)
            self._generate_LAMMPS_potential = potential_module._generate_LAMMPS_potential
            self._get_input_potential_lines = potential_module._get_input_potential_lines
        except ImportError:
            raise ImportError('This lammps potential is not implemented')

#        if self._type == 'tersoff':
#            from tersoff import _generate_LAMMPS_potential, _get_input_potential_lines
#        elif self._type == 'lennard_jones':
#            from lennard_jones import _generate_LAMMPS_potential, _get_input_potential_lines
#        else:
#            raise ValueError('This lammps potential is not implemented')

#        self._generate_LAMMPS_potential = _generate_LAMMPS_potential
#        self._get_input_potential_lines = _get_input_potential_lines

    def get_potential_file(self):
        return self._generate_LAMMPS_potential(self._data)

    def get_input_potential_lines(self):
        return self._get_input_potential_lines(self._data,
                                               potential_filename=self._potential_filename,
                                               names=self._names)
