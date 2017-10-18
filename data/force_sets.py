from aiida.orm import Data


class ForceSets(Data):
    """
    Store the force constants on disk as a numpy array. It requires numpy to be installed.
    """

    def __init__(self, *args, **kwargs):
        super(ForceSets, self).__init__(*args, **kwargs)
        self._cached_arrays = {}

        if 'data_sets' in kwargs:
            self.set_data_sets(kwargs['data_sets'])

    def get_number_of_atoms(self):
        """
        Return the shape of an array (read from the value cached in the
        properties for efficiency reasons).
        :param name: The name of the array.
        """
        return tuple(self.get_attr("natom"))

    def get_number_of_displacements(self):
        """
        Return the shape of an array (read from the value cached in the
        properties for efficiency reasons).
        :param name: The name of the array.
        """
        return self.get_attr("ndisplacements")

    def get_data_sets(self):
        """
        Return the force constants stored in the node as a numpy array
        """
        import numpy

        natom = self.get_attr("natom")
        ndisplacements = self.get_attr("ndisplacements")

        direction = numpy.load(self.get_abs_path('direction.npy'))
        number = numpy.load(self.get_abs_path('number.npy'))
        displacement = numpy.load(self.get_abs_path('displacement.npy'))

        first_atoms = []
        for i in range(ndisplacements):
            first_atoms.append({'directions': direction[i],
                                'number': number[i],
                                'displacement': displacement[i]})

        return {'natom': natom, 'first_atoms': first_atoms}

    def get_force_sets(self):
        """
        Return the force constants stored in the node as a numpy array
        """
        import numpy

        natom = self.get_attr("natom")
        ndisplacements = self.get_attr("ndisplacements")

        direction = numpy.load(self.get_abs_path('direction.npy'))
        number = numpy.load(self.get_abs_path('number.npy'))
        displacement = numpy.load(self.get_abs_path('displacement.npy'))
        forces = numpy.load(self.get_abs_path('forces.npy'))

        first_atoms = []
        for i in range(ndisplacements):
            first_atoms.append({'directions': direction[i],
                                'number': number[i],
                                'forces': forces[i],
                                'displacement': displacement[i]})

        return {'natom': natom, 'first_atoms': first_atoms}

    # {'natom': 64, 'first_atoms': [{'direction': [1, 0, 0], 'number': 0, 'displacement': array([0.01, 0., 0.])}]}

    def set_data_sets(self, data_sets):

        import tempfile
        import numpy

        self._set_attr('natom', data_sets['natom'])
        self._set_attr('ndisplacements', len(data_sets['first_atoms']))

        direction = []
        number = []
        displacement = []
        for first_atoms in data_sets['first_atoms']:
            direction.append(first_atoms['direction'])
            number.append(first_atoms['number'])
            displacement.append(first_atoms['displacement'])

        direction = numpy.array(direction)
        number = numpy.array(number)
        displacement = numpy.array(displacement)


        #if not (isinstance(array, numpy.ndarray)):
        #    raise TypeError("ArrayData can only store numpy arrays. Convert "
        #                    "the object to an array first")


        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, direction)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'direction.npy')

        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, number)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'number.npy')

        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, displacement)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'displacement.npy')

    def set_forces(self, forces):

        import tempfile
        import numpy

        forces = numpy.array(forces)
        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, forces)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'forces.npy')


