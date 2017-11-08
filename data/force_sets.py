from aiida.orm.data.array import ArrayData


class ForceSetsData(ArrayData):
    """
    Store the force constants on disk as a numpy array. It requires numpy to be installed.
    """

    def __init__(self, *args, **kwargs):
        super(ForceSetsData, self).__init__(*args, **kwargs)
        self._cached_arrays = {}

    def get_number_of_atoms(self):
        """
        Return the shape of an array (read from the value cached in the
        properties for efficiency reasons).
        :param name: The name of the array.
        """
        return self.get_attr("natom")

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

        direction = self.get_array('direction')
        number = self.get_array('number')
        displacement = self.get_array('displacement')

        first_atoms = []
        for i in range(ndisplacements):
            first_atoms.append({'direction': direction[i],
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

        direction = self.get_array('direction')
        number = self.get_array('number')
        displacement = self.get_array('displacement')
        forces = self.get_array('forces')

        first_atoms = []
        for i in range(ndisplacements):
            first_atoms.append({'directions': direction[i],
                                'number': number[i],
                                'forces': forces[i],
                                'displacement': displacement[i]})

        return {'natom': natom, 'first_atoms': first_atoms}

    # {'natom': 64, 'first_atoms': [{'direction': [1, 0, 0], 'number': 0, 'displacement': array([0.01, 0., 0.])}]}

    def set_data_sets(self, data_sets):

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

        self.set_array('direction', numpy.array(direction))
        self.set_array('number', numpy.array(number))
        self.set_array('displacement', numpy.array(displacement))

    def set_forces(self, forces):

        import numpy
        self.set_array('forces', numpy.array(forces))

    def get_phonopy_formatted_txt(self):

        data_sets = self.get_force_sets()

    #    data_list = []
    #    for name in names:
    #        data_list.append({'direction': data_sets_object.get_array(name)[0],
    #                          'number': data_sets_object.get_array(name)[1],
    #                          'displacement': data_sets_object.get_array(name)[2],
    #                          'forces': data_sets_object.get_array(name)[3]})
    #    data_sets = {'natom': num_atom, 'first_atoms': data_list}

        displacements = data_sets['first_atoms']
        forces = [x['forces'] for x in data_sets['first_atoms']]

        # Write FORCE_SETS
        force_sets_txt = "%-5d\n" % data_sets['natom']
        force_sets_txt += "%-5d\n" % len(displacements)
        for count, disp in enumerate(displacements):
            force_sets_txt += "\n%-5d\n" % (disp['number'] + 1)
            force_sets_txt += "%20.16f %20.16f %20.16f\n" % (tuple(disp['displacement']))

            for f in forces[count]:
                force_sets_txt += "%15.10f %15.10f %15.10f\n" % (tuple(f))
        return force_sets_txt