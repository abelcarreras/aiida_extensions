from aiida.orm.data.array import ArrayData


class PhononDosData(ArrayData):
    """
    Store the phonon DOS on disk as a numpy array. It requires numpy to be installed.
    """

    def __init__(self, *args, **kwargs):
        super(PhononDosData, self).__init__(*args, **kwargs)
        self._cached_arrays = {}

    def _get_equivalent_atom_list(self):
        import numpy

        partial_dos = numpy.array(self.get_array('partial_dos'))
        partial_symbols = self.get_attr("atom_labels")

        # Check atom equivalences
        delete_list = []
        for i, dos_i in enumerate(partial_dos):
            for j, dos_j in enumerate(partial_dos.copy()):
                if i < j:
                    print ('i',i, 'j', j)
                    if numpy.allclose(dos_i, dos_j, rtol=1, atol=1e-8) and partial_symbols[i] == partial_symbols[j]:
                        #dos_i += dos_j
                        delete_list.append(j)
        print delete_list
        return numpy.delete(range(len(partial_dos)), delete_list)

    def get_dos(self):
        """
        Return the force constants stored in the node as a numpy array
        """

        return self.get_array('dos')


    def get_number_of_partial_dos(self, full=False):
        """
        Return the force constants stored in the node as a numpy array
        """

        partial_dos = self.get_array('partial_dos')

        if full:
            return len(partial_dos)

        return len(partial_dos[self._get_equivalent_atom_list()])

    def get_partial_dos(self, full=False):
        """
        Return the force constants stored in the node as a numpy array
        """

        partial_dos = self.get_array('partial_dos')

        if full:
            return partial_dos
        return partial_dos[self._get_equivalent_atom_list()]

    def get_frequencies(self):
        """
        Return the frequencies stored in the node as a numpy array
        """

        return  self.get_array('frequencies')


    def get_atom_labels(self, full=False):
        """
        Store the phonon dos as a numpy array.
        :param array: The numpy array to store.
        """
        import numpy

        labels = self.get_attr("atom_labels")

        if full:
            return labels
        return numpy.array(labels)[self._get_equivalent_atom_list()].tolist()

    def set_atom_labels(self, labels):
        """
        Store the phonon dos as a numpy array.
        :param array: The numpy array to store.
        """
        self._set_attr("atom_labels", labels)

    def set_dos(self, array):
        """
        Store the phonon dos as a numpy array.
        :param array: The numpy array to store.
        """

        self.set_array('dos', array)

    def set_frequencies(self, array):
        """
        Store the frequencies as a numpy array.
        :param array: The numpy array to store.
        """
        self.set_array('frequencies', array)

    def set_partial_dos(self, array):
        """
        Store the partial dos as a numpy array.
        :param array: The numpy array to store.
        """

        self.set_array('partial_dos', array)
        self._set_attr("n_partial_dos", len(array))
