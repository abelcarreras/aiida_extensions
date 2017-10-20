from aiida.orm import Data


class BandStructureData(Data):
    """
    Store the band structure.
    """

    def __init__(self, *args, **kwargs):
        super(BandStructureData, self).__init__(*args, **kwargs)

    def get_number_of_bands(self):

        if 'nbands' in self.get_attrs():
            return self.get_attr('nbands')
        else:
            return None


    def set_bands(self, ranges):

        import tempfile
        import numpy

        ranges = numpy.array(ranges)



        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, ranges)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'band_ranges.npy')

        self._set_attr('nbands', len(ranges))

    def set_labels(self, band_labels):

        import tempfile
        import numpy

        band_labels = numpy.array(band_labels)

        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, band_labels)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'band_labels.npy')

    def set_band_structure_phonopy(self, band_structure_phonopy):

        import tempfile
        import numpy

        q_points = numpy.array(band_structure_phonopy[0])
        distances = numpy.array(band_structure_phonopy[1])
        frequencies = numpy.array(band_structure_phonopy[2])

        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, q_points)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'q_points.npy')

        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, distances)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'distances.npy')

        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, frequencies)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'frequencies.npy')

    def get_q_points(self, band=None):
        """
        Return the q_points in the node as a numpy array
        """
        import numpy

        fname = 'q_points.npy'

        array = numpy.load(self.get_abs_path(fname))
        if band is not None:
            array = array[band]

        return array


    def get_distances(self, band=None):
        """
        Return the distances in the node as a numpy array
        """
        import numpy

        fname = 'distances.npy'

        array = numpy.load(self.get_abs_path(fname))
        if band is not None:
            array = array[band]

        return array

    def get_frequencies(self, band=None):
        """
        Return the frequencies in the node as a numpy array
        """
        import numpy

        fname = 'frequencies.npy'

        array = numpy.load(self.get_abs_path(fname))
        if band is not None:
            array = array[band]

        return array

    def get_bands(self, band=None):
        """
        Return the bands in the node as a numpy array
        """
        import numpy

        fname = 'band_ranges.npy'

        array = numpy.load(self.get_abs_path(fname))
        if band is not None:
            array = array[band]

        return array

    def get_labels(self, band=None):
        """
        Return the band labels in the node as a numpy array
        """
        import numpy

        fname = 'band_labels.npy'

        array = numpy.load(self.get_abs_path(fname))
        if band is not None:
            array = array[band]

        return array

