from aiida.orm import Data


class BandStructureData(Data):
    """
    Store the band structure.
    """

    def __init__(self, *args, **kwargs):
        super(BandStructureData, self).__init__(*args, **kwargs)

    def set_bands(self, ranges):

        import tempfile
        import numpy

        ranges = numpy.array(ranges)

        with tempfile.NamedTemporaryFile() as f:
            numpy.save(f, ranges)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, 'band_ranges.npy')

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

    def get_q_points(self):
        """
        Return the q_points in the node as a numpy array
        """
        import numpy

        fname = 'q_points.npy'

        array = numpy.load(self.get_abs_path(fname))
        return array


    def get_distances(self):
        """
        Return the q_points in the node as a numpy array
        """
        import numpy

        fname = 'distances.npy'

        array = numpy.load(self.get_abs_path(fname))
        return array

    def get_frequencies(self):
        """
        Return the q_points in the node as a numpy array
        """
        import numpy

        fname = 'frequencies.npy'

        array = numpy.load(self.get_abs_path(fname))
        return array

    def get_bands(self):
        """
        Return the band ranges in the node as a numpy array
        """
        import numpy

        fname = 'band_ranges.npy'

        array = numpy.load(self.get_abs_path(fname))
        return array

    def get_labels(self):
        """
        Return the band labels in the node as a numpy array
        """
        import numpy

        fname = 'band_labels.npy'

        array = numpy.load(self.get_abs_path(fname))
        return array

