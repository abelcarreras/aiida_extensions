from aiida.orm import Data


class BandStructureData(Data):
    """
    Store the band structure.
    """

    def __init__(self, *args, **kwargs):
        super(BandStructureData, self).__init__(*args, **kwargs)

    def set_ranges(self, ranges):

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

    def get_ranges(self):
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

