from aiida.orm import Data


class ForceConstants(Data):
    """
    Store the force constants on disk as a numpy array. It requires numpy to be installed.
    """

    def __init__(self, *args, **kwargs):
        super(ForceConstants, self).__init__(*args, **kwargs)
        self._cached_arrays = {}

    def get_shape(self):
        """
        Return the shape of an array (read from the value cached in the
        properties for efficiency reasons).
        :param name: The name of the array.
        """
        return tuple(self.get_attr("shape"))

    def get_array(self):
        """
        Return the force constants stored in the node as a numpy array
        """
        import numpy

        fname = 'force_constants.npy'

        array = numpy.load(self.get_abs_path(fname))
        return array

    def set_array(self, array):
        """
        Store the force constants as a numpy array. Possibly overwrite the array
        if it already existed.
        Internally, it is stored as a force_constants.npy file in numpy format.
        :param array: The numpy array to store.
        """

        import tempfile

        import numpy

        if not (isinstance(array, numpy.ndarray)):
            raise TypeError("ArrayData can only store numpy arrays. Convert "
                            "the object to an array first")

        fname = "force_constants.npy"

        with tempfile.NamedTemporaryFile() as f:
            # Store in a temporary file, and then add to the node
            numpy.save(f, array)
            f.flush()  # Important to flush here, otherwise the next copy command
            # will just copy an empty file
            self.add_path(f.name, fname)

        self._set_attr("shape", list(array.shape))
