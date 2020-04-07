import h5py

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class H5pyMixin(object):
    """
    Use a h5py file to store raw video channel or similar as features.
    Implementation class must have attribute optional_dict2.
    """
    @staticmethod
    def open_h5py_file(h5py_filepath, mode='w'):
        f = h5py.File(h5py_filepath, mode)
        return f

    @staticmethod
    def close_h5py_file(f, mode='w'):
        if mode == 'w':
            f.flush()
            f.close()
        elif mode == 'r':
            f.close()
        else:
            assert False

    def assert_h5py_file(self):
        assert self.optional_dict2 is not None and 'h5py_file' in self.optional_dict2

    @property
    def h5py_file(self):
        return self.optional_dict2['h5py_file']