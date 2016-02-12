__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import uuid

class Parallelizable(object):
    """
    Facilitate objects in its derived class to be executed in parallel
    thread-safely by providing each object a unique working directory.
    """

    def __init__(self, workdir_root):
        self._get_workdir(workdir_root)

    def _get_workdir(self, workdir_root):
        subdir = str(uuid.uuid4())
        self.workdir = "{root}/{subdir}".format(root=workdir_root, subdir=subdir)

class Executor(object):
    """
    Derived class of which has a uniquely identifier combining a type name
    and a version number. The string is useful in identifying a result by which
    executor it is generated (e.g. VMAF_V0.1, or VMAF_feature_V0.1)
    """

    def __init__(self):
        self._assert()

    def _assert(self):
        assert hasattr(self, 'TYPE')
        assert hasattr(self, 'VERSION')

    @property
    def executor_id(self):
        return "{type}_V{version}".format(type=self.TYPE, version=self.VERSION)

