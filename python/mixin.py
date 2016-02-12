__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import uuid
import re

class WorkdirEnabled(object):
    """
    Facilitate objects in its derived class to be executed in parallel
    thread-safely by providing each object a unique working directory.
    """

    def __init__(self, workdir_root):
        self._get_workdir(workdir_root)

    def _get_workdir(self, workdir_root):
        subdir = str(uuid.uuid4())
        self.workdir = "{root}/{subdir}".format(root=workdir_root, subdir=subdir)

class TypeVersionEnabled(object):
    """
    Mandate a type name and a version string. Derived class (e.g. an Executor)
    then has a unique string combining type and version. The string is useful in
    identifying a Result by which Executor it is generated (e.g. VMAF_V0.1,
    SSIM_V0.1, or VMAF_feature_V0.1).
    """

    def __init__(self):
        self._assert_type_version()

    def _assert_type_version(self):
        assert hasattr(self, 'TYPE')
        assert hasattr(self, 'VERSION')

        assert re.match(r"[a-zA-Z0-9_]+", self.TYPE), \
            "TYPE can only contains alphabets, numbers and underscore (_)."

        assert re.match(r"[a-zA-Z0-9.]+", self.VERSION), \
            "VERSION can only contains alphabets, numbers and dot (.)."

    def get_type_version_string(self):
        return "{type}_V{version}".format(type=self.TYPE, version=self.VERSION)
