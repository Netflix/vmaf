import uuid

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

class Parallelizable(object):

    def __init__(self, workdir_root):
        self._get_workdir(workdir_root)

    def _get_workdir(self, workdir_root):
        subdir = str(uuid.uuid4())
        self.workdir = "{root}/{subdir}".format(root=workdir_root, subdir=subdir)