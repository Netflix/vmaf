__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

from common import Executor

class FeatureAssembler(object):
    pass

class FeatureExtractor(Executor):

    def __init__(self):
        Executor.__init__(self)

class VmafFeatureExtractor(FeatureExtractor):

    TYPE = "VMAF_feature"
    VERSION = '0.1'
