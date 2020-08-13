# TODO: dependency on src/libsvm/svmutil needs to be properly done, this is a temporary workaround wrapper

from __future__ import absolute_import

import sys
from vmaf.config import VmafConfig

if VmafConfig.root_path() not in sys.path:
    sys.path.append(VmafConfig.root_path())

# This will work only when running with a checked out vmaf source, but not via pip install
# libsvm_path = VmafConfig.root_path('third_party', 'libsvm', 'python')
# if libsvm_path not in sys.path:
#     # Inject {project}/src/libsvm/python to PYTHONPATH dynamically
#     sys.path.append(libsvm_path)

try:
    # This import will work only if above injection was meaningful (ie: user has the files in the right place)
    # from svmutil import *           # noqa
    from third_party.libsvm.python.svmutil import *

except ImportError as e:
    print("Can't import svmutil from %s: %s" % (VmafConfig.root_path()+".third_party.libsvm.python.svmutil", e))
    sys.exit(1)
