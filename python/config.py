__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os

PYTHON_ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.abspath(os.path.join(PYTHON_ROOT, '..',))

DAALA_TOOLS_DIR = os.getenv("DAALATOOLSPATH")
