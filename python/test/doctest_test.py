__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

"""
Run embedded doctests
"""

import doctest

from vmaf.tools import misc
from vmaf.tools import stats

def test_doctest():
    doctest.testmod(misc)
    doctest.testmod(stats)
