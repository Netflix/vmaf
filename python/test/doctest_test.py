"""
Run embedded doctests
"""

import doctest

from vmaf.tools import misc
from vmaf.tools import stats

def test_doctest():
    doctest.testmod(misc)
    doctest.testmod(stats)
