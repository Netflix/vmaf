__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

"""
Run embedded doctests
"""

import doctest

from vmaf.tools import misc
from vmaf.tools import stats


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(misc))
    # tests.addTests(doctest.DocTestSuite(stats)) # commented out because not numerically exact
    return tests
