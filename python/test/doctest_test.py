__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

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
