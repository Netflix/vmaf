__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
from stats import StatsList
from asset import Asset
import config
from result import Result, FileSystemResultStore
from quality_runner import VmafLegacyQualityRunner
import numpy as np

class StatsListTest(unittest.TestCase):

    def setUp(self):
        self.test_dict_1 = [1,2,3,4,5,11,12,13,14,15]
        pass

    def tearDown(self):
        pass


    def test_stats(self):
        self.assertAlmostEquals(StatsList.min(self.test_dict_1), 1)
        self.assertAlmostEquals(StatsList.max(self.test_dict_1), 15)
        self.assertAlmostEquals(StatsList.median(self.test_dict_1), 8)
        self.assertAlmostEquals(StatsList.mean(self.test_dict_1), 8)
        self.assertAlmostEquals(StatsList.stddev(self.test_dict_1), 5.19615242271)
        self.assertAlmostEquals(StatsList.var(self.test_dict_1), 27)
        self.assertAlmostEquals(StatsList.percentile(self.test_dict_1, 50), 8)
        self.assertAlmostEquals(StatsList.percentile(self.test_dict_1, 80), 13.2)
        self.assertAlmostEquals(StatsList.total_var(self.test_dict_1), 1.5555555555555556)

        expected_moving_avg_a = [2.26894142,2.26894142,2.26894142,3.26894142,4.26894142,6.61364853,11.26894142,12.26894142,13.26894142,14.26894142]
        for i in range(len(expected_moving_avg_a)): self.assertAlmostEquals(StatsList.moving_average(self.test_dict_1, 2)[i], expected_moving_avg_a[i])


        expected_moving_avg_b = [4.08330969,4.08330969,4.08330969,4.08330969,4.08330969,4.08330969,5.81552983,7.7557191,9.96294602,12.51305607]
        for i in range(len(expected_moving_avg_b)): self.assertAlmostEquals(StatsList.moving_average(self.test_dict_1, 5)[i], expected_moving_avg_b[i])