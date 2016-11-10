import numpy as np

from core.perf_metric import RmsePerfMetric, SrccPerfMetric, PccPerfMetric, \
    KendallPerfMetric

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

class AggrScorePerfMetricTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_rmse_perf_metric(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 3, 4]
        metric = RmsePerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 0.0)

    def test_rmse_perf_metric_enable_mapping(self):
        groundtruths = np.arange(0, 1, 0.0001)
        predictions = np.arange(0, 1, 0.0001)
        metric = RmsePerfMetric(groundtruths, predictions)
        result = metric.evaluate(enable_mapping=True)
        self.assertAlmostEqual(result['score'], 0.022753642178052261, places=6)

    def test_rmse_perf_metric2(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 3, 5]
        metric = RmsePerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 0.5, places=6)

    def test_srcc_perf_metric(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 3, 5]
        metric = SrccPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 1.0, places=6)

    def test_srcc_perf_metric2(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 5, 3]
        metric = SrccPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 0.79999999999999993, places=6)

    def test_srcc_perf_metric_enable_mapping(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 3, 5]
        metric = SrccPerfMetric(groundtruths, predictions)
        result = metric.evaluate(enable_mapping=True)
        self.assertAlmostEqual(result['score'], 1.0, places=6)

    def test_pcc_perf_metric(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 3, 5]
        metric = PccPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 0.98270762982399085, places=6)

    def test_kendall_perf_metric(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 3, 5]
        metric = KendallPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 1.0, places=6)

    def test_kendall_perf_metric2(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 5, 3]
        metric = KendallPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 0.66666666666666663, places=6)

    def test_kendall_perf_metric_enable_mapping(self):
        groundtruths = [1, 2, 3, 4]
        predictions = [1, 2, 3, 5]
        metric = KendallPerfMetric(groundtruths, predictions)
        result = metric.evaluate(enable_mapping=True)
        self.assertAlmostEqual(result['score'], 1.0, places=6)
