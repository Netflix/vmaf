import unittest

import numpy as np
import scipy.io

import config
from core.perf_metric import RmsePerfMetric, SrccPerfMetric, PccPerfMetric, \
    KendallPerfMetric, KflkPerfMetric

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

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

    def test_kflk_perf_metric(self):
        np.random.seed(0)
        groundtruths = np.random.normal(0, 1.0, [4, 10]) + \
                      np.tile(np.array([1, 2, 3, 4]), [10, 1]).T
        predictions = [1, 2, 3, 4]
        metric = KflkPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 0.95, places=6)
        self.assertAlmostEqual(result['AUC_BW'], 0.9166666666666666, places=6)
        self.assertAlmostEqual(result['AUC_DS'], 0.95, places=6)
        self.assertAlmostEqual(result['CC_0'], 1.0, places=6)
        self.assertAlmostEqual(result['THR'], 3.0, places=6)

    def test_kflk_metrics_performance(self):
        mat_filepath = config.ROOT + '/python/test/resource/data_Toyama.mat'
        mat_dict = scipy.io.loadmat(mat_filepath)
        results = KflkPerfMetric._metrics_performance(mat_dict['objScoDif'], mat_dict['signif'])
        self.assertAlmostEqual(np.mean(results['AUC_DS']), 0.69767003960902052, places=6)
        self.assertAlmostEqual(np.mean(results['AUC_BW']), 0.94454700301894534, places=6)
        self.assertAlmostEqual(np.mean(results['CC_0']), 0.88105386206276415, places=6)
        self.assertAlmostEqual(np.mean(results['THR']), 6.2392849606450556, places=6)
