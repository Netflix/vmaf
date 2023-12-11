import sys
import unittest

import numpy as np
import scipy.io

from vmaf.config import VmafConfig
from vmaf.core.perf_metric import RmsePerfMetric, SrccPerfMetric, PccPerfMetric, \
    KendallPerfMetric, AucPerfMetric, ResolvingPowerPerfMetric

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


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

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: py2 uses a different random seed.")
    def test_auc_perf_metric(self):
        np.random.seed(1)
        groundtruths = np.random.normal(0, 1.0, [4, 10]) + np.tile(np.array([1, 2, 3, 4]), [10, 1]).T
        predictions = [1, 2, 3, 4]
        metric = AucPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 0.9999999999999999, places=6)
        self.assertAlmostEqual(result['AUC_BW'], 0.9999999999999999, places=6)
        self.assertAlmostEqual(result['AUC_DS'], 0.9375, places=6)
        self.assertAlmostEqual(result['CC_0'], 1.0, places=6)
        self.assertAlmostEqual(result['THR'], 1.0, places=6)

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: py2 uses a different random seed.")
    def test_auc_perf_single_metric_in_list(self):
        np.random.seed(1)
        groundtruths = np.random.normal(0, 1.0, [4, 10]) + np.tile(np.array([1, 2, 3, 4]), [10, 1]).T
        predictions = [[1, 2, 3, 4]]
        metric = AucPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'][0], 0.9999999999999999, places=6)
        self.assertAlmostEqual(result['AUC_BW'][0], 0.9999999999999999, places=6)
        self.assertAlmostEqual(result['AUC_DS'][0], 0.9375, places=6)
        self.assertAlmostEqual(result['CC_0'][0], 1.0, places=6)
        self.assertAlmostEqual(result['THR'][0], 1.0, places=6)

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: py2 uses a different random seed.")
    def test_auc_perf_multiple_metrics(self):
        np.random.seed(1)
        groundtruths = np.random.normal(0, 1.0, [4, 10]) + np.tile(np.array([1, 2, 3, 4]), [10, 1]).T
        predictions = [[1, 2, 3, 4], [3, 1, 2, 4]]
        metric = AucPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'][0], 0.9999999999999999, places=6)
        self.assertAlmostEqual(result['AUC_BW'][0], 0.9999999999999999, places=6)
        self.assertAlmostEqual(result['AUC_DS'][0], 0.9375, places=6)
        self.assertAlmostEqual(result['CC_0'][0], 1.0, places=6)
        self.assertAlmostEqual(result['THR'][0], 1.0, places=6)
        self.assertAlmostEqual(result['score'][1], 0.8125, places=6)
        self.assertAlmostEqual(result['AUC_BW'][1], 0.8125, places=6)
        self.assertAlmostEqual(result['AUC_DS'][1], 0.6250, places=6)
        self.assertAlmostEqual(result['CC_0'][1], 0.75, places=6)
        self.assertAlmostEqual(result['THR'][1], 2, places=6)
        self.assertAlmostEqual(result['pDS_DL'][0, 1], 0.02746864, places=6)
        self.assertAlmostEqual(result['pBW_DL'][0, 1], 0.06136883, places=6)
        self.assertAlmostEqual(result['pCC0_b'][0, 1], 0.03250944, places=6)

    def test_auc_metrics_performance(self):
        mat_filepath = VmafConfig.test_resource_path('data_Toyama.mat')
        mat_dict = scipy.io.loadmat(mat_filepath)
        results = AucPerfMetric._metrics_performance(mat_dict['objScoDif'], mat_dict['signif'])
        self.assertAlmostEqual(float(np.mean(results['AUC_DS'])), 0.69767003960902052, places=6)
        self.assertAlmostEqual(float(np.mean(results['AUC_BW'])), 0.94454700301894534, places=6)
        self.assertAlmostEqual(float(np.mean(results['CC_0'])), 0.88105386206276415, places=6)
        self.assertAlmostEqual(float(np.mean(results['THR'])), 3.899105581509778, places=6)

    def test_respow_perf_metric(self):
        np.random.seed(0)
        groundtruths = np.random.normal(0, 1.0, [4, 10]) + np.tile(np.array([1, 2, 3, 4]), [10, 1]).T
        predictions = [1, 2, 3, 4]
        metric = ResolvingPowerPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['resolving_power_95perc'], 1.2176359647113211, places=6)
        self.assertAlmostEqual(result['score'], 1.2176359647113211, places=6)

    def test_respow_perf_metric2(self):
        np.random.seed(0)
        groundtruths = np.random.normal(0, 10.0, [100, 30]) + np.tile(np.array(np.arange(100)), [30, 1]).T
        predictions = np.arange(100)
        metric = ResolvingPowerPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['score'], 9.0014569671225111, places=6)

    def test_respow_perf_metric3(self):
        np.random.seed(0)
        groundtruths = []
        for i in range(3):
            groundtruths.append(list(np.random.normal(0, 1.0, [1, 10])[0] + i + 1))
        groundtruths.append(list(np.random.normal(0, 1.0, [1, 8])[0] + 4))
        predictions = [1, 2, 3, 4]
        metric = ResolvingPowerPerfMetric(groundtruths, predictions)
        result = metric.evaluate()
        self.assertAlmostEqual(result['resolving_power_95perc'], 1.2045849280245293, places=6)
        self.assertAlmostEqual(result['score'], 1.2045849280245293, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
