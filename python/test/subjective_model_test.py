import os
import unittest
import numpy as np
import config
from mos.dataset_reader import RawDatasetReader, MissingDataRawDatasetReader, \
    SyntheticRawDatasetReader, CorruptSubjectRawDatasetReader
from mos.subjective_model import MosModel, DmosModel, \
    MaximumLikelihoodEstimationModelReduced, MaximumLikelihoodEstimationModel, \
    LiveDmosModel, MaximumLikelihoodEstimationDmosModel, LeastSquaresModel, \
    SubjrejMosModel, ZscoringSubjrejMosModel, SubjrejDmosModel, \
    ZscoringSubjrejDmosModel, PerSubjectModel
from tools.misc import import_python_file

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class SubjectiveModelTest(unittest.TestCase):

    def setUp(self):
        self.dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        self.output_dataset_filepath = config.ROOT + '/workspace/workdir/NFLX_dataset_public_test.py'
        self.output_dataset_pyc_filepath = config.ROOT + '/workspace/workdir/NFLX_dataset_public_test.pyc'

    def tearDown(self):
        if os.path.exists(self.output_dataset_filepath):
            os.remove(self.output_dataset_filepath)
        if os.path.exists(self.output_dataset_pyc_filepath):
            os.remove(self.output_dataset_pyc_filepath)

    def test_mos_subjective_model(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 4.884615384615385, places=4)
        self.assertAlmostEquals(scores[10], 2.0769230769230771, places=4)
        self.assertAlmostEquals(np.mean(scores), 3.544790652385589, places=4)

    def test_mos_subjective_model_output(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        subjective_model.run_modeling()
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEquals(dis_video['groundtruth'], 4.884615384615385, places=4)

    def test_mos_subjective_model_output_custom_resampling(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        subjective_model.run_modeling()
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath, resampling_type='lanczos')
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        self.assertFalse(hasattr(dataset2, 'quality_height'))
        self.assertFalse(hasattr(dataset2, 'quality_width'))
        self.assertEquals(dataset2.resampling_type, 'lanczos')
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEquals(dis_video['groundtruth'], 4.884615384615385, places=4)

    def test_mos_subjective_model_output2(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        subjective_model.run_modeling()
        dataset2 = subjective_model.to_aggregated_dataset()
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEquals(dis_video['groundtruth'], 4.884615384615385, places=4)

    def test_mos_subjective_model_normalize_final(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 1.1318646945818083, places=4)
        self.assertAlmostEquals(scores[10], -1.2400334499143002, places=4)
        self.assertAlmostEquals(np.mean(scores), 0.0, places=4)

    def test_mos_subjective_model_transform_final(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling(transform_final={'p1': 10, 'p0': 1})
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 49.84615384615385, places=4)
        self.assertAlmostEquals(scores[10], 21.769230769230771, places=4)
        self.assertAlmostEquals(np.mean(scores), 36.44790652385589, places=4)

    def test_from_dataset_file(self):
        subjective_model = MosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 4.884615384615385, places=4)
        self.assertAlmostEquals(scores[10], 2.0769230769230771, places=4)
        self.assertAlmostEquals(np.mean(scores), 3.544790652385589, places=4)

    def test_dmos_subjective_model(self):
        subjective_model = DmosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 5.0, places=4)
        self.assertAlmostEquals(scores[10], 2.1923076923076921, places=4)
        self.assertAlmostEquals(np.mean(scores), 3.7731256085686473, places=4)

    def test_dmos_subjective_model_normalize_final(self):
        subjective_model = DmosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 1.0440613892053001, places=4)
        self.assertAlmostEquals(scores[10], -1.3452648137895296, places=4)
        self.assertAlmostEquals(np.mean(scores), 0.0, places=4)

    def test_dmos_subjective_model_dscore_mode_same(self):
        subjective_model = DmosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 1.0440613892053001, places=4)
        self.assertAlmostEquals(scores[10], -1.3452648137895296, places=4)
        self.assertAlmostEquals(np.mean(scores), 0.0, places=4)

    def test_observer_aware_subjective_model_with_dscoring(self):
        subjective_model = MaximumLikelihoodEstimationModelReduced.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(dscore_mode=True)

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.090840910829083799, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.089032585621095089, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 15.681766163430936, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.012565584832977776, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 298.35293969059796, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4163670233392607, places=4)

    def test_observer_aware_subjective_model_with_zscoring(self):
        subjective_model = MaximumLikelihoodEstimationModelReduced.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(zscore_mode=True)

        self.assertAlmostEquals(np.sum(result['observer_bias']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.0, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 11.568205661696393, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.0079989301785523791, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 0.80942484781493518, places=4)

    def test_observer_aware_subjective_model_with_dscoring_and_zscoring(self):
        subjective_model = MaximumLikelihoodEstimationModelReduced.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(dscore_mode=True, zscore_mode=True)

        self.assertAlmostEquals(np.sum(result['observer_bias']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.0, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 11.628499078069273, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.0082089371266301642, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 0.80806512456121071, places=4)

    def test_observer_aware_subjective_model_use_log(self):
        subjective_model = MaximumLikelihoodEstimationModelReduced.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(use_log=True)

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.082429594509296211, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.089032585621095089, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 15.681766163430936, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.012565584832977776, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.2889206910113, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4355485462027884, places=4)

    def test_observer_content_aware_subjective_model(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['content_bias']), 0, places=4)
        self.assertAlmostEquals(np.var(result['content_bias']), 0, places=4)

        self.assertAlmostEquals(np.sum(result['content_ambiguity']), 3.8972884776604402, places=4)
        self.assertAlmostEquals(np.var(result['content_ambiguity']), 0.0041122094732031289, places=4)

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.055712761348815837, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.085842891905121704, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 10.164665557559516, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.028749990587721687, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.20774261173619, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4351342153719635, places=4)

    def test_observer_content_aware_subjective_model_missingdata(self):

        dataset = import_python_file(self.dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.1,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)

        subjective_model = MaximumLikelihoodEstimationModel(dataset_reader)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['content_bias']), 0, places=4)
        self.assertAlmostEquals(np.var(result['content_bias']), 0, places=4)

        self.assertAlmostEquals(np.sum(result['content_ambiguity']), 3.9104244772977128, places=4)
        self.assertAlmostEquals(np.var(result['content_ambiguity']), 0.0037713583509767193, places=4)

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.21903272050455846, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.084353684687185043, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 9.8168943054654481, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.028159236075789944, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.05548186797336, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4339487982797514, places=4)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.5,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)

        subjective_model = MaximumLikelihoodEstimationModel(dataset_reader)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['content_bias']), 0, places=4)
        self.assertAlmostEquals(np.var(result['content_bias']), 0, places=4)

        self.assertAlmostEquals(np.sum(result['content_ambiguity']), 2.63184284168883, places=4)
        self.assertAlmostEquals(np.var(result['content_ambiguity']), 0.019164097909450246, places=4)

        self.assertAlmostEquals(np.sum(result['observer_bias']), 0.2263148440748638, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.070613033112114504, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 12.317917502439435, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.029455722248727296, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.29962156788139, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4717366222424826, places=4)

    def test_observer_content_aware_subjective_model_nocontent(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(mode='NO_CONTENT')

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.090840910829083799, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.089032585621095089, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 15.681766163430936, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.012565584832977776, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.31447815213642, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4355485462027884, places=4)

        self.assertAlmostEquals(np.sum(result['content_bias']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['content_bias']), 0.0, places=4)

        self.assertAlmostEquals(np.sum(result['content_ambiguity']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['content_ambiguity']), 0.0, places=4)

    def test_observer_content_aware_subjective_model_nosubject(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(mode='NO_SUBJECT')

        self.assertAlmostEquals(np.sum(result['observer_bias']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.0, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.0, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.0384615384616, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4012220200639218, places=4)

        self.assertAlmostEquals(np.sum(result['content_bias']), 0.0, places=4)
        self.assertAlmostEquals(np.var(result['content_bias']), 0.0, places=4)

        self.assertAlmostEquals(np.sum(result['content_ambiguity']), 6.06982228334157, places=4)
        self.assertAlmostEquals(np.var(result['content_ambiguity']), 0.0045809756997836721, places=4)

    def test_observer_aware_subjective_model_synthetic(self):

        np.random.seed(0)

        dataset = import_python_file(self.dataset_filepath)
        info_dict = {
            'quality_scores': np.random.uniform(1, 5, 79),
            'observer_bias': np.random.normal(0, 1, 26),
            'observer_inconsistency': np.abs(np.random.uniform(0.4, 0.6, 26)),
            'content_bias': np.zeros(9),
            'content_ambiguity': np.zeros(9),
        }
        dataset_reader = SyntheticRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModelReduced(dataset_reader)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.90138622499935517, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.84819162765420342, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 12.742288471632817, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.0047638169604076975, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 236.78529213581052, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.3059726132293354, places=4)

    def test_observer_aware_subjective_model(self):
        subjective_model = MaximumLikelihoodEstimationModelReduced.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.090840910829083799, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.089032585621095089, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 15.681766163430936, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.012565584832977776, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.31447815213642, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4355485462027884, places=4)

    def test_observer_aware_subjective_model_missingdata(self):

        dataset = import_python_file(self.dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.1,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModelReduced(dataset_reader)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['observer_bias']), -0.18504017984241944, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.087350553292201705, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 15.520738471447299, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.010940587327083341, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 279.94975274863879, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4325574378911554, places=4)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.5,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModelReduced(dataset_reader)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['observer_bias']), 0.057731868199093525, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.081341845650928557, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 14.996238224489693, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.013666025579465165, places=4)

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.67100837103203, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4637917512768972, places=4)

    def test_livedmos_subjective_model(self):
        subjective_model = LiveDmosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 65.307711974116913, places=4)
        self.assertAlmostEquals(scores[10], 30.204773267864258, places=4)
        self.assertAlmostEquals(np.mean(scores), 50.0, places=4)

    def test_livedmos_subjective_model_normalize_final(self):
        subjective_model = LiveDmosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEquals(scores[0], 1.0392964273048528, places=4)
        self.assertAlmostEquals(scores[10], -1.3439701802061783, places=4)
        self.assertAlmostEquals(np.mean(scores), 0.0, places=4)

    def test_livedmos_subjective_model_dscore_mode_bad(self):
        subjective_model = LiveDmosModel.from_dataset_file(self.dataset_filepath)
        with self.assertRaises(AssertionError):
            subjective_model.run_modeling(dscore_mode=True)

    def test_observer_aware_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModelReduced(dataset_reader)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.mean(result['quality_scores']), 3.5573073781669944, places=4) # 3.5482845335713469
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.3559834438740614, places=4) # 1.4355485462027884

    def test_mos_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEquals(np.mean(scores), 3.5447906523855899, places=4)
        self.assertAlmostEquals(np.var(scores), 0.95893305294535369, places=4) # 1.4012220200639218

    def test_mos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling(subject_rejection=True)
        scores = result['quality_scores']

        self.assertAlmostEquals(np.mean(scores), 3.5611814345991566, places=4)
        self.assertAlmostEquals(np.var(scores), 1.1049505732699529, places=4) # 1.4012220200639218

    def test_zscore_mos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling(zscore_mode=True, subject_rejection=True)
        scores = result['quality_scores']

        self.assertAlmostEquals(np.mean(scores), 0.0, places=4)
        self.assertAlmostEquals(np.var(scores), 0.66670826882879042, places=4)

    def test_observer_aware_subjective_model_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModelReduced(dataset_reader)
        with self.assertRaises(AssertionError):
            result = subjective_model.run_modeling(subject_rejection=True)

    def test_observer_content_aware_subjective_model_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModel(dataset_reader)
        with self.assertRaises(AssertionError):
            result = subjective_model.run_modeling(subject_rejection=True)

    def test_observer_content_aware_subjective_dmos_model(self):
        subjective_model = MaximumLikelihoodEstimationDmosModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['quality_scores']), 288.56842946051466, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4166132275824235, places=4)

        self.assertAlmostEquals(np.sum(result['content_bias']), 0, places=4)
        self.assertAlmostEquals(np.var(result['content_bias']), 0, places=4)

        self.assertAlmostEquals(np.sum(result['content_ambiguity']), 3.8972884776604402, places=4)
        self.assertAlmostEquals(np.var(result['content_ambiguity']), 0.0041122094732031289, places=4)

        self.assertAlmostEquals(np.sum(result['observer_bias']), 3.1293776428507774, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.085842891905121704, places=4)

        self.assertAlmostEquals(np.sum(result['observer_inconsistency']), 10.164665557559516, places=4)
        self.assertAlmostEquals(np.var(result['observer_inconsistency']), 0.028749990587721687, places=4)

    def test_least_squares_model(self):
        subjective_model = LeastSquaresModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling()

        self.assertAlmostEquals(np.sum(result['quality_scores']), 280.03846153847428, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1.4012220200638821, places=4)

        self.assertAlmostEquals(np.sum(result['observer_bias']), 0, places=4)
        self.assertAlmostEquals(np.var(result['observer_bias']), 0.089032585621522581, places=4)

    def test_subjrejmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjrejMosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEquals(np.mean(scores), 3.5611814345991566, places=4)
        self.assertAlmostEquals(np.var(scores), 1.1049505732699529, places=4) # 1.4012220200639218

    def test_zscoresubjrejmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = ZscoringSubjrejMosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEquals(np.mean(scores), 0, places=4)
        self.assertAlmostEquals(np.var(scores), 0.66670826882879042, places=4) # 1.4012220200639218

    def test_subjrejdmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjrejDmosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEquals(np.mean(scores), 4.0246673158065542, places=4)
        self.assertAlmostEquals(np.var(scores), 1.0932580358187849, places=4) # 1.4012220200639218

    def test_zscoresubjrejdmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = ZscoringSubjrejDmosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEquals(np.mean(scores), 0, places=4)
        self.assertAlmostEquals(np.var(scores), 0.66405245792414114, places=4) # 1.4012220200639218

    def test_persubject_subjective_model_output(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = PerSubjectModel(dataset_reader)
        subjective_model.run_modeling(transform_final={'p1':25, 'p0':-25})
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEquals(dis_video['groundtruth'], 100.0, places=4)

if __name__ == '__main__':
    unittest.main()
