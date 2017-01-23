__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

import numpy as np

import config
from tools.misc import import_python_file, indices
from mos.dataset_reader import RawDatasetReader, SyntheticRawDatasetReader, \
    MissingDataRawDatasetReader, SelectSubjectRawDatasetReader, \
    CorruptSubjectRawDatasetReader, CorruptDataRawDatasetReader

class RawDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        self.dataset = import_python_file(dataset_filepath)
        self.dataset_reader = RawDatasetReader(self.dataset)

    def test_read_dataset_stats(self):
        self.assertEquals(self.dataset_reader.num_ref_videos, 9)
        self.assertEquals(self.dataset_reader.num_dis_videos, 79)
        self.assertEquals(self.dataset_reader.num_observers, 26)

    def test_opinion_score_2darray(self):
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertAlmostEquals(np.mean(os_2darray), 3.544790652385589, places=4)
        self.assertAlmostEquals(np.mean(np.std(os_2darray, axis=1)), 0.64933186478291516, places=4)

    def test_dis_videos_content_ids(self):
        content_ids = self.dataset_reader.content_id_of_dis_videos
        self.assertAlmostEquals(np.mean(content_ids), 3.8607594936708862, places=4)

    def test_disvideo_is_refvideo(self):
        l = self.dataset_reader.disvideo_is_refvideo
        self.assertItemsEqual(indices(l, lambda e: e is True), range(9))

    def test_ref_score(self):
        self.assertEqual(self.dataset_reader.ref_score, 5.0)

    def test_to_persubject_dataset_wrong_dim(self):
        with self.assertRaises(AssertionError):
            dataset = self.dataset_reader.to_persubject_dataset(np.zeros(3000))
            self.assertEqual(len(dataset.dis_videos), 2054)

    def test_to_persubject_dataset(self):
        dataset = self.dataset_reader.to_persubject_dataset(np.zeros([79, 26]))
        self.assertEqual(len(dataset.dis_videos), 2054)

class SyntheticDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'quality_scores': np.random.randint(1, 6, 79),
            'observer_bias': np.random.normal(0, 1, 26),
            'observer_inconsistency': np.abs(np.random.normal(0, 0.1, 26)),
            'content_bias': np.zeros(9),
            'content_ambiguity': np.zeros(9),
        }

        self.dataset_reader = SyntheticRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEquals(self.dataset_reader.num_ref_videos, 9)
        self.assertEquals(self.dataset_reader.num_dis_videos, 79)
        self.assertEquals(self.dataset_reader.num_observers, 26)

    def test_opinion_score_2darray(self):
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertAlmostEquals(np.mean(os_2darray), 3.1912209428772669, places=4)

    def test_dis_videos_content_ids(self):
        content_ids = self.dataset_reader.content_id_of_dis_videos
        self.assertAlmostEquals(np.mean(content_ids), 3.8607594936708862, places=4)

    def test_disvideo_is_refvideo(self):
        l = self.dataset_reader.disvideo_is_refvideo
        self.assertItemsEqual(indices(l, lambda e: e is True), range(9))

    def test_ref_score(self):
        self.assertEqual(self.dataset_reader.ref_score, 5.0)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEquals(old_scores, new_scores)

class MissingDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.1,
        }

        self.dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)

    def test_opinion_score_2darray(self):
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertTrue(np.isnan(np.mean(os_2darray)))
        self.assertEquals(np.isnan(os_2darray).sum(), 201)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEquals(old_scores, new_scores)

class SelectedSubjectDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }

        self.dataset_reader = SelectSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEquals(self.dataset_reader.num_ref_videos, 9)
        self.assertEquals(self.dataset_reader.num_dis_videos, 79)
        self.assertEquals(self.dataset_reader.num_observers, 5)

    def test_opinion_score_2darray(self):
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertEquals(os_2darray.shape, (79, 5))

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEquals(old_scores, new_scores)

class CorruptSubjectDatasetReaderTestWithCorruptionProb(unittest.TestCase):

    def setUp(self):
        dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        self.dataset = import_python_file(dataset_filepath)

        np.random.seed(0)

    def test_opinion_score_2darray_with_corruption_prob(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.0,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertEquals(os_2darray.shape, (79, 26))
        self.assertAlmostEquals(np.mean(np.std(os_2darray, axis=1)), 0.64933186478291516, places=4)

    def test_opinion_score_2darray_with_corruption_prob2(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.2,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertEquals(os_2darray.shape, (79, 26))
        self.assertAlmostEquals(np.mean(np.std(os_2darray, axis=1)), 0.73123067709849221, places=4)

    def test_opinion_score_2darray_with_corruption_prob3(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.7,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertEquals(os_2darray.shape, (79, 26))
        self.assertAlmostEquals(np.mean(np.std(os_2darray, axis=1)), 0.85118397722242856, places=4)

    def test_opinion_score_2darray_with_corruption_prob4(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 1.0,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertEquals(os_2darray.shape, (79, 26))
        self.assertAlmostEquals(np.mean(np.std(os_2darray, axis=1)), 0.96532565883975119, places=4)


class CorruptSubjectDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }

        self.dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEquals(self.dataset_reader.num_ref_videos, 9)
        self.assertEquals(self.dataset_reader.num_dis_videos, 79)
        self.assertEquals(self.dataset_reader.num_observers, 26)

    def test_opinion_score_2darray(self):
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertEquals(os_2darray.shape, (79, 26))
        self.assertAlmostEquals(np.mean(np.std(os_2darray, axis=1)), 0.93177573807000225, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEquals(old_scores, new_scores)

class CorruptDataDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = config.ROOT + '/python/test/resource/NFLX_dataset_public_raw.py'
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'corrupt_probability': 0.1,
        }

        self.dataset_reader = CorruptDataRawDatasetReader(dataset, input_dict=info_dict)

    def test_opinion_score_2darray(self):
        os_2darray = self.dataset_reader.opinion_score_2darray
        self.assertAlmostEquals(np.mean(np.std(os_2darray, axis=1)), 0.79796204942957094, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEquals(old_scores, new_scores)

if __name__ == '__main__':
    unittest.main()
