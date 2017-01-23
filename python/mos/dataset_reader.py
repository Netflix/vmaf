import pprint
import copy

import numpy as np
from tools.misc import empty_object

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class DatasetReader(object):

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self._assert_dataset()

    def _assert_dataset(self):
        # assert content id is from 0 to the total_cotent - 1
        cids = []
        for ref_video in self.dataset.ref_videos:
            cids.append(ref_video['content_id'])
        expected_cids = range(self.num_ref_videos)
        assert len(cids) == len(expected_cids) and sorted(cids) == sorted(expected_cids), \
            "reference video content_ids must range from 0 to total contents -1"

        # assert dis_video content_id is content_ids
        for dis_video in self.dataset.dis_videos:
            assert dis_video['content_id'] in cids, \
                "dis_video {dis_video} must have content_id in {cids}".format(
                    dis_video=dis_video, cids=cids)

    @property
    def num_dis_videos(self):
        return len(self.dataset.dis_videos)

    @property
    def num_ref_videos(self):
        return len(self.dataset.ref_videos)

    @property
    def content_id_of_dis_videos(self):
        return map(lambda dis_video: dis_video['content_id'], self.dataset.dis_videos)

    @property
    def _contentid_to_refvideo_map(self):
        d = {}
        for ref_video in self.dataset.ref_videos:
            d[ref_video['content_id']] = ref_video
        return d

    @property
    def disvideo_is_refvideo(self):
        d = self._contentid_to_refvideo_map
        return map(
            lambda dis_video: d[dis_video['content_id']]['path'] == dis_video['path'],
            self.dataset.dis_videos
        )

    @property
    def ref_score(self):
        return self.dataset.ref_score if hasattr(self.dataset, 'ref_score') else None

    def to_dataset(self):
        return self.dataset

class RawDatasetReader(DatasetReader):
    """
    Reader for a subjective quality test dataset with raw scores (dis_video must
    has key of 'os' (opinion score)).
    """

    def _assert_dataset(self):
        """
        Override DatasetReader._assert_dataset
        """

        super(RawDatasetReader, self)._assert_dataset()

        # assert each dis_video dict has key 'os' (opinion score), and must
        # be iterable (list, tuple or dictionary)
        for dis_video in self.dataset.dis_videos:
            assert 'os' in dis_video, "dis_video must have key 'os' (opinion score)"
            assert isinstance(dis_video['os'], list) \
                   or isinstance(dis_video['os'], tuple) \
                   or isinstance(dis_video['os'], dict)

        # make sure each dis video has equal number of observers
        if isinstance(self.dataset.dis_videos[0]['os'], list) or isinstance(self.dataset.dis_videos[0]['os'], tuple):
            num_observers = len(self.dataset.dis_videos[0]['os'])
            for dis_video in self.dataset.dis_videos[1:]:
                assert num_observers == len(dis_video['os']), \
                    "expect number of observers {expected} but got {actual} for {dis_video}".format(
                        expected=num_observers, actual=len(dis_video['os']), dis_video=str(dis_video))

        # make sure dataset has ref_score
        assert self.dataset.ref_score is not None, "dataset must have attribute ref_score"

    @property
    def num_observers(self):
        if isinstance(self.dataset.dis_videos[0]['os'], list) \
                or isinstance(self.dataset.dis_videos[0]['os'], tuple):
            return len(self.dataset.dis_videos[0]['os'])
        elif isinstance(self.dataset.dis_videos[0]['os'], dict):
            list_observers = self._get_list_observers
            return len(list_observers)
        else:
            assert False, ''

    @property
    def _get_list_observers(self):
        list_observers = []
        for dis_video in self.dataset.dis_videos:
            assert isinstance(dis_video['os'], dict)
            list_observers += dis_video['os'].keys()
        list_observers = sorted(list(set(list_observers)))  # unique, sorted
        return list_observers

    @property
    def opinion_score_2darray(self):
        """
        2darray storing raw opinion scores, with first dimension the number of
        distorted videos, second dimension the number of observers
        """
        score_mtx = float('NaN') * np.ones([self.num_dis_videos, self.num_observers])

        if isinstance(self.dataset.dis_videos[0]['os'], list) \
                or isinstance(self.dataset.dis_videos[0]['os'], tuple):
            for i_dis_video, dis_video in enumerate(self.dataset.dis_videos):
                score_mtx[i_dis_video , :] = dis_video['os']
        elif isinstance(self.dataset.dis_videos[0]['os'], dict):
            list_observers = self._get_list_observers
            for i_dis_video, dis_video in enumerate(self.dataset.dis_videos):
                for i_observer, observer in enumerate(list_observers):
                    if observer in dis_video['os']:
                        score_mtx[i_dis_video , i_observer] = dis_video['os'][observer]
        else:
            assert False
        return score_mtx

    def to_aggregated_dataset(self, aggregate_scores, **kwargs):

        newone = empty_object()
        newone.dataset_name = self.dataset.dataset_name
        newone.yuv_fmt = self.dataset.yuv_fmt
        newone.width = self.dataset.width
        newone.height = self.dataset.height

        if 'quality_width' in kwargs and kwargs['quality_width'] is not None:
            newone.quality_width = kwargs['quality_width']
        elif hasattr(self.dataset, 'quality_width'):
            newone.quality_width = self.dataset.quality_width

        if 'quality_height' in kwargs and kwargs['quality_height'] is not None:
            newone.quality_height = kwargs['quality_height']
        elif hasattr(self.dataset, 'quality_height'):
            newone.quality_height = self.dataset.quality_height

        if 'resampling_type' in kwargs and kwargs['resampling_type'] is not None:
            newone.resampling_type = kwargs['resampling_type']
        elif hasattr(self.dataset, 'resampling_type'):
            newone.resampling_type = self.dataset.resampling_type

        # ref_videos: deepcopy
        newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)

        # dis_videos: use input aggregate scores
        dis_videos = []
        assert len(self.dataset.dis_videos) == len(aggregate_scores)
        for dis_video, score in zip(self.dataset.dis_videos, aggregate_scores):
            dis_video2 = copy.deepcopy(dis_video)
            if 'os' in dis_video2: # remove 'os' - opinion score
                del dis_video2['os']
            dis_video2['groundtruth'] = score
            dis_videos.append(dis_video2)
        newone.dis_videos = dis_videos

        return newone

    def to_aggregated_dataset_file(self, dataset_filepath, aggregate_scores, **kwargs):

        aggregate_dataset = self.to_aggregated_dataset(aggregate_scores, **kwargs)

        assert(hasattr(aggregate_dataset, 'ref_videos'))
        assert(hasattr(aggregate_dataset, 'dis_videos'))

        # write out
        with open(dataset_filepath, 'wt') as output_file:
            for key in aggregate_dataset.__dict__.keys():
                if key!='ref_videos' and key!='dis_videos':
                    output_file.write('{} = '.format(key) + repr(aggregate_dataset.__dict__[key]) + '\n')

            output_file.write('\n')
            output_file.write('ref_videos = ' + pprint.pformat(aggregate_dataset.ref_videos) + '\n')
            output_file.write('\n')
            output_file.write('dis_videos = ' + pprint.pformat(aggregate_dataset.dis_videos) + '\n')

    def to_persubject_dataset(self, quality_scores, **kwargs):

        newone = empty_object()
        newone.dataset_name = self.dataset.dataset_name
        newone.yuv_fmt = self.dataset.yuv_fmt
        newone.width = self.dataset.width
        newone.height = self.dataset.height

        if 'quality_width' in kwargs and kwargs['quality_width'] is not None:
            newone.quality_width = kwargs['quality_width']
        elif hasattr(self.dataset, 'quality_width'):
            newone.quality_width = self.dataset.quality_width

        if 'quality_height' in kwargs and kwargs['quality_height'] is not None:
            newone.quality_height = kwargs['quality_height']
        elif hasattr(self.dataset, 'quality_height'):
            newone.quality_height = self.dataset.quality_height

        if 'resampling_type' in kwargs and kwargs['resampling_type'] is not None:
            newone.resampling_type = kwargs['resampling_type']
        elif hasattr(self.dataset, 'resampling_type'):
            newone.resampling_type = self.dataset.resampling_type

        # ref_videos: deepcopy
        newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)

        # dis_videos: use input aggregate scores
        dis_videos = []
        for dis_video, quality_score in zip(self.dataset.dis_videos, quality_scores):
            assert 'os' in dis_video

            # quality_score should be a 1-D array with (processed) per-subject scores
            assert hasattr(quality_score, '__len__')
            assert len(dis_video['os']) == len(quality_score)

            for persubject_score in quality_score:
                dis_video2 = copy.deepcopy(dis_video)
                if 'os' in dis_video2: # remove 'os' - opinion score
                    del dis_video2['os']
                dis_video2['groundtruth'] = persubject_score
                dis_videos.append(dis_video2)
        newone.dis_videos = dis_videos

        return newone

    def to_persubject_dataset_file(self, dataset_filepath, quality_scores, **kwargs):

        persubject_dataset = self.to_persubject_dataset(quality_scores, **kwargs)

        assert(hasattr(persubject_dataset, 'ref_videos'))
        assert(hasattr(persubject_dataset, 'dis_videos'))

        # write out
        with open(dataset_filepath, 'wt') as output_file:
            for key in persubject_dataset.__dict__.keys():
                if key!='ref_videos' and key!='dis_videos':
                    output_file.write('{} = '.format(key) + repr(persubject_dataset.__dict__[key]) + '\n')

            output_file.write('\n')
            output_file.write('ref_videos = ' + pprint.pformat(persubject_dataset.ref_videos) + '\n')
            output_file.write('\n')
            output_file.write('dis_videos = ' + pprint.pformat(persubject_dataset.dis_videos) + '\n')

class MockedRawDatasetReader(RawDatasetReader):

    def __init__(self, dataset, **kwargs):
        super(MockedRawDatasetReader, self).__init__(dataset)
        if 'input_dict' in kwargs:
            self.input_dict = kwargs['input_dict']
        else:
            self.input_dict = {}
        self._assert_input_dict()

    def to_dataset(self):
        """
        Override DatasetReader.to_dataset(). Need to overwrite dis_video['os']
        """

        newone = empty_object()
        newone.__dict__.update(self.dataset.__dict__)

        # deep copy ref_videos and dis_videos
        newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)
        newone.dis_videos = copy.deepcopy(self.dataset.dis_videos)

        # overwrite dis_video['os']
        score_mtx = self.opinion_score_2darray
        num_videos, num_subjects = score_mtx.shape
        assert num_videos == len(newone.dis_videos)
        for scores, dis_video in zip(score_mtx, newone.dis_videos):
            dis_video['os'] = list(scores)

        return newone

class SyntheticRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that generates synthetic data. It reads a dataset as baseline,
    and override the opinion_score_2darray based on input_dict.
    """
    def _assert_input_dict(self):
        assert 'quality_scores' in self.input_dict
        assert 'observer_bias' in self.input_dict
        assert 'observer_inconsistency' in self.input_dict
        assert 'content_bias' in self.input_dict
        assert 'content_ambiguity' in self.input_dict

        E = len(self.input_dict['quality_scores'])
        S = len(self.input_dict['observer_bias'])
        C = len(self.input_dict['content_bias'])
        assert len(self.input_dict['observer_inconsistency']) == S
        assert len(self.input_dict['content_ambiguity']) == C

        assert E == self.num_dis_videos
        assert S == self.num_observers
        assert C == self.num_ref_videos

    @property
    def opinion_score_2darray(self):
        """
        Override DatasetReader.opinion_score_2darray(self), based on input
        synthetic_result.
        It follows the generative model:
        Z_e,s = Q_e + X_s + Y_[c(e)]
        where Q_e is the quality score of distorted video e, and X_s ~ N(b_s, sigma_s)
        is the term representing observer s's bias (b_s) and inconsistency (sigma_s).
        Y_c ~ N(mu_c, delta_c), where c is a function of e, or c = c(e), represents
        content c's bias (mu_c) and ambiguity (delta_c).
        """

        S = self.num_observers
        E = self.num_dis_videos

        q_e = np.array(self.input_dict['quality_scores'])
        q_es = np.tile(q_e, (S, 1)).T

        b_s = np.array(self.input_dict['observer_bias'])
        sigma_s = np.array(self.input_dict['observer_inconsistency'])
        x_es = np.tile(b_s, (E, 1)) + np.random.normal(0, 1, [E, S]) * np.tile(sigma_s, (E, 1))

        mu_c = np.array(self.input_dict['content_bias'])
        delta_c = np.array(self.input_dict['content_ambiguity'])
        mu_c_e = np.array(map(lambda i: mu_c[i], self.content_id_of_dis_videos))
        delta_c_e = np.array(map(lambda i: delta_c[i], self.content_id_of_dis_videos))
        y_es = np.tile(mu_c_e, (S, 1)).T + np.random.normal(0, 1, [E, S]) * np.tile(delta_c_e, (S, 1)).T

        z_es = q_es + x_es + y_es
        return z_es

class MissingDataRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that simulates random missing data. It reads a dataset as
    baseline, and override the opinion_score_2darray based on input_dict.
    """
    def _assert_input_dict(self):
        assert 'missing_probability' in self.input_dict

    @property
    def opinion_score_2darray(self):
        score_mtx = super(MissingDataRawDatasetReader, self).opinion_score_2darray

        mask = np.random.uniform(size=score_mtx.shape)
        mask[mask > self.input_dict['missing_probability']] = 1.0
        mask[mask <= self.input_dict['missing_probability']] = float('NaN')

        return score_mtx * mask

class SelectSubjectRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that only output selected subjects. It reads a dataset as a
    baseline, and override the opinion_score_2darray and other fields based on
    input_dict.
    """
    def _assert_input_dict(self):
        assert 'selected_subjects' in self.input_dict

        selected_subjects = self.input_dict['selected_subjects']

        # assert no repeated numbers
        assert len(list(set(selected_subjects))) == len(selected_subjects)

        # assert in 0, 1, 2...., num_observer -1
        observer_idxs = range(super(SelectSubjectRawDatasetReader, self).num_observers)
        for subject in selected_subjects:
            assert subject in observer_idxs

    @property
    def num_observers(self):
        return len(self.input_dict['selected_subjects'])

    @property
    def opinion_score_2darray(self):
        """
        2darray storing raw opinion scores, with first dimension the number of
        distorted videos, second dimension the number of observers
        """
        selected_subjects = self.input_dict['selected_subjects']
        score_mtx = np.zeros([self.num_dis_videos, self.num_observers])
        for i_dis_video, dis_video in enumerate(self.dataset.dis_videos):
            score_mtx[i_dis_video , :] = np.array(dis_video['os'])[selected_subjects]
        return score_mtx

class CorruptSubjectRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that have scores of selected subjects shuffled. It reads a
    dataset as a baseline, and override the opinion_score_2darray and other
    fields based on input_dict.
    """

    def _assert_input_dict(self):
        assert 'selected_subjects' in self.input_dict

        selected_subjects = self.input_dict['selected_subjects']

        # assert no repeated numbers
        assert len(list(set(selected_subjects))) == len(selected_subjects)

        # assert in 0, 1, 2...., num_observer -1
        observer_idxs = range(super(CorruptSubjectRawDatasetReader, self).num_observers)
        for subject in selected_subjects:
            assert subject in observer_idxs

    @property
    def opinion_score_2darray(self):
        """
        2darray storing raw opinion scores, with first dimension the number of
        distorted videos, second dimension the number of observers
        """
        score_mtx = super(CorruptSubjectRawDatasetReader, self).opinion_score_2darray
        num_video, num_subject = score_mtx.shape

        # for selected subjects, shuffle its score
        selected_subjects = self.input_dict['selected_subjects']
        for subject in selected_subjects:

            if 'corrupt_probability' in self.input_dict:
                videos = list(np.where(np.random.uniform(size=num_video)
                                       < self.input_dict['corrupt_probability'])[0])
                score_mtx[videos, subject] = \
                    np.random.permutation(score_mtx[videos, subject])
            else:
                np.random.shuffle(score_mtx[:, subject])

        return score_mtx

class CorruptDataRawDatasetReader(MockedRawDatasetReader):

    """
    Dataset reader that simulates random corrupted data. It reads a dataset as
    baseline, and override the opinion_score_2darray based on input_dict.
    """
    def _assert_input_dict(self):
        assert 'corrupt_probability' in self.input_dict

    @property
    def opinion_score_2darray(self):
        score_mtx = super(CorruptDataRawDatasetReader, self).opinion_score_2darray

        mask = np.random.uniform(size=score_mtx.shape)
        mask[mask > self.input_dict['corrupt_probability']] = 1.0
        mask[mask <= self.input_dict['corrupt_probability']] = float('NaN')

        score_mtx[np.isnan(mask)] = np.random.uniform(1, self.dataset.ref_score,
                                                      np.isnan(mask).sum())

        return score_mtx