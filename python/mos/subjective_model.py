import sys

import numpy as np
from scipy import linalg
from scipy import stats
import pandas as pd

from core.mixin import TypeVersionEnabled
from tools.misc import import_python_file, indices
from mos.dataset_reader import RawDatasetReader

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class SubjectiveModel(TypeVersionEnabled):
    """
    Base class for any model that takes the input of a subjective quality test
    experiment dataset with raw scores (dis_video must has key of 'os' (opinion
    score)) and output estimated quality for each impaired video (e.g. MOS, DMOS
    or more advanced estimate of subjective quality).

    A number of common functionalities are included: dscore_mode, zscore_mode,
     normalize_final, transform_final, subject_rejection
    """

    def __init__(self, dataset_reader):
        TypeVersionEnabled.__init__(self)
        self.dataset_reader = dataset_reader

    @classmethod
    def from_dataset_file(cls, dataset_filepath):
        dataset = import_python_file(dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        return cls(dataset_reader)

    def run_modeling(self, **kwargs):
        model_result = self._run_modeling(self.dataset_reader, **kwargs)
        self._postprocess_model_result(model_result, **kwargs)
        self.model_result = model_result
        return model_result

    def to_aggregated_dataset(self, **kwargs):
        self._assert_modeled()
        return self.dataset_reader.to_aggregated_dataset(
            self.model_result['quality_scores'], **kwargs)

    def to_aggregated_dataset_file(self, dataset_filepath, **kwargs):
        self._assert_modeled()
        self.dataset_reader.to_aggregated_dataset_file(
            dataset_filepath, self.model_result['quality_scores'], **kwargs)

    def _assert_modeled(self):
        assert hasattr(self, 'model_result'), \
            "self.model_result doesn't exist. Run run_modeling() first."
        assert 'quality_scores' in self.model_result, \
            "self.model_result must have quality_scores."

    @staticmethod
    def _get_ref_mos(dataset_reader, mos):
        ref_mos = []
        for dis_video in dataset_reader.dataset.dis_videos:
            # get the dis video's ref video's mos
            curr_content_id = dis_video['content_id']
            ref_indices = indices(
                zip(dataset_reader.content_id_of_dis_videos,
                    dataset_reader.disvideo_is_refvideo),
                lambda (content_id, is_refvideo):
                is_refvideo and content_id == curr_content_id
            )
            assert len(ref_indices) == 1, \
                'Should have only and one ref video for a dis video, ' \
                'but got {}'.format(len(ref_indices))
            ref_idx = ref_indices[0]

            ref_mos.append(mos[ref_idx])
        return np.array(ref_mos)

    @staticmethod
    def _get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs):

        s_es = dataset_reader.opinion_score_2darray

        # dscore_mode: True - do differential-scoring
        #              False - don't do differential-scoring
        dscore_mode= kwargs['dscore_mode'] if 'dscore_mode' in kwargs else False

        # zscore_mode: True - do z-scoring (normalizing to 0-mean 1-std)
        #              False - don't do z-scoring
        zscore_mode= kwargs['zscore_mode'] if 'zscore_mode' in kwargs else False

        # subject_rejection: True - do subject rejection
        #              False - don't do subject rejection
        subject_rejection= kwargs['subject_rejection'] if 'subject_rejection' in kwargs else False

        if dscore_mode is True:
            E, S = s_es.shape
            s_e = pd.DataFrame(s_es).mean(axis=1) # mean along s
            s_e_ref = DmosModel._get_ref_mos(dataset_reader, s_e)
            s_es = s_es + dataset_reader.ref_score - np.tile(s_e_ref, (S, 1)).T

        if zscore_mode is True:
            E, S = s_es.shape
            mu_s = pd.DataFrame(s_es).mean(axis=0) # mean along e
            simga_s = pd.DataFrame(s_es).std(ddof=1, axis=0) # std along e
            s_es = (s_es - np.tile(mu_s, (E, 1))) / np.tile(simga_s, (E, 1))

        if subject_rejection is True:
            E, S = s_es.shape

            ps = np.zeros(S)
            qs = np.zeros(S)

            for s_e in s_es:
                s_e_notnan = s_e[~np.isnan(s_e)]
                mu = np.mean(s_e_notnan)
                sigma = np.std(s_e_notnan)
                kurt = stats.kurtosis(s_e_notnan, fisher=False)

                if 2 <= kurt and kurt <= 4:
                    for idx_s, s in enumerate(s_e):
                        if not np.isnan(s):
                            if s >= mu + 2 * sigma:
                                ps[idx_s] += 1
                            if s <= mu - 2 * sigma:
                                qs[idx_s] += 1
                else:
                    for idx_s, s in enumerate(s_e):
                        if not np.isnan(s):
                            if s >= mu + np.sqrt(20) * sigma:
                                ps[idx_s] += 1
                            if s <= mu - np.sqrt(20) * sigma:
                                qs[idx_s] += 1
            rejections = []
            acceptions = []
            for idx_s, subject in zip(range(S), range(S)):
                if (ps[idx_s] + qs[idx_s]) / E > 0.05 and \
                    np.abs((ps[idx_s] - qs[idx_s]) / (ps[idx_s] + qs[idx_s])) < 0.3:
                    rejections.append(subject)
                else:
                    acceptions.append(subject)

            s_es = s_es[:, acceptions]

        return s_es

    def _postprocess_model_result(self, result, **kwargs):

        # normalize_final: True - do normalization on final quality score
        #                  False - don't do
        normalize_final= kwargs['normalize_final'] if 'normalize_final' in kwargs else False

        # transform_final: True - do (linear or other) transform on final quality score
        #                  False - don't do
        transform_final= kwargs['transform_final'] if 'transform_final' in kwargs else None

        assert 'quality_scores' in result

        if normalize_final is False:
            pass
        else:
            quality_scores = np.array(result['quality_scores'])
            quality_scores = (quality_scores - np.mean(quality_scores)) / \
                             np.std(quality_scores)
            result['quality_scores'] = list(quality_scores)

        if transform_final is None:
            pass
        else:
            quality_scores = np.array(result['quality_scores'])
            output_scores = np.zeros(quality_scores.shape)
            if 'p2' in transform_final:
                # polynomial coef of order 2
                output_scores += transform_final['p2'] * quality_scores * quality_scores
            if 'p1' in transform_final:
                # polynomial coef of order 1
                output_scores += transform_final['p1'] * quality_scores
            if 'p0' in transform_final:
                # polynomial coef of order 0
                output_scores += transform_final['p0']
            result['quality_scores'] = list(output_scores)


class MosModel(SubjectiveModel):
    """
    Mean Opinion Score (MOS) subjective model.
    """
    TYPE = 'MOS'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):
        os_2darray = cls._get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs)
        mos = pd.DataFrame(os_2darray).mean(axis=1) # mean along s, ignore NaN
        result = {'quality_scores': mos}
        return result


class DmosModel(MosModel):
    """
    Differential Mean Opinion Score (DMOS) subjective model.
    Use the formula:
    DMOS = MOS + ref_score (e.g. 5.0) - MOS_of_ref_video
    """
    TYPE = 'DMOS'
    VERSION = '1.0'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, 'DmosModel is already doing dscoring, no need to repeat.'
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        return super(DmosModel, self).run_modeling(**kwargs2)


class LiveDmosModel(SubjectiveModel):
    """
    Differential Mean Opinion Score (DMOS) subjective model based on:
    Study of Subjective and Objective Quality Assessment of Video,
    K. Seshadrinathan, R. Soundararajan, A. C. Bovik and L. K. Cormack,
    IEEE Trans. Image Processing, Vol. 19, No. 6, June 2010.

    Difference is:
    DMOS = MOS + ref_score (e.g. 5.0) - MOS_of_ref_video
    instead of
    DMOS = MOS_of_ref_video - MOS
    """
    TYPE = 'LIVE_DMOS'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, 'LiveDmosModel is already doing dscoring, no need to repeat.'

        if 'zscore_mode' in kwargs and kwargs['zscore_mode'] is True:
            assert False, 'LiveDmosModel is already doing zscoring, no need to repeat.'

        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        kwargs2['zscore_mode'] = True

        s_es = cls._get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs2)

        s_es = (s_es + 3.0) * 100.0 / 6.0

        score = pd.DataFrame(s_es).mean(axis=1) # mean along s
        result = {
            'quality_scores': score
        }
        return result


class LeastSquaresModel(SubjectiveModel):
    """
    Simple model considering:
    z_e,s = q_e + b_s
    Solve by forming linear systems and find least squares solution
    can recover q_e and b_s
    """
    TYPE = 'LS'
    VERSION = '0.1'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'SubjectAwareGenerativeModel must not and need not ' \
                          'apply subject rejection.'

        score_mtx = cls._get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs)

        num_video, num_subject = score_mtx.shape

        A = np.zeros([num_video * num_subject, num_video + num_subject])
        for idx_video in range(num_video):
            for idx_subject in range(num_subject):
                cur_row = idx_video * num_subject + idx_subject
                A[cur_row][idx_subject] = 1.0
                A[cur_row][num_subject + idx_video] = 1.0
        y = np.array(score_mtx.ravel())

        # add the extra constraint that the first ref video has score MOS
        mos = pd.DataFrame(score_mtx).mean(axis=1)
        row = np.zeros(num_subject + num_video)
        row[num_subject + 0] = 1
        score = mos[0]
        A = np.vstack([A, row])
        y = np.hstack([y, [score]])

        b_q = np.dot(linalg.pinv(A), y)
        b = b_q[:num_subject]
        q = b_q[num_subject:]

        result = {
            'quality_scores': list(q),
            'observer_bias': list(b),
        }
        return result


class MaximumLikelihoodEstimationModelReduced(SubjectiveModel):
    """
    Generative model that considers individual subject (or observer)'s bias and
    inconsistency. The observed score is modeled by:
    Z_e,s = Q_e + X_s
    where Q_e is the true quality of distorted video e, and X_s ~ N(b_s, sigma_s)
    is the term representing subject s's bias (b_s) and inconsistency (sigma_s).
    The model is then solved via likelihood maximization and belief propagation.

    Note: reduced model of MaximumLikelihoodEstimationModel without considering
    content.
    """

    # TYPE = 'Subject-Aware'
    TYPE = "MLER"
    VERSION = '0.1'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'SubjectAwareGenerativeModel must not and need not ' \
                          'apply subject rejection.'

        z_es = cls._get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs)
        E, S = z_es.shape

        use_log = kwargs['use_log'] if 'use_log' in kwargs else False

        # === initialization ===

        mos = pd.DataFrame(z_es).mean(axis=1)

        q_e = mos # use MOS as initial value for q_e
        b_s = np.zeros(S)

        x_es = z_es - np.tile(q_e, (S, 1)).T
        sigma_s = np.array(pd.DataFrame(x_es).std(axis=0, ddof=0))

        log_sigma_s = np.log(sigma_s)

        # === iteration ===

        MAX_ITR = 5000
        REFRESH_RATE = 0.1
        DELTA_THR = 1e-8

        print '=== Belief Propagation ==='

        itr = 0
        while True:

            q_e_prev = q_e

            # (8) b_s
            num = pd.DataFrame(z_es - np.tile(q_e, (S, 1)).T).sum(axis=0) # sum over e

            den = pd.DataFrame(z_es/z_es).sum(axis=0) # sum over e

            b_s_new = num / den
            b_s = b_s * (1.0 - REFRESH_RATE) + b_s_new * REFRESH_RATE

            a_es = z_es - np.tile(q_e, (S, 1)).T - np.tile(b_s, (E, 1))
            if use_log:
                # (9') log_sigma_s
                num = pd.DataFrame(-np.ones([E, S]) + a_es**2 / np.tile(sigma_s**2, (E, 1))).sum(axis=0) # sum over e

                den = pd.DataFrame(-2 * a_es**2 / np.tile(sigma_s**2, (E, 1))).sum(axis=0) # sum over e

                log_sigma_s_new = log_sigma_s - num / den
                log_sigma_s = log_sigma_s * (1.0 - REFRESH_RATE) + log_sigma_s_new * REFRESH_RATE
                sigma_s = np.exp(log_sigma_s)
            else:
                # (9) sigma_s
                num = pd.DataFrame(2 * np.ones([E, S]) * np.tile(sigma_s**3, (E, 1)) - 4 * np.tile(sigma_s, (E, 1)) * a_es**2).sum(axis=0) # sum over e

                den = pd.DataFrame(np.ones([E, S]) * np.tile(sigma_s**2, (E, 1)) - 3 * a_es**2).sum(axis=0) # sum over e

                sigma_s_new = num / den
                sigma_s = sigma_s * (1.0 - REFRESH_RATE) + sigma_s_new * REFRESH_RATE

                # sigma_s = np.maximum(sigma_s, np.zeros(sigma_s.shape))

            # (7) q_e
            num = pd.DataFrame((z_es - np.tile(b_s, (E, 1))) / np.tile(sigma_s**2, (E, 1))).sum(axis=1) # sum along s

            den = pd.DataFrame(z_es/z_es / np.tile(sigma_s**2, (E, 1))).sum(axis=1) # sum along s

            q_e_new = num / den
            q_e = q_e * (1.0 - REFRESH_RATE) + q_e_new * REFRESH_RATE

            itr += 1

            delta_q_e = linalg.norm(q_e_prev - q_e)

            msg = 'Iteration {itr:4d}: change {delta_q_e}, mean q_e {q_e}, mean b_s {b_s}, mean sigma_s {sigma_s}'.\
                format(itr=itr, delta_q_e=delta_q_e, q_e=np.mean(q_e), b_s=np.mean(b_s), sigma_s=np.mean(sigma_s))
            sys.stdout.write(msg + '\r')
            sys.stdout.flush()
            # time.sleep(0.001)

            if delta_q_e < DELTA_THR:
                break

            if itr >= MAX_ITR:
                break

        sys.stdout.write("\n")

        result = {
            'quality_scores': list(q_e),
            'observer_bias': list(b_s),
            'observer_inconsistency': list(sigma_s),
        }

        try:
            observers = dataset_reader._get_list_observers # may not exist
            result['observers'] = observers
        except AssertionError:
            pass

        return result

class MaximumLikelihoodEstimationModel(SubjectiveModel):
    """
    Generative model that considers individual subjective (or observer)'s bias
    and inconsistency, as well as content's bias and ambiguity.
    The observed score is modeled by:
    Z_e,s = Q_e + X_s + Y_[c(e)]
    where Q_e is the true quality of distorted video e, and X_s ~ N(b_s, sigma_s)
    is the term representing observer s's bias (b_s) and inconsistency (sigma_s).
    Y_c ~ N(mu_c, delta_c), where c is a function of e, or c = c(e), represents
    content c's bias (mu_c) and ambiguity (delta_c). The model is then solved
    via likelihood maximization and belief propagation.
    """

    # TYPE = 'Subject/Content-Aware'
    TYPE = 'MLE' # maximum likelihood estimation
    VERSION = '0.1'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        # mode: DEFAULT - subject and content-aware
        #       NO_SUBJECT - subject-unaware
        #       NO_CONTENT - content-unaware
        mode= kwargs['mode'] if 'mode' in kwargs else 'DEFAULT'

        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'SubjectAndContentAwareGenerativeModel must not ' \
                          'and need not apply subject rejection.'

        def sum_over_content_id(xs, cids):
            assert len(xs) == len(cids)
            num_c = np.max(cids) + 1
            assert sorted(list(set(cids))) == range(num_c)
            sums = np.zeros(num_c)
            for x, cid in zip(xs, cids):
                sums[cid] += x
            return sums

        def std_over_subject_and_content_id(x_es, cids):
            assert x_es.shape[0] == len(cids)
            num_c = np.max(cids) + 1
            assert sorted(list(set(cids))) == range(num_c)
            ls = [[] for _ in range(num_c)]
            for idx_cid, cid in enumerate(cids):
                ls[cid] = ls[cid] + list(x_es[idx_cid, :])
            stds = []
            for l in ls:
                stds.append(pd.Series(l).std(ddof=0))
            return np.array(stds)

        z_es = cls._get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs)
        E, S = z_es.shape
        C = dataset_reader.num_ref_videos

        # === initialization ===

        mos = np.array(MosModel(dataset_reader).run_modeling()['quality_scores'])

        q_e = mos # use MOS as initial value for q_e
        b_s = np.zeros(S)
        x_es = z_es - np.tile(q_e, (S, 1)).T

        if mode == 'NO_SUBJECT':
            sigma_s = np.zeros(S)
        else:
            sigma_s = pd.DataFrame(x_es).std(axis=0, ddof=0) # along e

        mu_c = np.zeros(C)

        if mode == 'NO_CONTENT':
            delta_c = np.zeros(C)
        else:
            delta_c = std_over_subject_and_content_id(
                x_es, dataset_reader.content_id_of_dis_videos)

        # === iterations ===

        MAX_ITR = 10000
        REFRESH_RATE = 0.1
        DELTA_THR = 1e-8

        print '=== Belief Propagation ==='

        itr = 0
        while True:

            q_e_prev = q_e

            # (12) b_s
            mu_c_e = np.array(map(lambda i: mu_c[i], dataset_reader.content_id_of_dis_videos))
            delta_c_e = np.array(map(lambda i: delta_c[i], dataset_reader.content_id_of_dis_videos))
            num_num = z_es - np.tile(q_e, (S, 1)).T - np.tile(mu_c_e, (S, 1)).T
            num_den = np.tile(sigma_s**2, (E, 1)) + np.tile(delta_c_e**2, (S, 1)).T
            num = pd.DataFrame(num_num / num_den).sum(axis=0) # sum over e
            den_num = z_es / z_es # 1 and nan
            den_den = num_den
            den = pd.DataFrame(den_num / den_den).sum(axis=0) # sum over e
            b_s_new = num / den
            b_s = b_s * (1.0 - REFRESH_RATE) + b_s_new * REFRESH_RATE

            if mode == 'NO_SUBJECT':
                b_s = np.zeros(S) # forcing zero, hence disabling

            # (13) mu_c
            num_num = z_es - np.tile(q_e, (S, 1)).T - np.tile(b_s, (E, 1))
            num_den = np.tile(sigma_s**2, (E, 1)) + np.tile(delta_c_e**2, (S, 1)).T
            num = pd.DataFrame(num_num / num_den).sum(axis=1) # sum over s
            num = sum_over_content_id(num, dataset_reader.content_id_of_dis_videos) # sum over e:c(e)=c
            den_num = z_es / z_es # 1 and nan
            den_den = num_den
            den = pd.DataFrame(den_num / den_den).sum(axis=1) # sum over s
            den = sum_over_content_id(den, dataset_reader.content_id_of_dis_videos) # sum over e:c(e)=c
            mu_c_new = num / den
            mu_c = mu_c * (1.0 - REFRESH_RATE) + mu_c_new * REFRESH_RATE

            # if mode == 'NO_CONTENT':
            if True: # disabling mu_c permanently, since will never able to
                     # distinguish mu_c from q_e
                mu_c = np.zeros(C) # forcing zero, hence disabling

            # (14) sigma_s
            mu_c_e = np.array(map(lambda i: mu_c[i], dataset_reader.content_id_of_dis_videos))
            delta_c_e = np.array(map(lambda i: delta_c[i], dataset_reader.content_id_of_dis_videos))
            a_es = z_es - np.tile(q_e, (S, 1)).T - np.tile(b_s, (E, 1)) - np.tile(mu_c_e, (S, 1)).T
            s2_add_d2   = np.tile(sigma_s**2, (E, 1)) + np.tile(delta_c_e**2, (S, 1)).T
            s2_minus_d2 = np.tile(sigma_s**2, (E, 1)) - np.tile(delta_c_e**2, (S, 1)).T
            num = - np.tile(sigma_s, (E, 1)) / s2_add_d2 + np.tile(sigma_s, (E, 1)) * a_es**2 / s2_add_d2**2
            num = pd.DataFrame(num).sum(axis=0) # sum over e
            poly_term =       np.tile(delta_c_e**4, (S, 1)).T \
                        - 3 * np.tile(sigma_s**4, (E, 1)) \
                        - 2 * np.tile(sigma_s**2, (E, 1)) * np.tile(delta_c_e**2, (S, 1)).T
            den = s2_minus_d2 / s2_add_d2**2 + a_es**2 * poly_term / s2_add_d2**4
            den = pd.DataFrame(den).sum(axis=0) # sum over e
            sigma_s_new = sigma_s - num / den
            sigma_s = sigma_s * (1.0 - REFRESH_RATE) + sigma_s_new * REFRESH_RATE

            # force non-negative
            sigma_s = np.maximum(sigma_s, 0.0 * np.ones(sigma_s.shape))

            if mode == 'NO_SUBJECT':
                sigma_s = np.zeros(S) # forcing zero, hence disabling

            # (15) delta_c
            mu_c_e = np.array(map(lambda i: mu_c[i], dataset_reader.content_id_of_dis_videos))
            delta_c_e = np.array(map(lambda i: delta_c[i], dataset_reader.content_id_of_dis_videos))
            a_es = z_es - np.tile(q_e, (S, 1)).T - np.tile(b_s, (E, 1)) - np.tile(mu_c_e, (S, 1)).T
            s2_add_d2   = np.tile(sigma_s**2, (E, 1)) + np.tile(delta_c_e**2, (S, 1)).T
            s2_minus_d2 = np.tile(sigma_s**2, (E, 1)) - np.tile(delta_c_e**2, (S, 1)).T
            num = - np.tile(delta_c_e, (S, 1)).T / s2_add_d2 + np.tile(delta_c_e, (S, 1)).T * a_es**2 / s2_add_d2**2
            num = pd.DataFrame(num).sum(axis=1) # sum over s
            num = sum_over_content_id(num, dataset_reader.content_id_of_dis_videos) # sum over e:c(e)=c
            poly_term =       np.tile(sigma_s**4, (E, 1)) \
                        - 3 * np.tile(delta_c_e**4, (S, 1)).T \
                        - 2 * np.tile(sigma_s**2, (E, 1)) * np.tile(delta_c_e**2, (S, 1)).T
            den = - s2_minus_d2 / s2_add_d2**2 + a_es**2 * poly_term / s2_add_d2**4
            den = pd.DataFrame(den).sum(axis=1) # sum over s
            den = sum_over_content_id(den, dataset_reader.content_id_of_dis_videos) # sum over e:c(e)=c
            delta_c_new = delta_c - num / den
            delta_c = delta_c * (1.0 - REFRESH_RATE) + delta_c_new * REFRESH_RATE

            # force non-negative
            delta_c = np.maximum(delta_c, 0.0 * np.ones(delta_c.shape))

            if mode == 'NO_CONTENT':
                delta_c = np.zeros(C) # forcing zero, hence disabling

            # (11) q_e
            mu_c_e = np.array(map(lambda i: mu_c[i], dataset_reader.content_id_of_dis_videos))
            delta_c_e = np.array(map(lambda i: delta_c[i], dataset_reader.content_id_of_dis_videos))
            num_num = z_es - np.tile(b_s, (E, 1)) - np.tile(mu_c_e, (S, 1)).T
            num_den = np.tile(sigma_s**2, (E, 1)) + np.tile(delta_c_e**2, (S, 1)).T
            num = pd.DataFrame(num_num / num_den).sum(axis=1) # sum over s
            den_num = z_es / z_es # 1 and nan
            den_den = num_den
            den = pd.DataFrame(den_num / den_den).sum(axis=1) # sum over s
            q_e_new = num / den
            q_e = q_e * (1.0 - REFRESH_RATE) + q_e_new * REFRESH_RATE

            itr += 1

            delta_q_e = linalg.norm(q_e_prev - q_e)

            msg = 'Iteration {itr:4d}: change {delta_q_e}, mean q_e {q_e}, mean b_s {b_s}, mean mu_c {mu_c}, mean sigma_s {sigma_s}, mean delta_c {delta_c}'.\
                format(itr=itr, delta_q_e=delta_q_e, q_e=np.mean(q_e), b_s=np.mean(b_s), mu_c=np.mean(mu_c), sigma_s=np.mean(sigma_s), delta_c=np.mean(delta_c))
            sys.stdout.write(msg + '\r')
            sys.stdout.flush()
            # time.sleep(0.001)

            if delta_q_e < DELTA_THR:
                break

            if itr >= MAX_ITR:
                break

        sys.stdout.write("\n")

        result = {
            'quality_scores': list(q_e),
            'observer_bias': list(b_s),
            'observer_inconsistency': list(sigma_s),
            'content_bias': list(mu_c),
            'content_ambiguity': list(delta_c),
        }

        return result

class SubjrejMosModel(MosModel):

    TYPE = 'SR_MOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'SubjrejMosModel is ' \
                          'already doing subject rejection, no need to repeat.'
        kwargs2 = kwargs.copy()
        kwargs2['subject_rejection'] = True
        return super(SubjrejMosModel, self).run_modeling(**kwargs2)

class ZscoringSubjrejMosModel(MosModel):

    TYPE = 'ZS_SR_MOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'zscore_mode' in kwargs and kwargs['zscore_mode'] is True:
            assert False, 'ZscoringSubjrejMosModel is ' \
                          'already doing zscoring, no need to repeat.'
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'ZscoringSubjrejMosModel is ' \
                          'already doing subject rejection, no need to repeat.'
        kwargs2 = kwargs.copy()
        kwargs2['zscore_mode'] = True
        kwargs2['subject_rejection'] = True
        return super(ZscoringSubjrejMosModel, self).run_modeling(**kwargs2)

class MaximumLikelihoodEstimationDmosModel(MaximumLikelihoodEstimationModel):

    TYPE = 'DMOS_MLE'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, 'SubjectAndContentAwareGenerativeDmosModel is ' \
                          'already doing dscoring, no need to repeat.'
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        return super(MaximumLikelihoodEstimationDmosModel, self).run_modeling(**kwargs2)

class SubjrejDmosModel(MosModel):

    TYPE = 'SR_DMOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, 'SubjrejDmosModel is ' \
                          'already doing dscoring, no need to repeat.'
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'SubjrejDmosModel is ' \
                          'already doing subject rejection, no need to repeat.'
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        kwargs2['subject_rejection'] = True
        return super(SubjrejDmosModel, self).run_modeling(**kwargs2)

class ZscoringSubjrejDmosModel(MosModel):

    TYPE = 'ZS_SR_DMOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, 'ZscoringSubjrejDmosModel is ' \
                          'already doing dscoring, no need to repeat.'
        if 'zscore_mode' in kwargs and kwargs['zscore_mode'] is True:
            assert False, 'ZscoringSubjrejDmosModel is ' \
                          'already doing zscoring, no need to repeat.'
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'ZscoringSubjrejDmosModel is ' \
                          'already doing subject rejection, no need to repeat.'
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        kwargs2['zscore_mode'] = True
        kwargs2['subject_rejection'] = True
        return super(ZscoringSubjrejDmosModel, self).run_modeling(**kwargs2)

class PerSubjectModel(SubjectiveModel):
    """
    Subjective model that takes a raw dataset and output a 'per-subject dataset'
    with repeated disvideos, each assigned a per-subject score
    """
    TYPE = 'PERSUBJECT'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):
        os_2darray = cls._get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs)
        result = {'quality_scores': os_2darray}
        return result

    def to_aggregated_dataset(self, **kwargs):
        self._assert_modeled()
        return self.dataset_reader.to_persubject_dataset(self.model_result['quality_scores'], **kwargs)

    def to_aggregated_dataset_file(self, dataset_filepath, **kwargs):
        self._assert_modeled()
        self.dataset_reader.to_persubject_dataset_file(dataset_filepath, self.model_result['quality_scores'], **kwargs)

