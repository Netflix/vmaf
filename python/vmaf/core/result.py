from collections import OrderedDict
import json
import re
from typing import Optional, Callable

import numpy as np

from vmaf.tools.misc import get_file_name_with_extension
from vmaf.core.asset import Asset

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class BasicResult(object):
    """
    Has some basic functions, but don't need asset or executor_id. To be used by
    FeatureAssembler, which is not an Executor.
    """
    def __init__(self, asset, result_dict):
        self.asset = asset
        self.result_dict = result_dict
        self.score_aggregate_method = np.mean

    def set_score_aggregate_method(self, score_aggregate_method: Optional[Callable]):
        if score_aggregate_method is not None:
            self.score_aggregate_method = score_aggregate_method
        else:
            self.score_aggregate_method = np.mean

    # make access dictionary-like, i.e. can do: result['vif_score']
    def __getitem__(self, key):
        return self.get_result(key)

    def get_result(self, key):
        try:
            return self.result_dict[key]
        except KeyError as e:
            return self._try_get_aggregate_score(key, e)

    @staticmethod
    def get_scores_key_from_score_key(score_key):
        # e.g. 'VMAF_scores'
        return score_key + 's'

    def _try_get_aggregate_score(self, key, error):
        """
        Get aggregate score from list of scores. Must follow the convention
        that if the aggregate score uses key '*_score', then there must be
        a corresponding list of scores that uses key '*_scores'. For example,
        if the key is 'VMAF_score', there must exist a corresponding key
        'VMAF_scores'.
        The list of scores (when present) should also follow the convention
        that IF multiple models are present (e.g. in bootstrapping),
        the first dimension will be the model and the
        second dimension the video frames. This means that a 2D result array
        should be created ONLY when multiple models are present.
        """
        if re.search(r"_score$", key):
            scores_key = self.get_scores_key_from_score_key(key)
            if scores_key in self.result_dict:
                scores = self.result_dict[scores_key]
                # if scores are in a list, wrap in an array to be consistent
                if type(scores) is list:
                    scores = np.asarray(scores)
                # dimension assertion: scores should be either 1-D (one prediction per frame) or 2-D (single/multiple predictions per frame)
                assert scores.ndim <= 2, 'Per frame score aggregation is not well-defined; scores cannot be saved in a N-D array with N > 2.'

                # check if there are more than one models (first dimension)
                if scores.ndim == 2:
                    # check that there are > 1 models present
                    assert scores.shape[0] > 1, '# models is <=1, but a 2D result array (models x frames) was used.'
                    # check that there are >= 1 frames predicted
                    assert scores.shape[1] >= 1, '# predicted frames is < 1.'
                    # apply score aggregation on each individual model
                    # a scores "piece" corresponds to a single model's predictions over all frames
                    return [self.score_aggregate_method(scores_piece) for scores_piece in scores]
                else:
                    # just one prediction per frame
                    return self.score_aggregate_method(scores)
        raise KeyError(error)

    def get_ordered_list_scores_key(self):
        # e.g. ['VMAF_scores', 'VMAF_vif_scores']
        # filter out scores that are > 1D (e.g. when having multiple models in bootstrapping)
        list_scores_key = filter(lambda key: re.search(r"_scores$", key) and np.asarray(self.result_dict[key]).ndim == 1,
                                 self.result_dict.keys())
        list_scores_key = sorted(list_scores_key)
        return list_scores_key

    def get_ordered_list_multimodel_scores_key(self):
        # e.g. ['BOOTSTRAP_VMAF_all_models_scores']
        # select only scores that are > 1D (e.g. when having multiple models in bootstrapping)
        list_scores_key = filter(lambda key: re.search(r"_scores$", key) and np.asarray(self.result_dict[key]).ndim > 1,
                                 self.result_dict.keys())
        list_scores_key = sorted(list_scores_key)
        return list_scores_key

    def get_ordered_list_score_key(self):
        # e.g. ['VMAF_score', 'VMAF_vif_score']
        list_scores_key = self.get_ordered_list_scores_key()
        return list(map(lambda scores_key: scores_key[:-1], list_scores_key))

    def get_ordered_list_multimodel_score_key(self):
        # e.g. ['BOOTSTRAP_VMAF_all_models_score']
        list_scores_key = self.get_ordered_list_multimodel_scores_key()
        return list(map(lambda scores_key: scores_key[:-1], list_scores_key))

    def _get_scores_str(self, unit_name='Frame'):
        list_scores_key = self.get_ordered_list_scores_key()
        list_score_key = self.get_ordered_list_score_key()
        list_scores = list(map(lambda key: self.result_dict[key], list_scores_key))
        str_perframe = "\n".join(
            list(map(
                lambda tframe_scores: "{unit} {num}: ".format(
                    unit=unit_name, num=tframe_scores[0]) + (
                ", ".join(
                    list(map(
                        lambda tscore: "{score_key}:{score:.6f}".format(score_key=tscore[0], score=tscore[1]),
                        zip(list_score_key, tframe_scores[1])))
                )),
                enumerate(zip(*list_scores))
            ))
        )
        str_perframe += '\n'
        return str_perframe

    def _get_aggregate_score_str(self):
        list_score_key = self.get_ordered_list_score_key()
        str_aggregate = "Aggregate ({}): ".format(self.score_aggregate_method.__name__) + (", ".join(
            list(map(
                lambda tscore: "{score_key}:{score:.6f}".format(score_key=tscore[0], score=tscore[1]),
                zip(
                    list_score_key, list(map(
                        lambda score_key: self[score_key],
                        list_score_key)
                ))
            ))
        ))
        return str_aggregate

    @staticmethod
    def scores_key_wildcard_match(result_dict, scores_key):
        """
        >>> BasicResult.scores_key_wildcard_match({'VMAF_integer_feature_vif_scale0_egl_1_scores': [0.983708]}, 'VMAF_integer_feature_vif_scale0_scores')
        'VMAF_integer_feature_vif_scale0_egl_1_scores'
        >>> BasicResult.scores_key_wildcard_match({'VMAF_integer_feature_vif_scale0_egl_1_scores': [0.983708]}, 'VMAF_integer_feature_vif_scale1_scores')
        Traceback (most recent call last):
        ...
        KeyError: 'no key matches VMAF_integer_feature_vif_scale1_scores'
        >>> d = {'VMAF_feature_adm_den_scale0_scores': [111.207703], \
                 'VMAF_feature_adm_den_scale1_scores': [109.423508], \
                 'VMAF_feature_adm_den_scale2_scores': [196.427551], \
                 'VMAF_feature_adm_den_scale3_scores': [328.864075], \
                 'VMAF_feature_adm_den_scores': [745.922836], \
                 'VMAF_feature_adm_num_scale0_scores': [107.988159], \
                 'VMAF_feature_adm_num_scale1_scores': [96.965546], \
                 'VMAF_feature_adm_num_scale2_scores': [178.08934], \
                 'VMAF_feature_adm_num_scale3_scores': [317.582733], \
                 'VMAF_feature_adm_num_scores': [700.625778], \
                 'VMAF_feature_adm_scores': [0.939274], \
                 'VMAF_feature_anpsnr_scores': [41.921087], \
                 'VMAF_feature_ansnr_scores': [30.230282], \
                 'VMAF_feature_motion2_scores': [0.0], \
                 'VMAF_feature_motion_scores': [0.0], \
                 'VMAF_feature_vif_den_scale0_scores': [30884050.0], \
                 'VMAF_feature_vif_den_scale1_scores': [7006361.0], \
                 'VMAF_feature_vif_den_scale2_scores': [1696758.0], \
                 'VMAF_feature_vif_den_scale3_scores': [429003.59375], \
                 'VMAF_feature_vif_den_scores': [40016172.59375], \
                 'VMAF_feature_vif_num_scale0_scores': [18583060.0], \
                 'VMAF_feature_vif_num_scale1_scores': [5836731.5], \
                 'VMAF_feature_vif_num_scale2_scores': [1545453.125], \
                 'VMAF_feature_vif_num_scale3_scores': [408037.5], \
                 'VMAF_feature_vif_num_scores': [26373282.125], \
                 'VMAF_feature_vif_scores': [0.659066]}
        >>> BasicResult.scores_key_wildcard_match(d, 'VMAF_feature_adm_num_scores')
        'VMAF_feature_adm_num_scores'
        >>> BasicResult.scores_key_wildcard_match(d, 'VMAF_feature_adm_num_scale_scores')
        Traceback (most recent call last):
        ...
        KeyError: "more than one keys matches VMAF_feature_adm_num_scale_scores: ['VMAF_feature_adm_num_scale0_scores', 'VMAF_feature_adm_num_scale1_scores', 'VMAF_feature_adm_num_scale2_scores', 'VMAF_feature_adm_num_scale3_scores']"
        >>> e = {'VMAF_feature_adm_den_egl_1_scores': [290.148182], \
                 'VMAF_feature_adm_den_scale0_egl_1_scores': [36.143059], \
                 'VMAF_feature_adm_den_scale1_egl_1_scores': [56.709286], \
                 'VMAF_feature_adm_den_scale2_egl_1_scores': [83.719971], \
                 'VMAF_feature_adm_den_scale3_egl_1_scores': [113.575867], \
                 'VMAF_feature_adm_num_egl_1_scores': [277.796806], \
                 'VMAF_feature_adm_num_scale0_egl_1_scores': [35.399879], \
                 'VMAF_feature_adm_num_scale1_egl_1_scores': [54.699654], \
                 'VMAF_feature_adm_num_scale2_egl_1_scores': [80.061874], \
                 'VMAF_feature_adm_num_scale3_egl_1_scores': [107.635399], \
                 'VMAF_feature_adm_scale0_egl_1_scores': [0.979438], \
                 'VMAF_feature_anpsnr_scores': [24.463257], \
                 'VMAF_feature_ansnr_scores': [11.619336], \
                 'VMAF_feature_motion2_scores': [0.0], \
                 'VMAF_feature_motion_scores': [0.0], \
                 'VMAF_feature_vif_den_egl_1_scores': [584014.634277], \
                 'VMAF_feature_vif_den_scale0_egl_1_scores': [452763.03125], \
                 'VMAF_feature_vif_den_scale1_egl_1_scores': [100037.859375], \
                 'VMAF_feature_vif_den_scale2_egl_1_scores': [24712.830078], \
                 'VMAF_feature_vif_den_scale3_egl_1_scores': [6500.913574], \
                 'VMAF_feature_vif_num_egl_1_scores': [576352.736328], \
                 'VMAF_feature_vif_num_scale0_egl_1_scores': [445400.1875], \
                 'VMAF_feature_vif_num_scale1_egl_1_scores': [99781.773438], \
                 'VMAF_feature_vif_num_scale2_egl_1_scores': [24675.099609], \
                 'VMAF_feature_vif_num_scale3_egl_1_scores': [6495.675781], \
                 'VMAF_feature_vif_scale0_egl_1_scores': [0.983738]}
        >>> BasicResult.scores_key_wildcard_match(e, 'VMAF_feature_adm_num_scores')
        'VMAF_feature_adm_num_egl_1_scores'
        """

        # first look for exact match
        result_keys_sorted = sorted(result_dict.keys())
        for result_key in result_keys_sorted:
            if result_key == scores_key:
                return result_key

        # then look for wildcard match
        matched_result_keys = []
        for result_key in result_keys_sorted:
            if result_key.startswith(scores_key[:-len('_scores')]):
                matched_result_keys.append(result_key)
        if len(matched_result_keys) == 0:
            raise KeyError(f"no key matches {scores_key}")
        elif len(matched_result_keys) == 1:
            return matched_result_keys[0]
        else:
            # look for shortest match
            strlens = [len(s) for s in matched_result_keys]
            argmin_strlen = np.concatenate(np.where(strlens == np.min(strlens))).tolist()
            if len(argmin_strlen) == 1:
                return matched_result_keys[argmin_strlen[0]]
            else:
                raise KeyError(f"more than one keys matches {scores_key}: {matched_result_keys}")


class Result(BasicResult):
    """
    Dictionary-like object that stores read-only result generated on an Asset
    by a Executor.
    """
    DATAFRAME_COLUMNS = (
        'dataset',
        'content_id',
        'asset_id',
        'ref_name',
        'dis_name',
        'asset',
        'executor_id',
        'scores_key',
        'scores' # one score per unit - frame, chunk or else
    )

    def __init__(self, asset, executor_id, result_dict):
        super(Result, self).__init__(asset, result_dict)
        self.executor_id = executor_id

    def __eq__(self, other):
        if self.asset != other.asset:
            return False
        if self.executor_id != other.executor_id:
            return False
        list_scores_key = self.get_ordered_list_scores_key()
        for scores_key in list_scores_key:
            if self.result_dict[scores_key] != other.result_dict[scores_key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # TODO: add __repr__, __hash__

    def __str__(self):
        return self.to_string()

    @staticmethod
    def get_unique_from_dataframe(df, scores_key, column):
        """
        Convenience method to access dataframe. Do two things 1) make assertion
        that only one row correspond to the scores_key, 2) retrive a column
        from that row
        :param df:
        :param scores_key:
        :param column:
        :return:
        """
        _df = df.loc[df['scores_key'] == scores_key]
        assert len(_df) == 1
        return _df.iloc[0][column]

    def to_string(self):
        """Example:
        Asset: {"asset_dict": {"height": 1080, "width": 1920}, "asset_id": 0, "content_id": 0, "dataset": "test",
            "dis_path": ".../vmaf/python/test/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv",
            "ref_path": ".../vmaf/python/test/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv",
            "workdir": ".../vmaf/workspace/workdir/d26050af-bd92-46a7-8519-7482306aa7fe"}
        Executor: SSIM_V1.0
        Result:
        Frame 0: SSIM_feature_ssim_c_score:0.997404, SSIM_feature_ssim_l_score:0.965512, SSIM_feature_ssim_s_score:0.935803, SSIM_score:0.901161
        Frame 1: SSIM_feature_ssim_c_score:0.997404, SSIM_feature_ssim_l_score:0.965512, SSIM_feature_ssim_s_score:0.935803, SSIM_score:0.901160
        Frame 2: SSIM_feature_ssim_c_score:0.997404, SSIM_feature_ssim_l_score:0.965514, SSIM_feature_ssim_s_score:0.935804, SSIM_score:0.901163
        Aggregate: SSIM_feature_ssim_c_score:0.997404, SSIM_feature_ssim_l_score:0.965513, SSIM_feature_ssim_s_score:0.935803, SSIM_score:0.901161
        """
        s = ""
        s += 'Asset: {}\n'.format( # unlike repr(asset), print workdir
            self.asset.to_full_repr())
        s += 'Executor: {}\n'.format(self.executor_id)
        s += 'Result:\n'
        s += self._get_scores_str()
        s += self._get_aggregate_score_str()
        return s

    def to_xml(self):
        """Example:
        <?xml version="1.0" ?>
        <result executorId="SSIM_V1.0">
          <asset identifier="test_0_0_checkerboard_1920_1080_10_3_0_0_1920x1080_vs_checkerboard_1920_1080_10_3_1_0_1920x1080_q_1920x1080"/>
          <frames>
            <frame SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.965512" SSIM_feature_ssim_s_score="0.935803" SSIM_score="0.901161" frameNum="0"/>
            <frame SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.965512" SSIM_feature_ssim_s_score="0.935803" SSIM_score="0.90116" frameNum="1"/>
            <frame SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.965514" SSIM_feature_ssim_s_score="0.935804" SSIM_score="0.901163" frameNum="2"/>
          </frames>
          <aggregate SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.965512666667" SSIM_feature_ssim_s_score="0.935803333333" SSIM_score="0.901161333333"/>
        </result>
        """

        from xml.etree import ElementTree
        from xml.dom import minidom

        list_scores_key = self.get_ordered_list_scores_key()
        list_score_key = self.get_ordered_list_score_key()
        list_scores = list(map(lambda key: self.result_dict[key], list_scores_key))
        list_aggregate_score = list(map(lambda key: self[key], list_score_key))

        list_multimodel_scores_key = self.get_ordered_list_multimodel_scores_key()
        list_multimodel_score_key = self.get_ordered_list_multimodel_score_key()
        # here we need to transpose, since printing is per frame and not per model
        # we also need to turn the 2D array to a list of lists, for unpacking to work as expected
        list_multimodel_scores = list(map(lambda key: self.result_dict[key].T.tolist(), list_multimodel_scores_key))
        list_aggregate_multimodel_score = list(map(lambda key: self[key], list_multimodel_score_key))

        # append multimodel scores and keys (if any)
        list_scores_key += list_multimodel_scores_key
        list_score_key += list_multimodel_score_key
        list_scores += list_multimodel_scores
        list_aggregate_score += list_aggregate_multimodel_score

        list_scores_reordered = zip(*list_scores)

        def prettify(elem):
            rough_string = ElementTree.tostring(elem, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")

        top = ElementTree.Element('result')
        top.set('executorId', self.executor_id)

        asset = ElementTree.SubElement(top, 'asset')
        asset.set('identifier', str(self.asset))

        frames = ElementTree.SubElement(top, 'frames')
        for i, list_score in enumerate(list_scores_reordered):
            frame = ElementTree.SubElement(frames, 'frame')
            frame.set('frameNum', str(i))
            for score_key, score in zip(list_score_key, list_score):
                frame.set(score_key, str(score))

        aggregate = ElementTree.SubElement(top, 'aggregate')
        aggregate.set('method', self.score_aggregate_method.__name__)
        for score_key, score in zip(list_score_key, list_aggregate_score):
            aggregate.set(score_key, str(score))

        return prettify(top)

    @classmethod
    def from_xml(cls, xml_string):

        from xml.etree import ElementTree

        top = ElementTree.fromstring(xml_string)
        asset = top.find('asset').get('identifier')
        executor_id = top.get('executorId')

        result_dict = {}

        frames = top.find('frames')

        for frame_ind, frame in enumerate(frames):
            for score_key, score_value in frame.attrib.items():
                if 'score' in score_key:
                    # same convention of 'score' and 'scores' as in _try_get_aggregate_score
                    if 'feature' in score_key:
                        result_dict.setdefault(Result.get_scores_key_from_score_key(score_key), []).append(np.float64(score_value))
                    else:
                        pred_result_key = Result.get_scores_key_from_score_key(score_key)
                        result_dict.setdefault(pred_result_key, []).append(np.float64(score_value))

        result_dict[pred_result_key] = np.array(result_dict[pred_result_key])

        return Result(asset, executor_id, result_dict)

    @classmethod
    def from_json(cls, json_string):

        json_data = json.loads(json_string)
        executor_id = json_data['executorId']
        asset = json_data['asset']['identifier']

        result_dict = {}

        frames = json_data['frames']

        for frame_ind, frame in enumerate(frames):
            for score_key, score_value in frame.items():
                if 'score' in score_key:
                    # same convention of 'score' and 'scores' as in _try_get_aggregate_score
                    if 'feature' in score_key:
                        result_dict.setdefault(Result.get_scores_key_from_score_key(score_key), []).append(np.float64(score_value))
                    else:
                        pred_result_key = Result.get_scores_key_from_score_key(score_key)
                        result_dict.setdefault(pred_result_key, []).append(np.float64(score_value))

        result_dict[pred_result_key] = np.array(result_dict[pred_result_key])

        return Result(asset, executor_id, result_dict)

    @classmethod
    def combine_result(cls, results):

        assert len(results) > 0, "Results list is empty."
        executor_ids = [result.executor_id for result in results]
        assert len(set(executor_ids)) == 1, "Executor ids do not match."
        executor_id = executor_ids[0]
        combined_result_dict = OrderedDict()

        sorted_scores_keys = sorted([key for key in results[0].result_dict.keys()], reverse=True)

        # initialize dictionary
        for scores_key in sorted_scores_keys:
            combined_result_dict[scores_key] = []

        for result in results:
            result_dict  = result.result_dict
            # assert if the keys in result_dict match
            assert sorted([key for key in result_dict.keys()], reverse=True) == sorted_scores_keys, "Score keys do not match."
            for scores_key in sorted_scores_keys:
                combined_result_dict[scores_key] += list(result_dict[scores_key])

        pred_result_key = [key for key in sorted_scores_keys if 'feature' not in key][0]
        combined_result_dict[pred_result_key] = np.array(combined_result_dict[pred_result_key])

        return Result('combined_asset', executor_id, combined_result_dict)

    def to_dict(self):
        """Example:
        {
            "executorId": "SSIM_V1.0",
            "asset": {
                "identifier": "test_0_0_checkerboard_1920_1080_10_3_0_0_1920x1080_vs_checkerboard_1920_1080_10_3_1_0_1920x1080_q_1920x1080"
            },
            "frames": [
                {
                    "frameNum": 0,
                    "SSIM_feature_ssim_c_score": 0.997404,
                    "SSIM_feature_ssim_l_score": 0.965512,
                    "SSIM_feature_ssim_s_score": 0.935803,
                    "SSIM_score": 0.901161
                },
                {
                    "frameNum": 1,
                    "SSIM_feature_ssim_c_score": 0.997404,
                    "SSIM_feature_ssim_l_score": 0.965512,
                    "SSIM_feature_ssim_s_score": 0.935803,
                    "SSIM_score": 0.90116
                },
                {
                    "frameNum": 2,
                    "SSIM_feature_ssim_c_score": 0.997404,
                    "SSIM_feature_ssim_l_score": 0.965514,
                    "SSIM_feature_ssim_s_score": 0.935804,
                    "SSIM_score": 0.901163
                }
            ],
            "aggregate": {
                "SSIM_feature_ssim_c_score": 0.99740399999999996,
                "SSIM_feature_ssim_l_score": 0.96551266666666669,
                "SSIM_feature_ssim_s_score": 0.93580333333333332,
                "SSIM_score": 0.90116133333333337
            }
        }
        """
        list_scores_key = self.get_ordered_list_scores_key()
        list_score_key = self.get_ordered_list_score_key()
        list_scores = list(map(lambda key: self.result_dict[key], list_scores_key))
        list_aggregate_score = list(map(lambda key: self[key], list_score_key))

        list_multimodel_scores_key = self.get_ordered_list_multimodel_scores_key()
        list_multimodel_score_key = self.get_ordered_list_multimodel_score_key()
        # here we need to transpose, since printing is per frame and not per model
        # we also need to turn the 2D array to a list of lists, for unpacking to work as expected
        list_multimodel_scores = list(map(lambda key: self.result_dict[key].T.tolist(), list_multimodel_scores_key))
        list_aggregate_multimodel_score = list(map(lambda key: self[key], list_multimodel_score_key))

        # append multimodel scores and keys (if any)
        list_scores_key += list_multimodel_scores_key
        list_score_key += list_multimodel_score_key
        list_scores += list_multimodel_scores
        list_aggregate_score += list_aggregate_multimodel_score

        list_scores_reordered = zip(*list_scores)

        top = OrderedDict()
        top['executorId'] = self.executor_id
        top['asset'] = {'identifier': str(self.asset)}

        top['frames'] = []
        for i, list_score in enumerate(list_scores_reordered):
            frame = OrderedDict()
            frame['frameNum'] = i
            for score_key, score in zip(list_score_key, list_score):
                frame[score_key] = score
            top['frames'].append(frame)
        top['aggregate'] = OrderedDict()

        for score_key, score in zip(list_score_key, list_aggregate_score):
            top['aggregate'][score_key] = score
        top['aggregate']['method'] = self.score_aggregate_method.__name__

        return top

    def to_json(self):
        """
        :return str: JSON representation
        """
        return json.dumps(self.to_dict(), sort_keys=False, indent=4)

    def to_dataframe(self):
        """
        Export to pandas dataframe with columns: dataset, content_id, asset_id,
        ref_name, dis_name, asset, executor_id, scores_key, scores
        Example:
                                                       asset  asset_id  content_id  \
        0  {"asset_dict": {"height": 1080, "width": 1920}...         0           0
        1  {"asset_dict": {"height": 1080, "width": 1920}...         0           0
        2  {"asset_dict": {"height": 1080, "width": 1920}...         0           0
        3  {"asset_dict": {"height": 1080, "width": 1920}...         0           0
        4  {"asset_dict": {"height": 1080, "width": 1920}...         0           0

          dataset                             dis_name executor_id  \
        0    test  checkerboard_1920_1080_10_3_1_0.yuv   VMAF_V0.1
        1    test  checkerboard_1920_1080_10_3_1_0.yuv   VMAF_V0.1
        2    test  checkerboard_1920_1080_10_3_1_0.yuv   VMAF_V0.1
        3    test  checkerboard_1920_1080_10_3_1_0.yuv   VMAF_V0.1
        4    test  checkerboard_1920_1080_10_3_1_0.yuv   VMAF_V0.1

                                      ref_name  \
        0  checkerboard_1920_1080_10_3_0_0.yuv
        1  checkerboard_1920_1080_10_3_0_0.yuv
        2  checkerboard_1920_1080_10_3_0_0.yuv
        3  checkerboard_1920_1080_10_3_0_0.yuv
        4  checkerboard_1920_1080_10_3_0_0.yuv

                                                  scores          scores_key
        0                  [0.798588, 0.84287, 0.800122]     VMAF_adm_scores
        1               [12.420815, 12.41775, 12.416308]   VMAF_ansnr_scores
        2                    [0.0, 18.489031, 18.542355]  VMAF_motion_scores
        3  [42.1117149479, 47.6544689539, 40.6168118533]         VMAF_scores
        4                 [0.156106, 0.156163, 0.156119]     VMAF_vif_scores

        [5 rows x 9 columns]
        :return:
        """
        import pandas as pd
        asset = self.asset
        executor_id = self.executor_id
        list_scores_key = self.get_ordered_list_scores_key()
        list_scores = list(map(lambda key: self.result_dict[key], list_scores_key))

        rows = []
        for scores_key, scores in zip(list_scores_key, list_scores):
            row = [asset.dataset,
                   asset.content_id,
                   asset.asset_id,
                   get_file_name_with_extension(asset.ref_path),
                   get_file_name_with_extension(asset.dis_path),
                   repr(asset),
                   executor_id,
                   scores_key,
                   scores]
            rows.append(row)

        # zip rows into a dict, and wrap with df
        df = pd.DataFrame(dict(zip(self.DATAFRAME_COLUMNS, zip(*rows))))

        return df

    @classmethod
    def from_dataframe(cls, df, AssetClass=Asset):

        # first, make sure the df conform to the format for a single asset
        cls._assert_asset_dataframe(df)

        asset_repr = df.iloc[0]['asset']
        asset = AssetClass.from_repr(asset_repr)

        executor_id = df.iloc[0]['executor_id']

        result_dict = {}
        for _, row in df.iterrows():
            result_dict[row['scores_key']] = row['scores']

        return Result(asset, executor_id, result_dict)

    @classmethod
    def _assert_asset_dataframe(cls, df):
        """
        Make sure that the input dataframe conforms w.r.t. an asset
        :param df:
        :return:
        """
        # check columns
        for col in list(df.columns.values):
            assert col in cls.DATAFRAME_COLUMNS

        # all rows should have the same dataset, content_id, asset_id, ref_name,
        # dis_name, asset and executor_id
        assert len(set(df['dataset'].tolist())) == 1
        assert len(set(df['content_id'].tolist())) == 1
        assert len(set(df['asset_id'].tolist())) == 1
        assert len(set(df['ref_name'].tolist())) == 1
        assert len(set(df['dis_name'].tolist())) == 1
        assert len(set(df['asset'].tolist())) == 1
        assert len(set(df['executor_id'].tolist())) == 1

        # each scores key must have one single row
        assert len(df) == len(set(df['scores_key'].tolist()))

        # all scores should have equal length
        assert len(set(map(lambda x:len(x), df['scores'].tolist()))) == 1


class RawResult(object):
    """
    Other than sharing the name 'Result' and have same initialization interface,
    RawResult is very different from vmaf.core.result.Result class -- it won't be
    stored using ResultStore, neither it has aggregation method like
    result['vmaf_score'] (which calls _try_get_aggregate_score()).
    """

    def __init__(self, asset, executor_id, result_dict):
        # same interface as Result
        self.asset = asset
        self.executor_id = executor_id
        self.result_dict = result_dict

    # make access dictionary-like, i.e. can do: result['pixels'], result['asset']
    def __getitem__(self, key):
        return self.get_result(key)

    def get_result(self, key):
        # different from BasicResult.get_result, it won't try to get
        # aggregate score
        return self.result_dict[key]

    def get_ordered_results(self):
        return sorted(self.result_dict.keys())


if __name__ == '__main__':
    import doctest
    doctest.testmod()
