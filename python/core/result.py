from collections import OrderedDict
import re

import numpy as np

from tools.misc import get_file_name_with_extension
from core.asset import Asset

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class BasicResult(object):
    """
    Has some basic functions, but don't need asset or executor_id. To be used by
    FeatureAssemler, which is not an Executor.
    """
    def __init__(self, asset, result_dict):
        self.asset = asset
        self.result_dict = result_dict
        self.score_aggregate_method = np.mean

    def set_score_aggregate_method(self, score_aggregate_method):
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

    def _try_get_aggregate_score(self, key, error):
        """
        Get aggregate score from list of scores. Must follow the convention
        that if the aggregate score uses key '*_score', then there must be
        a corresponding list of scores that uses key '*_scores'. For example,
        if the key is 'VMAF_score', there must exist a corresponding key
        'VMAF_scores'.
        """
        if re.search(r"_score$", key):
            scores_key = key + 's' # e.g. 'VMAF_scores'
            if scores_key in self.result_dict:
                scores = self.result_dict[scores_key]
                return self.score_aggregate_method(scores)
        raise KeyError(error)

    def get_ordered_list_scores_key(self):
        # e.g. ['VMAF_scores', 'VMAF_vif_scores']
        list_scores_key = filter(lambda key: re.search(r"_scores$", key),
                                 self.result_dict.keys())
        list_scores_key = sorted(list_scores_key)
        return list_scores_key

    def get_ordered_list_score_key(self):
        # e.g. ['VMAF_score', 'VMAF_vif_score']
        list_scores_key = self.get_ordered_list_scores_key()
        return map(lambda scores_key: scores_key[:-1], list_scores_key)

    def _get_scores_str(self, unit_name='Frame'):
        list_scores_key = self.get_ordered_list_scores_key()
        list_score_key = self.get_ordered_list_score_key()
        list_scores = map(lambda key: self.result_dict[key], list_scores_key)
        str_perframe = "\n".join(
            map(
                lambda (frame_num, scores): "{unit} {num}: ".format(
                    unit=unit_name, num=frame_num) + (
                ", ".join(
                    map(
                        lambda (score_key, score): "{score_key}:{score:.6f}".
                            format(score_key=score_key, score=score),
                        zip(list_score_key, scores))
                )),
                enumerate(zip(*list_scores))
            )
        )
        str_perframe += '\n'
        return str_perframe

    def _get_aggregate_score_str(self):
        list_score_key = self.get_ordered_list_score_key()
        str_aggregate = "Aggregate ({}): ".format(self.score_aggregate_method.__name__) + (", ".join(
            map(
                lambda (score_key, score): "{score_key}:{score:.6f}".
                    format(score_key=score_key, score=score),
                zip(
                    list_score_key, map(
                        lambda score_key: self[score_key],
                        list_score_key)
                )
            )
        ))
        return str_aggregate


class Result(BasicResult):
    """
    Dictionary-like object that stores read-only result generated on an Asset
    by a Executor.
    """
    DATAFRAME_COLUMNS = (  'dataset',
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
        Asset: {"asset_dict": {"height": 1080, "width": 1920}, "asset_id": 0, "content_id": 0, "dataset": "test", "dis_path": "/home/zli/Projects/stash/MCE/transcoder/vmaf_oss/vmaf/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv", "ref_path": "/home/zli/Projects/stash/MCE/transcoder/vmaf_oss/vmaf/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv", "workdir": "/home/zli/Projects/stash/MCE/transcoder/vmaf_oss/vmaf/workspace/workdir/d26050af-bd92-46a7-8519-7482306aa7fe"}
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
        list_scores = map(lambda key: self.result_dict[key], list_scores_key)
        list_aggregate_score = map(lambda key: self[key], list_score_key)
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

    def to_json(self):
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
        import json

        list_scores_key = self.get_ordered_list_scores_key()
        list_score_key = self.get_ordered_list_score_key()
        list_scores = map(lambda key: self.result_dict[key], list_scores_key)
        list_aggregate_score = map(lambda key: self[key], list_score_key)
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

        return json.dumps(top, sort_keys=False, indent=4)

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
        list_scores = map(lambda key: self.result_dict[key], list_scores_key)

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
    def from_dataframe(cls, df):

        # first, make sure the df conform to the format for a single asset
        cls._assert_asset_dataframe(df)

        asset_repr = df.iloc[0]['asset']
        asset = Asset.from_repr(asset_repr)

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
    RawResult is very different from core.result.Result class -- it won't be
    stored using ResultStore, neither it has aggregation method like
    result['vmaf_score'] (which calls _try_get_aggregate_score()).
    """

    def __init__(self, asset, executor_id, result_dict):
        # same interface as Result
        self.asset = asset
        self.result_dict = result_dict
        self.executor_id = executor_id

    # make access dictionary-like, i.e. can do: result['pixels'], result['asset']
    def __getitem__(self, key):
        return self.get_result(key)

    def get_result(self, key):
        # different from BasicResult.get_result, it won't try to get
        # aggregate score
        return self.result_dict[key]

    def get_ordered_results(self):
        return sorted(self.result_dict.keys())