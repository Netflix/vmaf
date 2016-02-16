__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import re
from tools import get_file_name_with_extension, make_parent_dirs_if_nonexist
from asset import Asset
import config

class Result(object):
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
        self.asset = asset
        self.executor_id = executor_id
        self.result_dict = result_dict

    def __eq__(self, other):
        if self.asset != other.asset:
            return False
        if self.executor_id != other.executor_id:
            return False
        list_scores_key = self._get_list_scores_key()
        for scores_key in list_scores_key:
            if self.result_dict[scores_key] != other.result_dict[scores_key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # TODO: add __repr__, __hash__

    def __str__(self):
        return self.to_string()

    # make access dictionary-like, i.e. can do: result['vif_score']
    def __getitem__(self, key):
        return self.get_score(key)

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

    def get_score(self, key):
        try:
            return self.result_dict[key]
        except KeyError as e:
            return self._try_get_aggregate_score(key, e)

    def to_string(self):
        s = ""
        s += 'Asset: {}\n'.format( # unlike repr(asset), print workdir
            self.asset.to_full_repr())
        s += 'Executor: {}\n'.format(self.executor_id)
        s += 'Result:\n'
        s += self._get_perframe_score_str()
        s += self._get_aggregate_score_str()
        return s

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
        list_scores_key = self._get_list_scores_key()
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

    def _get_perframe_score_str(self):
        list_scores_key = self._get_list_scores_key()
        list_score_key = self._get_list_score_key()
        list_scores = map(lambda key: self.result_dict[key], list_scores_key)
        str_perframe = "\n".join(
            map(
                lambda (frame_num, scores): "Frame {}: ".format(frame_num) + (
                ", ".join(
                    map(
                        lambda (score_key, score): "{score_key}:{score:.3f}".
                            format(score_key=score_key, score=score),
                        zip(list_score_key, scores))
                )),
                enumerate(zip(*list_scores))
            )
        )
        str_perframe += '\n'
        return str_perframe

    def _get_aggregate_score_str(self):
        list_score_key = self._get_list_score_key()
        str_aggregate = "Aggregate: " + (", ".join(
            map(
                lambda (score_key, score): "{score_key}:{score:.3f}".
                    format(score_key=score_key, score=score),
                zip(
                    list_score_key, map(
                        lambda score_key: self[score_key],
                        list_score_key)
                )
            )
        ))
        return str_aggregate

    def _get_list_scores_key(self):
        # e.g. ['VMAF_scores', 'VMAF_vif_scores']
        list_scores_key = filter(lambda key: re.search(r"_scores$", key),
                                 self.result_dict.keys())
        list_scores_key = sorted(list_scores_key)
        return list_scores_key

    def _get_list_score_key(self):
        # e.g. ['VMAF_score', 'VMAF_vif_score']
        list_scores_key = self._get_list_scores_key()
        return map(lambda scores_key: scores_key[:-1], list_scores_key)

    def _try_get_aggregate_score(self, key, error):
        """
        Get aggregate score from list of scores. Must follow the convention
        that if the aggregate score uses key '*_score', then there must be
        a corresponding list of scores that uses key '*_scores'. For example,
        if the key is 'VMAF_score', there must exist a corresponding key
        'VMAF_scores'.
        :param key:
        :return:
        """
        if re.search(r"_score$", key):
            scores_key = key + 's' # e.g. 'VMAF_scores'
            if scores_key in self.result_dict:
                scores = self.result_dict[scores_key]
                return float(sum(scores)) / len(scores)
        raise KeyError(error)

class ResultStore(object):
    """
    Provide capability to save and load a Result.
    """
    pass

class FileSystemResultStore(ResultStore):
    """
    persist result by a simple file-system that save/load result in a directory.
    The directory has multiple subdirectories, each corresponding to a result
    generator (e.g. a VMAF feature extractor, or a NO19 feature extractor, or a
    VMAF quality runner, or a SSIM quality runner). Each subdirectory contains
    multiple files, each file stores dataframe for an asset, and has file name
    str(asset).
    """
    def __init__(self,
                 logger,
                 result_store_dir=config.ROOT +
                                "/workspace/result_store_dir/file_result_store"
                 ):
        self.logger = logger
        self.result_store_dir = result_store_dir

    def save(self, result):
        result_file_path = self._get_result_file_path(result)
        make_parent_dirs_if_nonexist(result_file_path)
        with open(result_file_path, "wt") as result_file:
            result_file.write(str(result.to_dataframe().to_dict()))

    def load(self, asset, executor_id):
        import pandas as pd
        import ast
        result_file_path = self._get_result_file_path2(asset, executor_id)

        if not os.path.isfile(result_file_path):
            return None

        with open(result_file_path, "rt") as result_file:
            df = pd.DataFrame.from_dict(ast.literal_eval(result_file.read()))
            result = Result.from_dataframe(df)
        return result

    def delete(self, asset, executor_id):
        result_file_path = self._get_result_file_path2(asset, executor_id)
        if os.path.isfile(result_file_path):
            os.remove(result_file_path)

    def clean_up(self):
        """
        WARNING: USE WITH CAUTION!!!
        :return:
        """
        import shutil
        if os.path.isdir(self.result_store_dir):
            shutil.rmtree(self.result_store_dir)

    def _get_result_file_path(self, result):
        return "{dir}/{executor_id}/{str}".format(dir=self.result_store_dir,
                                                  executor_id=result.executor_id,
                                                  str=str(result.asset))

    def _get_result_file_path2(self, asset, executor_id):
        return "{dir}/{executor_id}/{str}".format(dir=self.result_store_dir,
                                                  executor_id=executor_id,
                                                  str=str(asset))