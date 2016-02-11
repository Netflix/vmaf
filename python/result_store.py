__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import re
from tools import get_file_name_without_extension

class ResultStore(object):
    """
    Stores read-only result generated on an asset by feature extraction or
    train test model.
    """
    DATAFRAME_COLUMNS = (  'dataset',
                           'content_id',
                           'asset_id',
                           'ref_name',
                           'dis_name',
                           'asset',
                           'sn', # serial number for frame, chunk etc.
                           'score_key',
                           'score'
                        )

    def __init__(self, asset, result_dict):
        self.asset = asset
        self._result_dict = result_dict

    @classmethod
    def _assert_dataframe(cls, df):
        """
        Make sure the input dataframe conforms
        :param df:
        :return:
        """
        # check columns
        for col in list(df.columns.values):
            assert col in cls.DATAFRAME_COLUMNS

        # all rows should have the same dataset, content_id, asset_id, ref_name,
        # dis_name, asset string

        # all rows with the same score_key must each have a unique sn

    @classmethod
    def from_dataframe(cls, df):

        cls._assert_dataframe(df)

        asset = None
        result_dict = None
        return ResultStore(None, None)

    def __str__(self):
        return self.to_string()

    # make access dictionary-like, i.e. can do: result['vif_score']
    def __getitem__(self, key):
        return self.get_score(key)

    def get_score(self, key):
        try:
            return self._result_dict[key]
        except KeyError as e:
            return self._try_get_aggregate_score(key, e)

    def to_string(self):
        str = ""
        str_perframe = self._get_perframe_score_str()
        str += str_perframe
        str_aggregate = self._get_aggregate_score_str()
        str += str_aggregate
        return str

    def to_dataframe(self):
        """
        Export to pandas dataframe with columns:
        dataset, content_id, asset_id, ref_name, dis_name, asset_str,
        frame_number, score_key, score
        :return:
        """
        import pandas as pd
        asset = self.asset
        list_scores_key = self._get_list_scores_key()
        list_score_key = self._get_list_score_key()
        list_scores = map(lambda key: self._result_dict[key], list_scores_key)

        df = pd.DataFrame(columns=self.DATAFRAME_COLUMNS)
        for sn, scores in enumerate(zip(*list_scores)):
            for score_key, score in zip(list_score_key, scores):
                row = [asset.dataset,
                       asset.content_id,
                       asset.asset_id,
                       get_file_name_without_extension(asset.ref_path),
                       get_file_name_without_extension(asset.dis_path),
                       str(asset),
                       sn,
                       score_key,
                       score]
                df.loc[len(df)] = row

        return df

    def _get_perframe_score_str(self):
        list_scores_key = self._get_list_scores_key()
        list_score_key = self._get_list_score_key()
        list_scores = map(lambda key: self._result_dict[key], list_scores_key)
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
                                 self._result_dict.keys())
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
            if scores_key in self._result_dict:
                scores = self._result_dict[scores_key]
                return float(sum(scores)) / len(scores)
        raise KeyError(error)