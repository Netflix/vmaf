__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import re
from tools import get_file_name_with_extension
from asset import Asset

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
                           'scores_key',
                           'scores' # one score per unit - frame, chunk or else
                        )

    def __init__(self, asset, result_dict):
        self.asset = asset
        self.result_dict = result_dict

    def __eq__(self, other):
        if self.asset != other.asset:
            return False
        list_scores_key = self._get_list_scores_key()
        for scores_key in list_scores_key:
            if self.result_dict[scores_key] != other.result_dict[scores_key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # TODO: add __repr__, __hash__

    @classmethod
    def from_dataframe(cls, df):

        # first, make sure the df conform to the format for a single asset
        cls._assert_assert_dataframe(df)

        asset_repr = df.iloc[0]['asset']
        asset = Asset.from_repr(asset_repr)

        result_dict = {}
        for _, row in df.iterrows():
            result_dict[row['scores_key']] = row['scores']

        return ResultStore(asset, result_dict)

    def __str__(self):
        return self.to_string()

    # make access dictionary-like, i.e. can do: result['vif_score']
    def __getitem__(self, key):
        return self.get_score(key)

    def get_score(self, key):
        try:
            return self.result_dict[key]
        except KeyError as e:
            return self._try_get_aggregate_score(key, e)

    def to_string(self):
        s = ""
        s += 'Asset:\n'
        s += str(self.asset.__dict__) + '\n'
        s += 'Result:\n'
        str_perframe = self._get_perframe_score_str()
        s += str_perframe
        str_aggregate = self._get_aggregate_score_str()
        s += str_aggregate
        return s

    def to_dataframe(self):
        """
        Export to pandas dataframe with columns:
        dataset, content_id, asset_id, ref_name, dis_name, asset, scores_key, scores
        :return:
        """
        import pandas as pd
        asset = self.asset
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
                   scores_key,
                   scores]
            rows.append(row)

        # zip rows into a dict, and wrap with df
        df = pd.DataFrame(dict(zip(self.DATAFRAME_COLUMNS, zip(*rows))))

        return df

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

    @classmethod
    def _assert_assert_dataframe(cls, df):
        """
        Make sure the input dataframe conforms
        :param df:
        :return:
        """
        # check columns
        for col in list(df.columns.values):
            assert col in cls.DATAFRAME_COLUMNS

        # all rows should have the same dataset, content_id, asset_id, ref_name,
        # dis_name, asset
        assert len(set(df['dataset'].tolist())) == 1
        assert len(set(df['content_id'].tolist())) == 1
        assert len(set(df['asset_id'].tolist())) == 1
        assert len(set(df['ref_name'].tolist())) == 1
        assert len(set(df['dis_name'].tolist())) == 1
        assert len(set(df['asset'].tolist())) == 1

        # each scores key must have one single row
        assert len(df) == len(set(df['scores_key'].tolist()))

        # all scores should have equal length
        assert len(set(map(lambda x:len(x), df['scores'].tolist()))) == 1

