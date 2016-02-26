__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

from train_test_model import TrainTestModel
from math import floor
import itertools

class FeatureCrossValidation(object):

    @staticmethod
    def run_cross_validation(train_test_model_class, model_param,
                             dataframe, train_indices, test_indices):
        xys_train = TrainTestModel.get_xys_from_dataframe(dataframe, train_indices)
        xs_test = TrainTestModel.get_xs_from_dataframe(dataframe, test_indices)
        ys_test = TrainTestModel.get_ys_from_dataframe(dataframe, test_indices)

        train_test_model = train_test_model_class(model_param, None)
        train_test_model.train(xys_train)
        result = train_test_model.evaluate(xs_test, ys_test)

        output = {}
        output['result'] = result
        output['train_test_model'] = train_test_model
        output['content_id'] = ys_test['content_id']

        return output

    @classmethod
    def run_kfold_cross_validation(cls, train_test_model_class, model_param, dataframe, kfold):
        """
        Standard k-fold cross validation, given hyper-parameter set model_param
        :param train_test_model_class:
        :param model_param:
        :param dataframe:
        :param kfold: if it is an integer, it is the number of folds; if it is
        list of indices, then each list contains row indices of the dataframe
        selected as one fold
        :return: output
        """

        if isinstance(kfold, (int, long)):
            kfold_type = 'int'
        elif isinstance(kfold, (list, tuple)):
            kfold_type = 'list'
        else:
            assert False, 'kfold must be either a list of lists or an integer.'

        # if input is integer (e.g. 4), reconstruct kfold in list of indices
        # format
        if kfold_type == 'int':
            num_fold = kfold
            dataframe_size = len(dataframe)
            fold_size = int(floor(dataframe_size / num_fold))
            kfold = []
            for fold in range(num_fold):
                index_start = fold * fold_size
                index_end = min((fold+1)*fold_size, dataframe_size)
                kfold.append(range(index_start, index_end))

        assert len(kfold) >= 2, 'kfold list must have length >= 2 for k-fold ' \
                                'cross validation.'

        results = []
        train_test_models = []
        content_ids = []

        for fold in range(len(kfold)):

            test_index_range = kfold[fold]
            train_index_range = []
            for train_fold in range(len(kfold)):
                if train_fold != fold:
                    train_index_range += kfold[train_fold]

            output = cls.run_cross_validation(
                train_test_model_class, model_param, dataframe,
                train_index_range, test_index_range)

            result = output['result']
            train_test_model = output['train_test_model']

            results.append(result)
            train_test_models.append(train_test_model)

            content_ids += list(output['content_id'])

        aggregated_result = TrainTestModel.aggregate_stats_list(results)

        output = {}
        output['aggregated_result'] = aggregated_result
        output['results'] = results
        output['train_test_models'] = train_test_models

        assert content_ids is not None
        output['content_id'] = content_ids

        return output

    @classmethod
    def run_nested_kfold_cross_validation(cls, train_test_model_class,
                                          model_param_search_range,
                                          dataframe, kfold):
        """
        Nested k-fold cross validation, given hyper-parameter search range. The
        search range is specified in the format of, e.g.:
        {'norm_type':['normalize', 'clip_0to1', 'clip_minus1to1'],
         'n_estimators':[10, 50],
         'random_state': [0]}
        :param train_test_model_class:
        :param model_param_search_range:
        :param dataframe:
        :param kfold: if it is an integer, it is the number of folds; if it is
        lists of indices, then each list contains row indices of the dataframe
        selected as one fold
        :return: output
        """

        if isinstance(kfold, (int, long)):
            kfold_type = 'int'
        elif isinstance(kfold, (list, tuple)):
            kfold_type = 'list'
        else:
            assert False, 'kfold must be either a list of lists or an integer.'

        # if input is integer (e.g. 4), reconstruct kfold in list of indices
        # format
        if kfold_type == 'int':
            num_fold = kfold
            dataframe_size = len(dataframe)
            fold_size = int(floor(dataframe_size / num_fold))
            kfold = []
            for fold in range(num_fold):
                index_start = fold * fold_size
                index_end = min((fold+1)*fold_size, dataframe_size)
                kfold.append(range(index_start, index_end))

        assert len(kfold) >= 3, 'kfold list must have length >= 2 for nested ' \
                                'k-fold cross validation.'

        list_model_param = cls._unroll_dict_of_lists(model_param_search_range)

        results = []
        model_params = []
        content_ids = []

        for fold in range(len(kfold)):
            test_index_range = kfold[fold]
            train_index_range = []
            train_index_range_in_list_of_indices = []
                        # in this case, train_index_range is list of lists
            for train_fold in range(len(kfold)):
                if train_fold != fold:
                    train_index_range += kfold[train_fold]
                    train_index_range_in_list_of_indices.append(kfold[train_fold])

            # iterate through all possible combinations of model_params
            best_model_param = None
            best_result = None
            for model_param in list_model_param:
                output = cls.run_kfold_cross_validation(
                    train_test_model_class, model_param, dataframe,
                    train_index_range_in_list_of_indices)
                result = output['aggregated_result']

                if (best_result is None) or (
                            TrainTestModel.get_objective_score(result, type='SRCC')
                            >
                            TrainTestModel.get_objective_score(best_result, type='SRCC')
                ):
                    best_result = result
                    best_model_param = model_param

            # run cross validation based on best model parameters
            cv_output = cls.run_cross_validation(train_test_model_class,
                                                 best_model_param,
                                                 dataframe,
                                                 train_index_range,
                                                 test_index_range)
            cv_result = cv_output['result']

            results.append(cv_result)
            model_params.append(best_model_param)

            content_ids += list(cv_output['content_id'])

        aggregated_result = TrainTestModel.aggregate_stats_list(results)
        dominated_model_param, count = cls._find_most_frequent_dict(model_params)

        output = {}
        output['aggregated_result'] = aggregated_result
        output['dominated_model_param'] = dominated_model_param
        output['model_param_dominance'] = float(count) / len(model_params)
        output['results'] = results
        output['model_params'] = model_params

        assert content_ids is not None
        output['content_id'] = content_ids

        return output

    @classmethod
    def print_output(cls, output):
        if 'result' in output:
            print 'Result: {}'.format(cls.format_result(output['result']))
        if 'aggregated_result' in output:
            print 'Aggregated result: {}'.format(cls.format_result(output['aggregated_result']))
        if 'dominated_model_param' in output:
            print 'Dominated model param ({dominance:.3f}): {modelparam}'.format(
                dominance=output['model_param_dominance'],
                modelparam=output['dominated_model_param'])
        if 'results' in output and 'model_params' in output:
            for fold, (result, model_param) in enumerate(zip(output['results'], output['model_params'])):
                print 'Fold {fold}: {model_param}, {result}'.format(fold=fold, model_param=model_param, result=cls.format_result(result))

    @staticmethod
    def format_result(result):
        return '(SRCC: {srcc:.3f}, PCC: {pcc:.3f}, MSE: {rmse:.3f})'.format(
            srcc=result['SRCC'], pcc=result['PCC'], rmse=result['RMSE'])

    @staticmethod
    def _unroll_dict_of_lists(dict_of_lists):
        """
        Unfold a dictionary of lists into a list of dictionaries. For example,
        dict_of_lists = {'norm_type':['normalize'],
         'n_estimators':[10, 50],
         'random_state': [0]}
        the output list of dictionaries will be:
        [{'norm_type':'normalize', 'n_estimators':10, 'random_state':0},
         {'norm_type':'normalize', 'n_estimators':10, 'random_state':0}]
        :param dict_of_lists:
        :return:
        """
        keys = sorted(dict_of_lists.keys()) # normalize order
        list_of_key_value_pairs = []
        for key in keys:
            values = dict_of_lists[key]
            key_value_pairs = []
            for value in values:
                key_value_pairs.append((key, value))
            list_of_key_value_pairs.append(key_value_pairs)

        list_of_key_value_pairs_rearranged = itertools.product(*list_of_key_value_pairs)

        list_of_dicts = []
        for key_value_pairs in list_of_key_value_pairs_rearranged:
            list_of_dicts.append(dict(key_value_pairs))

        return list_of_dicts

    @staticmethod
    def _find_most_frequent_dict(dicts):
        """
        Find dict that appears the most frequently. The issue is to deal with
        that a dictionary is non-hashable. Workaround is to define a hash
        function.
        :param dicts:
        :return:
        """

        def _hash_dict(dict):
            return tuple(sorted(dict.items()))

        dict_count = {} # key: hash, value: list of indices
        for idx_dict, dict in enumerate(dicts):
            hash = _hash_dict(dict)
            dict_count.setdefault(hash, []).append(idx_dict)

        most_frequent_dict_hash = None
        most_frequent_dict_count = None
        for hash in dict_count:
            curr_count = len(dict_count[hash])
            if (most_frequent_dict_count is None) or \
                (most_frequent_dict_count < curr_count):
                most_frequent_dict_hash = hash
                most_frequent_dict_count = curr_count

        # find the dict matching the hash
        most_frequent_dict = None
        for dict in dicts:
            if _hash_dict(dict) == most_frequent_dict_hash:
                most_frequent_dict = dict
                break

        assert most_frequent_dict is not None

        return most_frequent_dict, most_frequent_dict_count
