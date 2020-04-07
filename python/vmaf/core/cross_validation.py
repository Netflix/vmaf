from vmaf.tools.misc import unroll_dict_of_lists

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import random
from math import floor


try:
    long
except NameError:
    # TODO: remove this once python2 support is dropped, in python3 all int are as long as you want
    long = int


class ModelCrossValidation(object):

    @staticmethod
    def run_cross_validation(train_test_model_class,
                             model_param,
                             results_or_df,
                             train_indices,
                             test_indices,
                             optional_dict2=None
                             ):
        """
        Simple cross validation.
        :param train_test_model_class:
        :param model_param:
        :param results_or_df: list of BasicResult, or pandas.DataFrame
        :param train_indices:
        :param test_indices:
        :return:
        """
        xys_train = train_test_model_class.get_xys_from_results(results_or_df, train_indices)
        xs_test = train_test_model_class.get_xs_from_results(results_or_df, test_indices)
        ys_test = train_test_model_class.get_ys_from_results(results_or_df, test_indices)

        model = train_test_model_class(model_param,
                                       logger=None,
                                       optional_dict2=optional_dict2)
        model.train(xys_train)
        stats = model.evaluate(xs_test, ys_test)

        output = {'stats': stats, 'model': model,
                  'contentids': ys_test['content_id']}

        return output

    @classmethod
    def run_kfold_cross_validation(cls,
                                   train_test_model_class,
                                   model_param,
                                   results_or_df,
                                   kfold,
                                   logger=None,
                                   optional_dict2=None):
        """
        Standard k-fold cross validation, given hyper-parameter set model_param
        :param train_test_model_class:
        :param model_param:
        :param results_or_df: list of BasicResult, or pandas.DataFrame
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
            dataframe_size = len(results_or_df)
            fold_size = int(floor(dataframe_size / num_fold))
            kfold = []
            for fold in range(num_fold):
                index_start = fold * fold_size
                index_end = min((fold+1)*fold_size, dataframe_size)
                kfold.append(range(index_start, index_end))

        assert len(kfold) >= 2, 'kfold list must have length >= 2 for k-fold ' \
                                'cross validation.'

        statss = []
        models = []
        contentids = []

        for fold in range(len(kfold)):

            # to avoid interference among folds
            if hasattr(train_test_model_class, 'reset'):
                train_test_model_class.reset()

            if logger:
                logger.info("Fold {}...".format(fold))

            test_index_range = kfold[fold]
            train_index_range = []
            for train_fold in range(len(kfold)):
                if train_fold != fold:
                    train_index_range += kfold[train_fold]

            output = cls.run_cross_validation(train_test_model_class,
                                              model_param,
                                              results_or_df,
                                              train_index_range,
                                              test_index_range,
                                              optional_dict2)

            stats = output['stats']
            model = output['model']

            statss.append(stats)
            models.append(model)

            contentids += list(output['contentids'])

        aggr_stats = train_test_model_class.aggregate_stats_list(statss)

        output = {'aggr_stats': aggr_stats, 'statss': statss, 'models': models}

        assert contentids is not None
        output['contentids'] = contentids

        return output

    @classmethod
    def run_nested_kfold_cross_validation(cls,
                                          train_test_model_class,
                                          model_param_search_range,
                                          results_or_df,
                                          kfold,
                                          search_strategy='grid',
                                          random_search_times=100,
                                          logger=None,
                                          optional_dict2=None,
                                          ):
        """
        Nested k-fold cross validation, given hyper-parameter search range. The
        search range is specified in the format of, e.g.:
        {'norm_type':['normalize', 'clip_0to1', 'clip_minus1to1'],
         'n_estimators':[10, 50],
         'random_state': [0]}
        :param train_test_model_class:
        :param model_param_search_range:
        :param results_or_df: list of BasicResult, or pandas.DataFrame
        :param kfold: if it is an integer, it is the number of folds; if it is
        lists of indices, then each list contains row indices of the dataframe
        selected as one fold
        :param search_strategy: either 'grid' or 'random'
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
            dataframe_size = len(results_or_df)
            fold_size = int(floor(dataframe_size / num_fold))
            kfold = []
            for fold in range(num_fold):
                index_start = fold * fold_size
                index_end = min((fold+1)*fold_size, dataframe_size)
                kfold.append(range(index_start, index_end))

        assert len(kfold) >= 3, 'kfold list must have length >= 2 for nested ' \
                                'k-fold cross validation.'

        if search_strategy == 'grid':
            cls._assert_grid_search(model_param_search_range)
            list_model_param = unroll_dict_of_lists(
                model_param_search_range)
        elif search_strategy == 'random':
            cls._assert_random_search(model_param_search_range)
            list_model_param = cls._sample_model_param_list(
                model_param_search_range, random_search_times)
        else:
            assert False, "Unknown search_strategy: {}".format(search_strategy)

        statss = []
        model_params = []
        contentids = []

        for fold in range(len(kfold)):

            if logger:
                logger.info("Fold {}...".format(fold))

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
            best_stats = None
            for model_param in list_model_param:

                if logger:
                    logger.info("\tModel parameter: {}".format(model_param))

                output = \
                    cls.run_kfold_cross_validation(train_test_model_class,
                                                   model_param,
                                                   results_or_df,
                                                   train_index_range_in_list_of_indices,
                                                   optional_dict2)
                stats = output['aggr_stats']

                if (best_stats is None) or (
                    train_test_model_class.get_objective_score(stats, type='SRCC')
                    >
                    train_test_model_class.get_objective_score(best_stats, type='SRCC')
                ):
                    best_stats = stats
                    best_model_param = model_param

            # run cross validation based on best model parameters
            output_ = cls.run_cross_validation(train_test_model_class,
                                              best_model_param,
                                              results_or_df,
                                              train_index_range,
                                              test_index_range,
                                               optional_dict2)
            stats_ = output_['stats']

            statss.append(stats_)
            model_params.append(best_model_param)

            contentids += list(output_['contentids'])

        aggr_stats = train_test_model_class.aggregate_stats_list(statss)
        top_model_param, count = cls._find_most_frequent_dict(model_params)

        assert contentids is not None
        output__ = {
            'aggr_stats':aggr_stats,
            'top_model_param':top_model_param,
            'top_ratio':float(count) / len(model_params),
            'statss':statss,
            'model_params':model_params,
            'contentids':contentids,
        }

        return output__

    @classmethod
    def _assert_grid_search(cls, model_param_search_range):
        assert isinstance(model_param_search_range, dict)
        # for grid search, model_param_search_range's values must all be lists
        # or tuples
        for v in model_param_search_range.values():
            assert isinstance(v, (list, tuple))

    @classmethod
    def _assert_random_search(cls, model_param_search_range):
        assert isinstance(model_param_search_range, dict)
        # for random search, model_param_search_range's values must either be
        # lists/tuples, or dictionary containing 'low' and 'high' bounds.
        for v in model_param_search_range.values():
            assert (isinstance(v, (list, tuple))
                    or
                    (isinstance(v, dict) and 'low' in v
                     and 'high' in v and 'decimal' in v)
                    )

    @classmethod
    def print_output(cls, output):
        if 'stats' in output:
            print('Stats: {}'.format(cls.format_stats(output['stats'])))
        if 'aggr_stats' in output:
            print('Aggregated stats: {}'.format(cls.format_stats(output['aggr_stats'])))
        if 'top_model_param' in output:
            print('Top model param ({ratio:.3f}): {modelparam}'.format(
                ratio=output['top_ratio'],
                modelparam=output['top_model_param']))
        if 'statss' in output and 'model_params' in output:
            for fold, (stats, model_param) in \
                    enumerate(zip(output['statss'], output['model_params'])):
                print('Fold {fold}: {model_param}, {stats}'.format(
                    fold=fold, model_param=model_param,
                    stats=cls.format_stats(stats)))

    @staticmethod
    def format_stats(stats):
        return '(SRCC: {srcc:.3f}, PCC: {pcc:.3f}, MSE: {rmse:.3f})'.format(
            srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'])

    @staticmethod
    def _sample_model_param_list(model_param_search_range, random_search_times):
        keys = sorted(model_param_search_range.keys()) # normalize order
        list_of_dicts = []
        for i in range(random_search_times):
            d = {}
            for k in keys:
                v = model_param_search_range[k]
                if isinstance(v, (list, tuple)):
                    d[k] = random.choice(v)
                elif isinstance(v, dict) and 'low' in v and 'high' in v and 'decimal' in v:
                    num = random.uniform(v['low'], v['high'])
                    scale = (10**v['decimal'])
                    num = int(num * scale) / float(scale)
                    d[k] = num
                else:
                    assert False
            list_of_dicts.append(d)

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
            if most_frequent_dict_count is None or most_frequent_dict_count < curr_count:
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
