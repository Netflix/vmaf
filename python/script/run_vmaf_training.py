#!/usr/bin/env python

import os
import sys

import numpy as np
from vmaf.config import DisplayConfig

from vmaf.tools.misc import import_python_file, cmd_option_exists, get_cmd_option
from vmaf.core.result_store import FileSystemResultStore
from vmaf.routine import print_matplotlib_warning, train_test_vmaf_on_dataset
from vmaf.tools.stats import ListStats

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

SUBJECTIVE_MODELS = ['DMOS (default)', 'DMOS_MLE', 'MLE', 'MOS', 'SR_DMOS', 'SR_MOS', 'ZS_SR_DMOS', 'ZS_SR_MOS']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) + \
        " train_dataset_filepath feature_param_filepath model_param_filepath output_model_filepath [--subj-model subjective_model] [--cache-result] [--parallelize]\n"
    print "subjective_model:\n\t" + "\n\t".join(SUBJECTIVE_MODELS) + "\n"

def main():

    if len(sys.argv) < 5:
        print_usage()
        return 2

    try:
        train_dataset_filepath = sys.argv[1]
        feature_param_filepath = sys.argv[2]
        model_param_filepath = sys.argv[3]
        output_model_filepath = sys.argv[4]
    except ValueError:
        print_usage()
        return 2

    try:
        train_dataset = import_python_file(train_dataset_filepath)
        feature_param = import_python_file(feature_param_filepath)
        model_param = import_python_file(model_param_filepath)
    except Exception as e:
        print "Error: " + str(e)
        return 1

    cache_result = cmd_option_exists(sys.argv, 3, len(sys.argv), '--cache-result')
    parallelize = cmd_option_exists(sys.argv, 3, len(sys.argv), '--parallelize')
    suppress_plot = cmd_option_exists(sys.argv, 3, len(sys.argv), '--suppress-plot')

    pool_method = get_cmd_option(sys.argv, 3, len(sys.argv), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print '--pool can only have option among {}'.format(', '.join(POOL_METHODS))
        return 2

    subj_model = get_cmd_option(sys.argv, 3, len(sys.argv), '--subj-model')

    try:
        if subj_model is not None:
            from sureal.subjective_model import SubjectiveModel
            subj_model_class = SubjectiveModel.find_subclass(subj_model)
        else:
            subj_model_class = None
    except Exception as e:
        print "Error: " + str(e)
        return 1

    if cache_result:
        result_store = FileSystemResultStore()
    else:
        result_store = None

    # pooling
    if pool_method == 'harmonic_mean':
        aggregate_method = ListStats.harmonic_mean
    elif pool_method == 'min':
        aggregate_method = np.min
    elif pool_method == 'median':
        aggregate_method = np.median
    elif pool_method == 'perc5':
        aggregate_method = ListStats.perc5
    elif pool_method == 'perc10':
        aggregate_method = ListStats.perc10
    elif pool_method == 'perc20':
        aggregate_method = ListStats.perc20
    else: # None or 'mean'
        aggregate_method = np.mean

    logger = None

    try:
        if suppress_plot:
            raise AssertionError

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)

        train_test_vmaf_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                   feature_param=feature_param, model_param=model_param,
                                   train_ax=ax, test_ax=None,
                                   result_store=result_store,
                                   parallelize=parallelize,
                                   logger=logger,
                                   output_model_filepath=output_model_filepath,
                                   aggregate_method=aggregate_method,
                                   subj_model_class=subj_model_class,
                                   )

        bbox = {'facecolor':'white', 'alpha':0.5, 'pad':20}
        ax.annotate('Training Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

        # ax.set_xlim([-10, 110])
        # ax.set_ylim([-10, 110])

        plt.tight_layout()
        DisplayConfig.show()
    except ImportError:
        print_matplotlib_warning()
        train_test_vmaf_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                   feature_param=feature_param, model_param=model_param,
                                   train_ax=None, test_ax=None,
                                   result_store=result_store,
                                   parallelize=parallelize,
                                   logger=logger,
                                   output_model_filepath=output_model_filepath,
                                   aggregate_method=aggregate_method,
                                   subj_model_class=subj_model_class,
                                   )
    except AssertionError:
        train_test_vmaf_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                   feature_param=feature_param, model_param=model_param,
                                   train_ax=None, test_ax=None,
                                   result_store=result_store,
                                   parallelize=parallelize,
                                   logger=logger,
                                   output_model_filepath=output_model_filepath,
                                   aggregate_method=aggregate_method,
                                   subj_model_class=subj_model_class,
                                   )

    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)
