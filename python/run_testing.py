#!/usr/bin/env python

import os
import sys

import numpy as np

from core.result_store import FileSystemResultStore
from tools.misc import import_python_file, get_cmd_option, cmd_option_exists
from core.quality_runner import QualityRunner, VmafQualityRunner
from routine import test_on_dataset, print_matplotlib_warning
from tools.stats import ListStats
from mos.subjective_model import SubjectiveModel

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

SUBJECTIVE_MODELS = ['DMOS (default)', 'DMOS_MLE', 'MLE', 'MOS', 'SR_DMOS', 'SR_MOS', 'ZS_SR_DMOS', 'ZS_SR_MOS']

def print_usage():
    quality_runner_types = ['VMAF', 'PSNR', 'SSIM', 'MS_SSIM']
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " quality_type test_dataset_filepath [--vmaf-model VMAF_model_path] [--subj-model subjective_model] [--cache-result] [--parallelize] [--print-result]\n"
    print "quality_type:\n\t" + "\n\t".join(quality_runner_types) +"\n"
    print "subjective_model:\n\t" + "\n\t".join(SUBJECTIVE_MODELS) + "\n"

def main():
    if len(sys.argv) < 3:
        print_usage()
        return 2

    try:
        quality_type = sys.argv[1]
        test_dataset_filepath = sys.argv[2]
    except ValueError:
        print_usage()
        return 2

    vmaf_model_path = get_cmd_option(sys.argv, 3, len(sys.argv), '--vmaf-model')
    cache_result = cmd_option_exists(sys.argv, 3, len(sys.argv), '--cache-result')
    parallelize = cmd_option_exists(sys.argv, 3, len(sys.argv), '--parallelize')
    print_result = cmd_option_exists(sys.argv, 3, len(sys.argv), '--print-result')
    suppress_plot = cmd_option_exists(sys.argv, 3, len(sys.argv), '--suppress-plot')

    pool_method = get_cmd_option(sys.argv, 3, len(sys.argv), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print '--pool can only have option among {}'.format(', '.join(POOL_METHODS))
        return 2

    subj_model = get_cmd_option(sys.argv, 3, len(sys.argv), '--subj-model')

    try:
        if subj_model is not None:
            subj_model_class = SubjectiveModel.find_subclass(subj_model)
        else:
            subj_model_class = None
    except Exception as e:
        print "Error: " + str(e)
        return 1

    if vmaf_model_path is not None and quality_type != VmafQualityRunner.TYPE:
        print "Input error: only quality_type of VMAF accepts --vmaf-model."
        print_usage()
        return 2

    try:
        test_dataset = import_python_file(test_dataset_filepath)
    except Exception as e:
        print "Error: " + str(e)
        return 1

    try:
        runner_class = QualityRunner.find_subclass(quality_type)
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

    try:
        if suppress_plot:
            raise AssertionError

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)

        assets, results = test_on_dataset(test_dataset, runner_class, ax,
                        result_store, vmaf_model_path,
                        parallelize=parallelize,
                        aggregate_method=aggregate_method,
                        subj_model_class=subj_model_class,
                        )

        bbox = {'facecolor':'white', 'alpha':0.5, 'pad':20}
        ax.annotate('Testing Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

        # ax.set_xlim([-10, 110])
        # ax.set_ylim([-10, 110])

        plt.tight_layout()
        plt.show()
    except ImportError:
        print_matplotlib_warning()
        assets, results = test_on_dataset(test_dataset, runner_class, None,
                        result_store, vmaf_model_path,
                        parallelize=parallelize,
                        aggregate_method=aggregate_method,
                        subj_model_class=subj_model_class,
                        )
    except AssertionError:
        assets, results = test_on_dataset(test_dataset, runner_class, None,
                        result_store, vmaf_model_path,
                        parallelize=parallelize,
                        aggregate_method=aggregate_method,
                        subj_model_class=subj_model_class,
                        )

    if print_result:
        for result in results:
            print result
            print ''

    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)