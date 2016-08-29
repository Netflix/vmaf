#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys

from core.result_store import FileSystemResultStore
from tools.misc import import_python_file
from core.quality_runner import QualityRunner, VmafQualityRunner
from routine import test_on_dataset, print_matplotlib_warning


def print_usage():
    # quality_runner_types = map(lambda runner: runner.TYPE, QualityRunner.get_subclasses())
    quality_runner_types = ['PSNR', 'VMAF']
    cache_result = ['yes', 'no']
    parallelize = ['yes', 'no']
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " quality_type cache_result parallelize test_dataset_file [optional_VMAF_model_file]\n"
    print "quality_types:\n\t" + "\n\t".join(quality_runner_types) +"\n"
    print "cache_result:\n\t" + "\n\t".join(cache_result) +"\n"
    print "parallelize:\n\t" + "\n\t".join(parallelize) +"\n"

if __name__ == '__main__':

    if len(sys.argv) < 5:
        print_usage()
        exit(2)

    try:
        quality_type = sys.argv[1]
        cache_result = sys.argv[2]
        do_parallelize = sys.argv[3]
        test_dataset_filepath = sys.argv[4]
    except ValueError:
        print_usage()
        exit(2)

    if do_parallelize == 'yes':
        parallelize = True
    elif do_parallelize == 'no':
        parallelize = False
    else:
        print_usage()
        exit(2)

    if len(sys.argv) >= 6:
        model_filepath = sys.argv[5]
    else:
        model_filepath = None

    if model_filepath is not None and quality_type != VmafQualityRunner.TYPE:
        print "Input error: only quality_type VMAF accepts optional_VMAF_model_file."
        print_usage()
        exit(2)

    try:
        test_dataset = import_python_file(test_dataset_filepath)
    except Exception as e:
        print "Error: " + str(e)
        exit(1)

    try:
        runner_class = QualityRunner.find_subclass(quality_type)
    except:
        print_usage()
        exit(2)

    if cache_result == 'yes':
        result_store = FileSystemResultStore()
    elif cache_result == 'no':
        result_store = None
    else:
        print_usage()
        exit(2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
        test_on_dataset(test_dataset, runner_class, ax,
                        result_store, model_filepath,
                        parallelize=parallelize)

        bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
        ax.annotate('Testing Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print_matplotlib_warning()
        test_on_dataset(test_dataset, runner_class, None,
                        result_store, model_filepath,
                        parallelize=parallelize)

    print 'Done.'

    exit(0)