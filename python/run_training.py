#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys

from tools.misc import import_python_file
from core.result_store import FileSystemResultStore
from routine import print_matplotlib_warning, train_test_vmaf_on_dataset


def print_usage():
    cache_result = ['yes', 'no']
    parallelize = ['yes', 'no']
    print "usage: " + os.path.basename(sys.argv[0]) + \
        " cache_result parallelize train_dataset_file feature_param_file model_param_file output_model_file\n"
    print "cache_result:\n\t" + "\n\t".join(cache_result) +"\n"
    print "parallelize:\n\t" + "\n\t".join(parallelize) +"\n"

if __name__ == '__main__':

    if len(sys.argv) < 7:
        print_usage()
        exit(2)

    try:
        cache_result = sys.argv[1]
        do_parallelize = sys.argv[2]
        train_dataset_filepath = sys.argv[3]
        feature_param_filepath = sys.argv[4]
        model_param_filepath = sys.argv[5]
        output_model_filepath = sys.argv[6]
    except ValueError:
        print_usage()
        exit(2)

    try:
        train_dataset = import_python_file(train_dataset_filepath)
        feature_param = import_python_file(feature_param_filepath)
        model_param = import_python_file(model_param_filepath)
    except Exception as e:
        print "Error: " + str(e)
        exit(1)

    if do_parallelize == 'yes':
        parallelize = True
    elif do_parallelize == 'no':
        parallelize = False
    else:
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

        train_test_vmaf_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                   feature_param=feature_param, model_param=model_param,
                                   train_ax=ax, test_ax=None,
                                   result_store=result_store,
                                   parallelize=parallelize,
                                   output_model_filepath=output_model_filepath
                                   )

        bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
        ax.annotate('Training Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

        # ax.set_xlim([-10, 110])
        # ax.set_ylim([-10, 110])

        plt.tight_layout()
        plt.show()
    except ImportError:
        print_matplotlib_warning()
        train_test_vmaf_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                   feature_param=feature_param, model_param=model_param,
                                   train_ax=None, test_ax=None,
                                   result_store=result_store,
                                   parallelize=parallelize,
                                   output_model_filepath=output_model_filepath
                                   )

    print 'Done.'

    exit(0)