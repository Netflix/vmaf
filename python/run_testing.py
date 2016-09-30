#!/usr/bin/env python

import os
import sys

from core.result_store import FileSystemResultStore
from tools.misc import import_python_file, get_cmd_option, cmd_option_exists
from core.quality_runner import QualityRunner, VmafQualityRunner
from routine import test_on_dataset, print_matplotlib_warning

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

def print_usage():
    quality_runner_types = ['VMAF', 'PSNR', 'SSIM', 'MS_SSIM']
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " quality_type test_dataset_filepath [--vmaf-model VMAF_model_path] [--cache-result] [--parallelize]\n"
    print "quality_type:\n\t" + "\n\t".join(quality_runner_types) +"\n"

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

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
        test_on_dataset(test_dataset, runner_class, ax,
                        result_store, vmaf_model_path,
                        parallelize=parallelize)

        bbox = {'facecolor':'white', 'alpha':0.5, 'pad':20}
        ax.annotate('Testing Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

        # ax.set_xlim([-10, 110])
        # ax.set_ylim([-10, 110])

        plt.tight_layout()
        plt.show()
    except ImportError:
        print_matplotlib_warning()
        test_on_dataset(test_dataset, runner_class, None,
                        result_store, vmaf_model_path,
                        parallelize=parallelize)

    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)