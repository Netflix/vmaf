#!/usr/bin/env python
import os
import sys
from core.quality_runner import QualityRunner
from core.result_store import FileSystemResultStore
from routine import run_remove_results_for_dataset
from tools.misc import import_python_file

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

def print_usage():
    quality_runner_types = ['VMAF', 'PSNR', 'SSIM', 'MS_SSIM']
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " quality_type dataset_filepath\n"
    print "quality_type:\n\t" + "\n\t".join(quality_runner_types) +"\n"

def main():
    if len(sys.argv) < 3:
        print_usage()
        return 2

    try:
        quality_type = sys.argv[1]
        dataset_filepath = sys.argv[2]
    except ValueError:
        print_usage()
        return 2

    try:
        dataset = import_python_file(dataset_filepath)
    except Exception as e:
        print "Error: " + str(e)
        return 1

    try:
        runner_class = QualityRunner.find_subclass(quality_type)
    except:
        print_usage()
        return 2

    result_store = FileSystemResultStore()

    run_remove_results_for_dataset(result_store, dataset, runner_class)

    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)