#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys
import re

from core.asset import Asset
import config
from core.executor import run_executors_in_parallel
from core.quality_runner import VmafQualityRunner

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']

def print_usage():
    parallelize = ['yes', 'no']
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " parallelize input_file [optional_model_file]\n"
    print "parallelize:\n\t" + "\n\t".join(parallelize) +"\n"
    print "input_file contains lines of:"
    print "\tfmt width height ref_file dis_file\\n"
    print "fmts:\n\t" + "\n\t".join(FMTS) +"\n"

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print_usage()
        exit(2)

    do_parallelize = sys.argv[1]
    if do_parallelize == 'yes':
        parallelize = True
    elif do_parallelize == 'no':
        parallelize = False
    else:
        print_usage()
        exit(2)

    input_filename = sys.argv[2]

    if len(sys.argv) >= 4:
        model_filepath = sys.argv[3]
    else:
        model_filepath = None

    assets = []
    line_idx = 0
    with open(input_filename, "rt") as input_file:
        for line in input_file.readlines():

            # match comment
            mo = re.match(r"^#", line)
            if mo:
                print "Skip commented line: {}".format(line)
                continue

            # match whitespace
            mo = re.match(r"[\s]+", line)
            if mo:
                continue

            # example: yuv420p 576 324 ref.yuv dis.yuv
            mo = re.match(r"([\S]+) ([0-9]+) ([0-9]+) ([\S]+) ([\S]+)", line)
            if not mo or mo.group(1) not in FMTS:
                print "Unknown format: {}".format(line)
                print_usage()
                exit(1)

            fmt = mo.group(1)
            width = int(mo.group(2))
            height = int(mo.group(3))
            ref_file = mo.group(4)
            dis_file = mo.group(5)

            asset = Asset(dataset="cmd", content_id=0, asset_id=line_idx,
                          workdir_root=config.ROOT + "/workspace/workdir",
                          ref_path=ref_file,
                          dis_path=dis_file,
                          asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                          )
            assets.append(asset)
            line_idx += 1

    runner_class = VmafQualityRunner

    optional_dict = {
        'model_filepath':model_filepath
    }

    # construct an VmafQualityRunner object to assert assets, and to remove
    runner = runner_class(assets,
                 None,
                 log_file_dir=config.ROOT + "/workspace/log_file_dir",
                 fifo_mode=True,
                 delete_workdir=True,
                 result_store=None,
                 optional_dict=optional_dict,
                 )

    try:
        # run
        runners, results = run_executors_in_parallel(
            runner_class,
            assets,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=parallelize,
            result_store=None,
            optional_dict=optional_dict,
        )

        # output
        for result in results:
            print '============================'
            print 'Asset {asset_id}:'.format(asset_id=result.asset.asset_id)
            print '============================'
            print str(result)

    finally:
        runner.remove_logs()

    print 'Done.'
