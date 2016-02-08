#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys
import re
from asset import Asset
import config
from quality_runner import run_quality_runners_in_parallel
from vmaf_quality_runner import VmafQualityRunner

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) + " [input_file]\n"
    print "input_file contains lines of:"
    print "\t[fmt] [width] [height] [ref_file] [dis_file]\\n"
    print "fmts:\n\t" + "\n\t".join(FMTS) +"\n"

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print_usage()
        exit(0)

    input_filename = sys.argv[1]

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
                exit(0)

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

    # construct an VmafQualityRunner object merely to remove logs
    VmafQualityRunner(assets,
                      None,
                      log_file_dir=config.ROOT + "/workspace/log_file_dir",
                      fifo_mode=True,
                      delete_workdir=True).remove_logs()

    # run
    runners, results = run_quality_runners_in_parallel(
        VmafQualityRunner,
        assets,
        log_file_dir=config.ROOT + "/workspace/log_file_dir",
        fifo_mode=True,
        delete_workdir=True,
        parallelize=True)

    # output
    for asset, result in zip(assets, results):
        print '============================'
        print 'Asset {asset_id}:'.format(asset_id=asset.asset_id)
        print '============================'
        print ''
        print 'Input:'
        print asset.__dict__
        print ''
        print 'Output:'
        print str(result)
        print ''

    # clean up
    for runner in runners:
        runner.remove_logs()

    print 'Done.'
