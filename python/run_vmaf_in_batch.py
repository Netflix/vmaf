#!/usr/bin/env python

import os
import sys
import re

import numpy as np

from core.asset import Asset
import config
from core.executor import run_executors_in_parallel
from core.quality_runner import VmafQualityRunner
from tools.misc import cmd_option_exists, get_cmd_option
from tools.stats import ListStats

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']
OUT_FMTS = ['text (default)', 'xml', 'json']
POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " input_file [--model model_path] [--out-fmt out_fmt] [--parallelize]\n"
    print "out_fmt:\n\t" + "\n\t".join(OUT_FMTS) + "\n"
    print "input_file contains lines of:"
    print "\tfmt width height ref_path dis_path\\n"
    print "fmt:\n\t" + "\n\t".join(FMTS) + "\n"

def main():
    if len(sys.argv) < 2:
        print_usage()
        return 2

    input_filepath = sys.argv[1]

    model_path = get_cmd_option(sys.argv, 2, len(sys.argv), '--model')

    out_fmt = get_cmd_option(sys.argv, 2, len(sys.argv), '--out-fmt')
    if not (out_fmt is None
            or out_fmt == 'xml'
            or out_fmt == 'json'
            or out_fmt == 'text'):
        print_usage()
        return 2

    pool_method = get_cmd_option(sys.argv, 2, len(sys.argv), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print '--pool can only have option among {}'.format(', '.join(POOL_METHODS))
        return 2

    parallelize = cmd_option_exists(sys.argv, 2, len(sys.argv), '--parallelize')

    assets = []
    line_idx = 0
    with open(input_filepath, "rt") as input_file:
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
                return 1

            fmt = mo.group(1)
            width = int(mo.group(2))
            height = int(mo.group(3))
            ref_file = mo.group(4)
            dis_file = mo.group(5)

            asset = Asset(dataset="cmd",
                          content_id=0,
                          asset_id=line_idx,
                          workdir_root=config.ROOT + "/workspace/workdir",
                          ref_path=ref_file,
                          dis_path=dis_file,
                          asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                          )
            assets.append(asset)
            line_idx += 1

    runner_class = VmafQualityRunner

    if model_path is None:
        optional_dict = None
    else:
        optional_dict = {'model_filepath':model_path}

    runners, results = run_executors_in_parallel(
        runner_class,
        assets,
        fifo_mode=True,
        delete_workdir=True,
        parallelize=parallelize,
        result_store=None,
        optional_dict=optional_dict,
        optional_dict2=None,
    )

    # output
    for result in results:

        # pooling
        if pool_method == 'harmonic_mean':
            result.set_score_aggregate_method(ListStats.harmonic_mean)
        elif pool_method == 'min':
            result.set_score_aggregate_method(np.min)
        elif pool_method == 'median':
            result.set_score_aggregate_method(np.median)
        elif pool_method == 'perc5':
            result.set_score_aggregate_method(ListStats.perc5)
        elif pool_method == 'perc10':
            result.set_score_aggregate_method(ListStats.perc10)
        elif pool_method == 'perc20':
            result.set_score_aggregate_method(ListStats.perc20)
        else: # None or 'mean'
            pass

        if out_fmt == 'xml':
            print result.to_xml()
        elif out_fmt == 'json':
            print result.to_json()
        else: # None or 'json'
            print '============================'
            print 'Asset {asset_id}:'.format(asset_id=result.asset.asset_id)
            print '============================'
            print str(result)

    return 0

if __name__ == "__main__":
    ret = main()
    exit(ret)
