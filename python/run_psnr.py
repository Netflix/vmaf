#!/usr/bin/env python
from functools import partial

import sys
import os

import numpy as np

import config
from core.asset import Asset
from core.quality_runner import PsnrQualityRunner
from tools.misc import get_cmd_option
from tools.stats import ListStats

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']
OUT_FMTS = ['text (default)', 'xml', 'json']
POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) \
          + " fmt width height ref_path dis_path [--out-fmt out_fmt]\n"
    print "fmt:\n\t" + "\n\t".join(FMTS) + "\n"
    print "out_fmt:\n\t" + "\n\t".join(OUT_FMTS) + "\n"

def main():
    if len(sys.argv) < 6:
        print_usage()
        return 2

    try:
        fmt = sys.argv[1]
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        ref_path = sys.argv[4]
        dis_path = sys.argv[5]
    except ValueError:
        print_usage()
        return 2

    if width < 0 or height < 0:
        print "width and height must be non-negative, but are {w} and {h}".format(w=width, h=height)
        print_usage()
        return 2

    if fmt not in FMTS:
        print_usage()
        return 2

    out_fmt = get_cmd_option(sys.argv, 6, len(sys.argv), '--out-fmt')
    if not (out_fmt is None
            or out_fmt == 'xml'
            or out_fmt == 'json'
            or out_fmt == 'text'):
        print_usage()
        return 2

    pool_method = get_cmd_option(sys.argv, 6, len(sys.argv), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print '--pool can only have option among {}'.format(', '.join(POOL_METHODS))
        return 2

    asset = Asset(dataset="cmd", content_id=0, asset_id=0,
                  workdir_root=config.ROOT + "/workspace/workdir",
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                  )
    assets = [asset]

    runner_class = PsnrQualityRunner

    runner = runner_class(
        assets, None, fifo_mode=True,
        delete_workdir=True,
        result_store=None,
        optional_dict=None,
        optional_dict2=None,
    )

    # run
    runner.run()
    result = runner.results[0]

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

    # output
    if out_fmt == 'xml':
        print result.to_xml()
    elif out_fmt == 'json':
        print result.to_json()
    else: # None or 'text'
        print str(result)

    return 0

if __name__ == "__main__":
    ret = main()
    exit(ret)