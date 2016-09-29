#!/usr/bin/env python

import sys
import os

import config
from core.asset import Asset
from core.quality_runner import PsnrQualityRunner
from tools.misc import get_cmd_option

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']
OUT_FMTS = ['json (default)', 'xml', 'text']

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

    # output
    if out_fmt == 'xml':
        print result.to_xml()
    elif out_fmt == 'text':
        print str(result)
    else: # None or 'json'
        print result.to_json()

    return 0

if __name__ == "__main__":
    ret = main()
    exit(ret)