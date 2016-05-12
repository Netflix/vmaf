#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import os

import config
from core.asset import Asset
from core.quality_runner import VmafQualityRunner
from core.result_store import FileSystemResultStore
from tools.misc import get_file_name_without_extension

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) \
          + " fmt width height ref_file dis_file [optional_model_file]\n"
    print "fmts:\n\t" + "\n\t".join(FMTS) +"\n"

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print_usage()
        exit(2)

    try:
        fmt = sys.argv[1]
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        ref_file = sys.argv[4]
        dis_file = sys.argv[5]
    except ValueError:
        print_usage()
        exit(2)

    if len(sys.argv) >= 7:
        model_filepath = sys.argv[6]
        if model_filepath == 'none':
            model_filepath = None
    else:
        model_filepath = None

    if len(sys.argv) >= 8:
        cache_result = sys.argv[7]
        if cache_result == 'yes':
            result_store = FileSystemResultStore()
        else:
            result_store = None
    else:
        result_store = None

    asset = Asset(dataset="run_vmaf",
                  content_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  asset_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  workdir_root=config.ROOT + "/workspace/workdir",
                  ref_path=ref_file,
                  dis_path=dis_file,
                  asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                  )
    assets = [asset]

    runner_class = VmafQualityRunner

    optional_dict = {
        'model_filepath':model_filepath
    }

    runner = runner_class(
        assets, None, fifo_mode=True,
        delete_workdir=True,
        result_store=result_store,
        optional_dict=optional_dict,
    )

    # run
    runner.run()
    result = runner.results[0]

    # output
    print str(result)

    print 'Done.'

    exit(0)