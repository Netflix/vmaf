#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
import os
import pprint
from asset import Asset
from vmaf_quality_runner import VmafQualityRunner

FMTS = ['yuv420p', 'yuv422p', 'yuv444p']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) \
          + " [fmt] [width] [height] [ref_file] [dis_file]\n"
    print "fmts:\n\t" + "\n\t".join(FMTS) +"\n"

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print_usage()
        exit(0)

    fmt = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    ref_file = sys.argv[4]
    dis_file = sys.argv[5]

    asset = Asset(dataset="cmd", content_id=0, asset_id=0,
                  workdir_root=config.ROOT + "/workspace/workdir",
                  ref_path=ref_file,
                  dis_path=dis_file,
                  asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                  )
    assets = [asset]

    runner = VmafQualityRunner(
        assets, None, fifo_mode=True,
        log_file_dir=config.ROOT + "/workspace/log_file_dir")

    # run
    runner.run()
    result = runner.results[0]

    # output
    print 'Input:'
    print asset.__dict__
    print ''
    print 'Output:'
    print str(result)
    print ''

    # clean up
    runner.remove_logs()

    print 'Done.'