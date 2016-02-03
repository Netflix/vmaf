__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
import os
from asset import Asset
from vmaf_quality_runner import VmafQualityRunner

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print "usage: " + os.path.basename(sys.argv[0]) \
              + " [fmt] [width] [height] [ref_file] [dis_file]"
        print "fmts:\n\tyuv420\n\tyuv422\n\tyuv444\n"
        exit(0)

    fmt = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    ref_file = sys.argv[4]
    dis_file = sys.argv[5]

    asset = Asset(dataset="single", content_id=0,
                  workdir_root=config.ROOT + "/workspace/workdir",
                  ref_path=ref_file,
                  dis_path=dis_file,
                  asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                  )
    runner = VmafQualityRunner(
        [asset], None, fifo_mode=True,
        log_file_dir=config.ROOT + "/workspace/log_file_dir")

    # force run
    runner.remove_logs()

    runner.run()

    print map(lambda result:str(result), runner.results)

    # clean up
    runner.remove_logs()