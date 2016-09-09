#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import os

import config
from core.asset import Asset
from core.quality_runner import VmafQualityRunner
from core.quality_runner_adhoc import VmafQualityRunnerWithLocalExplainer
from tools.misc import get_file_name_without_extension

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']
SHOW_LOCAL_EXPLANATION = ['yes', 'no']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) \
          + " fmt width height ref_file dis_file [optional_model_file or none] [show_local_explanation]\n"
    print "fmts:\n\t" + "\n\t".join(FMTS) + "\n"
    print "show_local_explanation:\n\t" + "\n\t".join(SHOW_LOCAL_EXPLANATION) + "\n"

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
        show_local_explanation = sys.argv[7]
        if show_local_explanation == 'yes':
            show_local_explanation = True
        elif show_local_explanation == 'no':
            show_local_explanation = False
        else:
            print_usage()
            exit(2)
    else:
        show_local_explanation = False

    asset = Asset(dataset="run_vmaf",
                  content_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  asset_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  workdir_root=config.ROOT + "/workspace/workdir",
                  ref_path=ref_file,
                  dis_path=dis_file,
                  asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                  )
    assets = [asset]

    if not show_local_explanation:
        runner_class = VmafQualityRunner
    else:
        runner_class = VmafQualityRunnerWithLocalExplainer

    if model_filepath is None:
        optional_dict = None
    else:
        optional_dict = {'model_filepath':model_filepath}

    runner = runner_class(
        assets, None, fifo_mode=True,
        delete_workdir=True,
        result_store=None,
        optional_dict=optional_dict,
        optional_dict2=None,
    )

    # run
    runner.run()
    result = runner.results[0]

    # output
    print str(result)

    # local explanation
    if show_local_explanation:
        import matplotlib.pyplot as plt
        runner.show_local_explanations([result])
        plt.show()

    print 'Done.'

    exit(0)