#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import os

import config
from core.asset import Asset
from core.quality_runner import VmafQualityRunner
from core.quality_runner_adhoc import VmafQualityRunnerWithLocalExplainer
from tools.misc import get_file_name_without_extension, get_cmd_option, \
    cmd_option_exists

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']
OUT_FMTS = ['json (default)', 'xml', 'text']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) \
          + " fmt width height ref_path dis_path [--model model_path] [--out-fmt out_fmt] [--local-explain]\n"
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
        ref_file = sys.argv[4]
        dis_file = sys.argv[5]
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

    model_path = get_cmd_option(sys.argv, 6, len(sys.argv), '--model')

    out_fmt = get_cmd_option(sys.argv, 6, len(sys.argv), '--out-fmt')
    if not (out_fmt is None
            or out_fmt == 'xml'
            or out_fmt == 'json'
            or out_fmt == 'text'):
        print_usage()
        return 2

    show_local_explanation = cmd_option_exists(sys.argv, 6, len(sys.argv), '--local-explain')

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

    if model_path is None:
        optional_dict = None
    else:
        optional_dict = {'model_filepath':model_path}

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
    if out_fmt == 'xml':
        print result.to_xml()
    elif out_fmt == 'text':
        print str(result)
    else: # None or 'json'
        print result.to_json()

    # local explanation
    if show_local_explanation:
        import matplotlib.pyplot as plt
        runner.show_local_explanations([result])
        plt.show()

    return 0

if __name__ == "__main__":
    ret = main()
    exit(ret)