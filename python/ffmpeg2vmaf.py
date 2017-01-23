#!/usr/bin/env python

import sys
import os

import numpy as np

import config
from core.asset import Asset
from core.quality_runner import VmafQualityRunner
from core.quality_runner_extra import VmafQualityRunnerWithLocalExplainer
from tools.misc import get_file_name_without_extension, get_cmd_option, \
    cmd_option_exists
from tools.stats import ListStats

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

OUT_FMTS = ['text (default)', 'xml', 'json']
POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) \
          + " quality_width quality_height ref_path dis_path [--model model_path] [--out-fmt out_fmt] [--work-dir work_dir]\n"
    print "out_fmt:\n\t" + "\n\t".join(OUT_FMTS) + "\n"

def main():
    if len(sys.argv) < 5:
        print_usage()
        return 2

    try:
        q_width = int(sys.argv[1])
        q_height = int(sys.argv[2])
        ref_file = sys.argv[3]
        dis_file = sys.argv[4]
    except ValueError:
        print_usage()
        return 2

    if q_width < 0 or q_height < 0:
        print "quality_width and quality_height must be non-negative, but are {w} and {h}".format(w=q_width, h=q_height)
        print_usage()
        return 2

    model_path = get_cmd_option(sys.argv, 5, len(sys.argv), '--model')

    out_fmt = get_cmd_option(sys.argv, 5, len(sys.argv), '--out-fmt')
    if not (out_fmt is None
            or out_fmt == 'xml'
            or out_fmt == 'json'
            or out_fmt == 'text'):
        print_usage()
        return 2

    work_dir = get_cmd_option(sys.argv, 5, len(sys.argv), '--work-dir')

    pool_method = get_cmd_option(sys.argv, 5, len(sys.argv), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print '--pool can only have option among {}'.format(', '.join(POOL_METHODS))
        return 2

    show_local_explanation = cmd_option_exists(sys.argv, 5, len(sys.argv), '--local-explain')

    if work_dir is None:
        work_dir = config.ROOT + "/workspace/workdir"

    asset = Asset(dataset="cmd",
                  content_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  asset_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  workdir_root=work_dir,
                  ref_path=ref_file,
                  dis_path=dis_file,
                  asset_dict={'quality_width':q_width, 'quality_height':q_height, 'yuv_type': 'notyuv'}
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

    # local explanation
    if show_local_explanation:
        import matplotlib.pyplot as plt
        runner.show_local_explanations([result])
        plt.show()

    return 0

if __name__ == "__main__":
    ret = main()
    exit(ret)