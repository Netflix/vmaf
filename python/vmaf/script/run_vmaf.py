#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

import sys
import os

import numpy as np

from vmaf.config import VmafConfig, DisplayConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafQualityRunner
from vmaf.tools.misc import get_file_name_without_extension, get_cmd_option, \
    cmd_option_exists
from vmaf.tools.stats import ListStats

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 
        'yuv420p10le', 'yuv422p10le', 'yuv444p10le',
        'yuv420p12le', 'yuv422p12le', 'yuv444p12le',
        'yuv420p16le', 'yuv422p16le', 'yuv444p16le',
        ]
OUT_FMTS = ['text (default)', 'xml', 'json']
POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']


def print_usage():
    print("usage: " + os.path.basename(sys.argv[0]) \
          + " fmt width height ref_path dis_path [--model model_path] [--out-fmt out_fmt] " \
            "[--phone-model] [--ci] [--save-plot plot_dir]\n")
    print("fmt:\n\t" + "\n\t".join(FMTS) + "\n")
    print("out_fmt:\n\t" + "\n\t".join(OUT_FMTS) + "\n")


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
        print("width and height must be non-negative, but are {w} and {h}".format(w=width, h=height))
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

    pool_method = get_cmd_option(sys.argv, 6, len(sys.argv), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print('--pool can only have option among {}'.format(', '.join(POOL_METHODS)))
        return 2

    show_local_explanation = cmd_option_exists(sys.argv, 6, len(sys.argv), '--local-explain')

    phone_model = cmd_option_exists(sys.argv, 6, len(sys.argv), '--phone-model')

    enable_conf_interval = cmd_option_exists(sys.argv, 6, len(sys.argv), '--ci')

    save_plot_dir = get_cmd_option(sys.argv, 6, len(sys.argv), '--save-plot')

    if show_local_explanation and enable_conf_interval:
        print('cannot set both --local-explain and --ci flags')
        return 2

    asset = Asset(dataset="cmd",
                  content_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  asset_id=abs(hash(get_file_name_without_extension(ref_file))) % (10 ** 16),
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_file,
                  dis_path=dis_file,
                  asset_dict={'width':width, 'height':height, 'yuv_type':fmt}
                  )
    assets = [asset]

    if show_local_explanation:
        from vmaf.core.quality_runner_extra import VmafQualityRunnerWithLocalExplainer
        runner_class = VmafQualityRunnerWithLocalExplainer
    elif enable_conf_interval:
        from vmaf.core.quality_runner import BootstrapVmafQualityRunner
        runner_class = BootstrapVmafQualityRunner
    else:
        runner_class = VmafQualityRunner

    if model_path is None:
        optional_dict = None
    else:
        optional_dict = {'model_filepath':model_path}

    if phone_model:
        if optional_dict is None:
            optional_dict = {}
        optional_dict['enable_transform_score'] = True

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
        print(result.to_xml())
    elif out_fmt == 'json':
        print(result.to_json())
    else: # None or 'text'
        print(str(result))

    # local explanation
    if show_local_explanation:
        runner.show_local_explanations([result])

        if save_plot_dir is None:
            DisplayConfig.show()
        else:
            DisplayConfig.show(write_to_dir=save_plot_dir)

    return 0


if __name__ == "__main__":
    ret = main()
    exit(ret)
