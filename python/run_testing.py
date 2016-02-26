#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys
import config
from asset import Asset
from executor import run_executors_in_parallel
from result import FileSystemResultStore
from train_test_model import TrainTestModel
from tools import import_python_file
from quality_runner import QualityRunner, VmafQualityRunner

def read_dataset(dataset):

    data_set_name = dataset.dataset_name
    yuv_fmt = dataset.yuv_fmt
    width = dataset.width
    height = dataset.height
    ref_videos = dataset.ref_videos
    dis_videos = dataset.dis_videos

    ref_path_dict = {} # dictionary of content_id -> path for ref videos
    for ref_video in ref_videos:
        ref_path_dict[ref_video['content_id']] = ref_video['path']

    assets = []
    for dis_video in dis_videos:
        asset = Asset(dataset=data_set_name,
                      content_id=dis_video['content_id'],
                      asset_id=dis_video['asset_id'],
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path_dict[dis_video['content_id']],
                      dis_path=dis_video['path'],
                      asset_dict={'width':width,
                                  'height':height,
                                  'yuv_type':yuv_fmt,
                                  'groundtruth':dis_video['dmos']
                                  }
                      )
        assets.append(asset)

    return assets

def plot_scatter(ax, assets, results, runner_class):

    assert len(assets) == len(results)

    groundtruths = map(lambda asset: asset.groundtruth, assets)
    predictions = map(lambda result: result[runner_class.get_score_key()], results)
    content_ids = map(lambda asset: asset.content_id, assets)
    stats = TrainTestModel.get_stats(groundtruths, predictions)

    if ax is not None:
        TrainTestModel.plot_scatter(ax, stats, content_ids)
        ax.set_xlabel('DMOS')
        ax.set_ylabel("Predicted Score")
        ax.grid()
        ax.set_title( "Dataset: {dataset}, Runner: {runner}\n{stats}".format(
            dataset=assets[0].dataset,
            runner=results[0].executor_id,
            stats=TrainTestModel.format_stats(stats)
        ))

        bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
        ax.annotate('Testing Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

def test_on_dataset(test_dataset, quality_runner_class, ax,
                    result_store, model_filepath, parallelize=True):

    test_assets = read_dataset(test_dataset)

    optional_dict = {
        'model_filepath':model_filepath
    }

    # construct an quality runner object to assert assets only
    runner = quality_runner_class(test_assets,
                 None,
                 log_file_dir=config.ROOT + "/workspace/log_file_dir",
                 fifo_mode=True,
                 delete_workdir=True,
                 result_store=result_store,
                 optional_dict=optional_dict,
                 )
    try:
        # run
        _, results = run_executors_in_parallel(
            quality_runner_class,
            test_assets,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=parallelize,
            result_store=result_store,
            optional_dict=optional_dict,
        )

        # plot
        plot_scatter(ax, test_assets, results, quality_runner_class)

    except Exception as e:
        print "Error: " + str(e)


def print_usage():
    # quality_runner_types = map(lambda runner: runner.TYPE, QualityRunner.get_subclasses())
    quality_runner_types = ['PSNR', 'VMAF']
    cache_result = ['yes', 'no']
    parallelize = ['yes', 'no']
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " quality_type cache_result parallelize test_dataset_file [optional_VMAF_model_file]\n"
    print "quality_types:\n\t" + "\n\t".join(quality_runner_types) +"\n"
    print "cache_result:\n\t" + "\n\t".join(cache_result) +"\n"
    print "parallelize:\n\t" + "\n\t".join(parallelize) +"\n"

if __name__ == '__main__':

    if len(sys.argv) < 5:
        print_usage()
        exit(2)

    try:
        quality_type = sys.argv[1]
        cache_result = sys.argv[2]
        do_parallelize = sys.argv[3]
        test_dataset_filepath = sys.argv[4]
    except ValueError:
        print_usage()
        exit(2)

    if do_parallelize == 'yes':
        parallelize = True
    elif do_parallelize == 'no':
        parallelize = False
    else:
        print_usage()
        exit(2)

    if len(sys.argv) >= 6:
        model_filepath = sys.argv[5]
    else:
        model_filepath = None

    if model_filepath is not None and quality_type != VmafQualityRunner.TYPE:
        print "Input error: only quality_type VMAF accepts optional_VMAF_model_file."
        print_usage()
        exit(2)

    try:
        test_dataset = import_python_file(test_dataset_filepath)
    except Exception as e:
        print "Error: " + str(e)
        exit(1)

    try:
        runner_class = QualityRunner.find_subclass(quality_type)
    except:
        print_usage()
        exit(2)

    if cache_result == 'yes':
        result_store = FileSystemResultStore()
    elif cache_result == 'no':
        result_store = None
    else:
        print_usage()
        exit(2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
        test_on_dataset(test_dataset, runner_class, ax,
                        result_store, model_filepath, parallelize)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print "Warning: cannot import matplotlib, no picture displayed. " \
              "Install by: \npip install matplotlib"
        test_on_dataset(test_dataset, runner_class, None,
                        result_store, model_filepath, parallelize)

    print 'Done.'

    exit(0)