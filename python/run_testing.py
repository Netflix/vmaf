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
import matplotlib.pylab as plt
from tools import get_dir_without_last_slash, get_file_name_without_extension, \
    import_python_file
from quality_runner import QualityRunner, VmaftQualityRunner, PsnrQualityRunner, \
    VmafQualityRunner


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

    TrainTestModel.plot_scatter(ax, stats, content_ids)
    ax.set_xlabel('DMOS')
    ax.set_ylabel("Predicted Score")
    ax.grid()
    ax.set_title( "Dataset: {dataset}, Runner: {runner}\n{stats}".format(
        dataset=assets[0].dataset,
        runner=results[0].executor_id,
        stats=TrainTestModel.format_stats(stats)
    ))

def test_on_dataset(dataset, quality_runner_class, ax, result_store, model_filepath):

    assets = read_dataset(dataset)

    optional_dict = {
        'model_filepath':model_filepath
    }

    runner = quality_runner_class(assets,
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
            assets,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=result_store,
            optional_dict=optional_dict,
        )

        # plot
        plot_scatter(ax, assets, results, quality_runner_class)

    except Exception as e:
        print "Error: " + str(e)


def print_usage():
    # quality_runner_types = map(lambda runner: runner.TYPE, QualityRunner.get_subclasses())
    quality_runner_types = map(lambda runner: runner.TYPE,
                               [PsnrQualityRunner,
                                VmafQualityRunner,
                                VmaftQualityRunner,
                                ])
    cache_result = ['yes', 'no']
    print "usage: " + os.path.basename(sys.argv[0]) + \
          " quality_type cache_result dataset_file [model_file]\n"
    print "quality_types:\n\t" + "\n\t".join(quality_runner_types) +"\n"
    print "cache_result:\n\t" + "\n\t".join(cache_result) +"\n"
    print "dataset_file:\n\t" + 'dataset to run test on' +"\n"
    print "model_file:\n\t" + 'custom model for VMAFT only' +"\n"

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print_usage()
        exit(2)

    try:
        quality_type = sys.argv[1]
        cache_result = sys.argv[2]
        dataset_filepath = sys.argv[3]
    except ValueError:
        print_usage()
        exit(2)

    if len(sys.argv) >= 5:
        model_filepath = sys.argv[4]
    else:
        model_filepath = None

    if model_filepath is not None and quality_type != VmaftQualityRunner.TYPE:
        print_usage()
        exit(2)

    sys.path.append(config.ROOT + '/python/private/script')

    try:
        dataset = import_python_file(dataset_filepath)
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

    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    test_on_dataset(dataset, runner_class, ax, result_store, model_filepath)
    plt.tight_layout()
    plt.show()

    print 'Done.'

    exit(0)