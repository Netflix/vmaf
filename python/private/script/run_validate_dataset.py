__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
from asset import Asset
from executor import run_executors_in_parallel
from result import FileSystemResultStore
from train_test_model import TrainTestModel

def read_dataset(dataset, train_or_test):

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
        if train_or_test == 'all' or dis_video['train_or_test'] == train_or_test:
            assets.append(asset)

    return assets

def plot_scatter(assets, results, runner_class):

    assert len(assets) == len(results)

    groundtruths = map(lambda asset: asset.groundtruth, assets)
    predictions = map(lambda result: result[runner_class.get_score_key()], results)
    content_ids = map(lambda asset: asset.content_id, assets)
    stats = TrainTestModel.get_stats(groundtruths, predictions)
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    TrainTestModel.plot_scatter(ax, stats, content_ids)
    plt.xlabel('Groundtruth (DMOS)')
    plt.ylabel("Prediction")
    plt.grid()
    plt.title("Dataset: {dataset}, Runner: {runner}\n{stats}".format(
        dataset=assets[0].dataset,
        runner=results[0].executor_id,
        stats=TrainTestModel.format_stats(
            TrainTestModel.get_stats(groundtruths, predictions))
    ))
    plt.show()


def validate_dataset(dataset, quality_runner_class, train_or_test='all'):

    assets = read_dataset(dataset, train_or_test)
    result_store = FileSystemResultStore()

    # construct an VmafQualityRunner object only to assert assets, and to remove
    runner = runner_class(assets,
                 None,
                 log_file_dir=config.ROOT + "/workspace/log_file_dir",
                 fifo_mode=True,
                 delete_workdir=True,
                 result_store=result_store,
                 )

    try:
        # run
        runners, results = run_executors_in_parallel(
            quality_runner_class,
            assets,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=result_store,
        )

        # plot
        plot_scatter(assets, results, runner_class)

    finally:
        # runner.remove_logs()
        # runner.remove_results()
        pass



if __name__ == '__main__':

    sys.path.append(config.ROOT + '/python/private/script')

    # import example_dataset as dataset
    import NFLX_dataset as dataset

    from quality_runner import VmafQualityRunner as runner_class

    # validate_dataset(dataset, runner_class)
    # validate_dataset(dataset, runner_class, train_or_test='train')
    validate_dataset(dataset, runner_class, train_or_test='test')

    print 'Done.'