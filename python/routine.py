__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np
from matplotlib import pyplot as plt

from core.cross_validation import ModelCrossValidation
from core.feature_assembler import FeatureAssembler
from core.quality_runner import VmafQualityRunner
from core.result_store import FileSystemResultStore
from tools.misc import indices, get_stdout_logger, import_python_file, \
    close_logger
import config
from core.asset import Asset
from core.executor import run_executors_in_parallel
from core.train_test_model import TrainTestModel

def read_dataset(dataset):

    # asserts, can add more to the list...
    assert hasattr(dataset, 'dataset_name')
    assert hasattr(dataset, 'yuv_fmt')
    assert hasattr(dataset, 'ref_videos')
    assert hasattr(dataset, 'dis_videos')
    assert hasattr(dataset, 'width') or 'width' in dataset.ref_videos[0]
    assert hasattr(dataset, 'height') or 'height' in dataset.ref_videos[0]

    data_set_name = dataset.dataset_name
    yuv_fmt = dataset.yuv_fmt
    ref_videos = dataset.ref_videos
    dis_videos = dataset.dis_videos

    width = dataset.width if hasattr(dataset, 'width') else None
    height = dataset.height if hasattr(dataset, 'height') else None

    ref_dict = {} # dictionary of content_id -> path for ref videos
    for ref_video in ref_videos:
        ref_dict[ref_video['content_id']] = ref_video

    assets = []
    for dis_video in dis_videos:

        if 'dmos' in dis_video:
            groundtruth = dis_video['dmos']
        elif 'mos' in dis_video:
            groundtruth = dis_video['mos']
        else:
            groundtruth = None
            # assert False, 'Each distorted video entry must provide either ' \
            #               'a mos or dmos score.'

        ref_path = ref_dict[dis_video['content_id']]['path']
        width_ = width if width is not None else ref_dict[dis_video['content_id']]['width']
        height_ = height if height is not None else ref_dict[dis_video['content_id']]['height']

        asset_dict = {'width': width_,
                      'height': height_,
                      'yuv_type': yuv_fmt,
                      }
        if groundtruth is not None:
            asset_dict['groundtruth'] = groundtruth

        asset = Asset(dataset=data_set_name,
                      content_id=dis_video['content_id'],
                      asset_id=dis_video['asset_id'],
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_video['path'],
                      asset_dict=asset_dict,
                      )
        assets.append(asset)

    return assets

def test_on_dataset(test_dataset, runner_class, ax,
                    result_store, model_filepath,
                    parallelize=True, fifo_mode=True,
                    aggregate_method=np.mean):

    test_assets = read_dataset(test_dataset)

    optional_dict = {
        'model_filepath':model_filepath
    }

    # construct an quality runner object to assert assets only
    runner = runner_class(test_assets,
                 None,
                 fifo_mode=fifo_mode,
                 delete_workdir=True,
                 result_store=result_store,
                 optional_dict=optional_dict,
                 )
    try:
        # run
        _, results = run_executors_in_parallel(
            runner_class,
            test_assets,
            fifo_mode=fifo_mode,
            delete_workdir=True,
            parallelize=parallelize,
            result_store=result_store,
            optional_dict=optional_dict,
        )

        for result in results:
            result.set_score_aggregate_method(aggregate_method)

        # plot
        groundtruths = map(lambda asset: asset.groundtruth, test_assets)
        predictions = map(lambda result: result[runner_class.get_score_key()], results)
        stats = TrainTestModel.get_stats(groundtruths, predictions)

        print 'Stats on testing data: {}'.format(TrainTestModel.format_stats(stats))

        if ax is not None:
            content_ids = map(lambda asset: asset.content_id, test_assets)
            TrainTestModel.plot_scatter(ax, stats, content_ids)
            ax.set_xlabel('DMOS')
            ax.set_ylabel("Predicted Score")
            ax.grid()
            ax.set_title( "{runner}\n{stats}".format(
                dataset=test_assets[0].dataset,
                runner=runner_class.TYPE,
                stats=TrainTestModel.format_stats(stats),
            ))

    except Exception as e:
        print "Error: " + str(e)

    return test_assets, results

def print_matplotlib_warning():
    print "Warning: cannot import matplotlib, no picture displayed. " \
          "If you are on Mac OS and have installed matplotlib, you " \
          "possibly need to run: \nsudo pip uninstall python-dateutil \n" \
          "sudo pip install python-dateutil==2.2 \n" \
          "Refer to: http://stackoverflow.com/questions/27630114/matplotlib-issue-on-os-x-importerror-cannot-import-name-thread"


def train_test_on_dataset(train_dataset, test_dataset,
                          feature_param, model_param,
                          train_ax, test_ax, result_store,
                          parallelize=True, logger=None, fifo_mode=True,
                          output_model_filepath=None):

    train_assets = read_dataset(train_dataset)
    train_fassembler = FeatureAssembler(
        feature_dict = feature_param.feature_dict,
        feature_option_dict = None,
        assets = train_assets,
        logger=logger,
        fifo_mode=fifo_mode,
        delete_workdir=True,
        result_store=result_store,
        parallelize=parallelize,
    )
    train_fassembler.run()
    train_features = train_fassembler.results

    train_xys = TrainTestModel.get_xys_from_results(train_features)
    train_xs = TrainTestModel.get_xs_from_results(train_features)
    train_ys = TrainTestModel.get_ys_from_results(train_features)

    model_type = model_param.model_type
    model_param_dict = model_param.model_param_dict

    model_class = TrainTestModel.find_subclass(model_type)
    model = model_class(model_param_dict, logger)

    model.train(train_xys)

    # append additional information to model before saving, so that
    # VmafQualityRunner can read and process
    model.append_info('feature_dict', feature_param.feature_dict)
    if 'score_clip' in model_param_dict:
        VmafQualityRunner.set_clip_score(model, model_param_dict['score_clip'])

    train_ys_pred = model.predict(train_xs)

    # apply instructions indicated in the appended info
    train_ys_pred = VmafQualityRunner.clip_score(model, train_ys_pred)

    train_stats = TrainTestModel.get_stats(train_ys['label'], train_ys_pred)

    if logger:
        logger.info('Stats on training data: {}'.format(TrainTestModel.
                                                        format_stats(train_stats)))

    # save model
    if output_model_filepath is not None:
        model.to_file(output_model_filepath)

    if train_ax is not None:
        train_content_ids = map(lambda asset: asset.content_id, train_assets)
        TrainTestModel.plot_scatter(train_ax, train_stats, train_content_ids)
        train_ax.set_xlabel('DMOS')
        train_ax.set_ylabel("Predicted Score")
        train_ax.grid()
        train_ax.set_title( "Dataset: {dataset}, Model: {model}\n{stats}".format(
            dataset=train_dataset.dataset_name,
            model=model.model_id,
            stats=TrainTestModel.format_stats(train_stats)
        ))

    # === test model on test dataset ===

    if test_dataset is None:
        test_assets = None
        test_stats = None
        test_fassembler = None
    else:
        test_assets = read_dataset(test_dataset)
        test_fassembler = FeatureAssembler(
            feature_dict = feature_param.feature_dict,
            feature_option_dict = None,
            assets = test_assets,
            logger=logger,
            fifo_mode=fifo_mode,
            delete_workdir=True,
            result_store=result_store,
            parallelize=True,
        )
        test_fassembler.run()
        test_features = test_fassembler.results

        test_xs = TrainTestModel.get_xs_from_results(test_features)
        test_ys = TrainTestModel.get_ys_from_results(test_features)

        test_ys_pred = model.predict(test_xs)

        # apply instructions indicated in the appended info
        test_ys_pred = VmafQualityRunner.clip_score(model, test_ys_pred)

        test_stats = TrainTestModel.get_stats(test_ys['label'], test_ys_pred)

        if logger:
            logger.info('Stats on testing data: {}'.format(
                TrainTestModel.format_stats(test_stats)))

        if test_ax is not None:
            test_content_ids = map(lambda asset: asset.content_id, test_assets)
            TrainTestModel.plot_scatter(test_ax, test_stats, test_content_ids)
            test_ax.set_xlabel('DMOS')
            test_ax.set_ylabel("Predicted Score")
            test_ax.grid()
            test_ax.set_title( "Dataset: {dataset}, Model: {model}\n{stats}".format(
                dataset=test_dataset.dataset_name,
                model=model.model_id,
                stats=TrainTestModel.format_stats(test_stats)
            ))

    return train_fassembler, train_assets, train_stats, \
           test_fassembler, test_assets, test_stats


def construct_kfold_list(assets, contentid_groups):
    # construct cross validation kfold input list
    content_ids = map(lambda asset: asset.content_id, assets)
    kfold = []
    for curr_content_group in contentid_groups:
        curr_indices = indices(content_ids, lambda x: x in curr_content_group)
        kfold.append(curr_indices)
    return kfold


def cv_on_dataset(dataset, feature_param, model_param, ax, result_store,
                  contentid_groups, logger=None, aggregate_method=np.mean):

    assets = read_dataset(dataset)
    kfold = construct_kfold_list(assets, contentid_groups)

    fassembler = FeatureAssembler(
        feature_dict = feature_param.feature_dict,
        feature_option_dict = None,
        assets = assets,
        logger=logger,
        delete_workdir=True,
        result_store=result_store,
        parallelize=True, fifo_mode=True,
        # parallelize=False, fifo_mode=False, # VQM
    )
    fassembler.run()
    results = fassembler.results

    for result in results:
        result.set_score_aggregate_method(aggregate_method)

    # run nested kfold cv for each combintation
    cv_output = ModelCrossValidation.run_kfold_cross_validation(
        TrainTestModel.find_subclass(model_param.model_type),
        model_param.model_param_dict,
        results,
        kfold,
        logger=logger,
    )

    print 'Feature parameters: {}'.format(feature_param.feature_dict)
    print 'Model type: {}'.format(model_param.model_type)
    print 'Model parameters: {}'.format(model_param.model_param_dict)
    print 'Stats: {}'.format(TrainTestModel.format_stats(cv_output['aggr_stats']))

    if ax is not None:
        TrainTestModel.plot_scatter(ax, cv_output['aggr_stats'], cv_output['contentids'])
        ax.set_xlabel('DMOS')
        ax.set_ylabel("Predicted Score")
        ax.grid()
        ax.set_title( "Dataset: {dataset}, Model: {model},\n{stats}".format(
            dataset=dataset.dataset_name,
            model=model_param.model_type,
            stats=TrainTestModel.format_stats(cv_output['aggr_stats'])
        ))

    return assets, cv_output


def run_remove_results_for_dataset(result_store, dataset, executor_class):
    assets = read_dataset(dataset)
    executor = executor_class(assets=assets,
                                logger=None,
                                result_store=result_store)
    executor.remove_results()


def run_vmaf_cv(train_dataset_filepath,
                test_dataset_filepath,
                param_filepath,
                output_model_filepath=None):

    logger = get_stdout_logger()
    result_store = FileSystemResultStore()

    train_dataset = import_python_file(train_dataset_filepath)
    test_dataset = import_python_file(test_dataset_filepath)

    param = import_python_file(param_filepath)

    # === plot scatter ===

    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(figsize=(5*ncols, 5*nrows), nrows=nrows, ncols=ncols)

    train_test_on_dataset(train_dataset, test_dataset, param, param, axs[0], axs[1],
                          result_store, parallelize=True, logger=None,
                          output_model_filepath=output_model_filepath)

    # axs[0].set_xlim([0, 120])
    # axs[0].set_ylim([0, 120])

    # axs[1].set_xlim([0, 120])
    # axs[1].set_ylim([0, 120])

    bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
    axs[0].annotate('Training Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)
    axs[1].annotate('Testing Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

    plt.tight_layout()

    # === clean up ===
    close_logger(logger)


def run_vmaf_kfold_cv(dataset_filepath,
                      contentid_groups,
                      param_filepath,
                      aggregate_method,
                      ):

    logger = get_stdout_logger()
    result_store = FileSystemResultStore()
    dataset = import_python_file(dataset_filepath)
    param = import_python_file(param_filepath)

    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)

    cv_on_dataset(dataset, param, param, ax, result_store, contentid_groups,
                  logger, aggregate_method)

    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    plt.tight_layout()

    # === clean up ===
    close_logger(logger)