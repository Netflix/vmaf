import numpy as np
from matplotlib import pyplot as plt

from core.cross_validation import ModelCrossValidation
from core.feature_assembler import FeatureAssembler
from core.quality_runner import VmafQualityRunner
from core.result_store import FileSystemResultStore
from mos.dataset_reader import RawDatasetReader
from mos.subjective_model import DmosModel
from tools.misc import indices, get_stdout_logger, import_python_file, \
    close_logger, get_file_name_without_extension
import config
from core.asset import Asset
from core.executor import run_executors_in_parallel
from core.train_test_model import TrainTestModel, RegressorMixin, \
    ClassifierMixin
from core.local_explainer import LocalExplainer

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

def read_dataset(dataset, **kwargs):

    groundtruth_key = kwargs['groundtruth_key'] if 'groundtruth_key' in kwargs else None
    skip_asset_with_none_groundtruth = kwargs['skip_asset_with_none_groundtruth'] \
        if 'skip_asset_with_none_groundtruth' in kwargs else False

    # asserts, can add more to the list...
    assert hasattr(dataset, 'dataset_name')
    assert hasattr(dataset, 'ref_videos')
    assert hasattr(dataset, 'dis_videos')

    assert hasattr(dataset, 'width') or all(['width' in ref_video for ref_video in dataset.ref_videos])
    assert hasattr(dataset, 'height') or all(['height' in ref_video for ref_video in dataset.ref_videos])
    assert hasattr(dataset, 'yuv_fmt') or all(['yuv_fmt' in ref_video for ref_video in dataset.ref_videos])

    data_set_name = dataset.dataset_name
    ref_videos = dataset.ref_videos
    dis_videos = dataset.dis_videos

    width = dataset.width if hasattr(dataset, 'width') else None
    height = dataset.height if hasattr(dataset, 'height') else None
    yuv_fmt = dataset.yuv_fmt if hasattr(dataset, 'yuv_fmt') else None

    quality_width = dataset.quality_width if hasattr(dataset, 'quality_width') else None
    quality_height = dataset.quality_height if hasattr(dataset, 'quality_height') else None
    resampling_type = dataset.resampling_type if hasattr(dataset, 'resampling_type') else None
    crop_cmd = dataset.crop_cmd if hasattr(dataset, 'crop_cmd') else None
    pad_cmd = dataset.pad_cmd if hasattr(dataset, 'pad_cmd') else None

    ref_dict = {} # dictionary of content_id -> path for ref videos
    for ref_video in ref_videos:
        ref_dict[ref_video['content_id']] = ref_video

    assets = []
    for dis_video in dis_videos:

        if groundtruth_key is not None:
            groundtruth = dis_video[groundtruth_key]
        else:
            if 'dmos' in dis_video:
                groundtruth = dis_video['dmos']
            elif 'mos' in dis_video:
                groundtruth = dis_video['mos']
            elif 'groundtruth' in dis_video:
                groundtruth = dis_video['groundtruth']
            else:
                groundtruth = None

        if 'os' in dis_video:
            raw_groundtruth = dis_video['os']
        else:
            raw_groundtruth = None

        ref_path = ref_dict[dis_video['content_id']]['path']
        width_ = width if width is not None else ref_dict[dis_video['content_id']]['width']
        height_ = height if height is not None else ref_dict[dis_video['content_id']]['height']
        yuv_fmt_ = yuv_fmt if yuv_fmt is not None else ref_dict[dis_video['content_id']]['yuv_fmt']

        if quality_width is not None:
            quality_width_ = quality_width
        elif 'quality_width' in dis_video:
            quality_width_ = dis_video['quality_width']
        else:
            quality_width_ = None

        if quality_height is not None:
            quality_height_ = quality_height
        elif 'quality_height' in dis_video:
            quality_height_ = dis_video['quality_height']
        else:
            quality_height_ = None

        if resampling_type is not None:
            resampling_type_ = resampling_type
        elif 'resampling_type' in dis_video:
            resampling_type_ = dis_video['resampling_type']
        else:
            resampling_type_ = None

        if crop_cmd is not None:
            crop_cmd_ = crop_cmd
        elif 'crop_cmd' in dis_video:
            crop_cmd_ = dis_video['crop_cmd']
        else:
            crop_cmd_ = None

        if pad_cmd is not None:
            pad_cmd_ = pad_cmd
        elif 'pad_cmd' in dis_video:
            pad_cmd_ = dis_video['pad_cmd']
        else:
            pad_cmd_ = None

        asset_dict = {'width': width_,
                      'height': height_,
                      'yuv_type': yuv_fmt_,
                      }
        if groundtruth is not None:
            asset_dict['groundtruth'] = groundtruth
        if raw_groundtruth is not None:
            asset_dict['raw_groundtruth'] = raw_groundtruth
        if quality_width_ is not None:
            asset_dict['quality_width'] = quality_width_
        if quality_height_ is not None:
            asset_dict['quality_height'] = quality_height_
        if resampling_type_ is not None:
            asset_dict['resampling_type'] = resampling_type_
        if crop_cmd_ is not None:
            asset_dict['crop_cmd'] = crop_cmd_
        if pad_cmd_ is not None:
            asset_dict['pad_cmd'] = pad_cmd_

        if groundtruth is None and skip_asset_with_none_groundtruth:
            pass
        else:
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
                    aggregate_method=np.mean,
                    type = 'regressor',
                    **kwargs):

    if type == 'regressor':
        model_type = RegressorMixin
    elif type == 'classifier':
        model_type = ClassifierMixin
    else:
        assert False

    test_assets = read_dataset(test_dataset, **kwargs)
    test_raw_assets = None
    try:
        for test_asset in test_assets:
            assert test_asset.groundtruth is not None
    except AssertionError:
        # no groundtruth, try do subjective modeling
        subj_model_class = kwargs['subj_model_class'] if 'subj_model_class' in kwargs and kwargs['subj_model_class'] is not None else DmosModel
        subjective_model = subj_model_class(RawDatasetReader(test_dataset))
        subjective_model.run_modeling(**kwargs)
        test_dataset_aggregate = subjective_model.to_aggregated_dataset(**kwargs)
        test_raw_assets = test_assets
        test_assets = read_dataset(test_dataset_aggregate, **kwargs)

    if model_filepath is not None:
        optional_dict = {'model_filepath': model_filepath}
    else:
        optional_dict = None

    # run
    runner = runner_class(
        test_assets,
        None, fifo_mode=fifo_mode,
        delete_workdir=True,
        result_store=result_store,
        optional_dict=optional_dict,
        optional_dict2=None,
    )
    runner.run(parallelize=parallelize)
    results = runner.results

    for result in results:
        result.set_score_aggregate_method(aggregate_method)

    # plot
    groundtruths = map(lambda asset: asset.groundtruth, test_assets)
    predictions = map(lambda result: result[runner_class.get_score_key()], results)
    raw_grountruths = None if test_raw_assets is None else \
        map(lambda asset: asset.raw_groundtruth, test_raw_assets)
    stats = model_type.get_stats(groundtruths, predictions, ys_label_raw=raw_grountruths)

    print 'Stats on testing data: {}'.format(model_type.format_stats(stats))

    if ax is not None:
        content_ids = map(lambda asset: asset.content_id, test_assets)
        model_type.plot_scatter(ax, stats, content_ids)
        ax.set_xlabel('True Score')
        ax.set_ylabel("Predicted Score")
        ax.grid()
        ax.set_title( "{runner}\n{stats}".format(
            dataset=test_assets[0].dataset,
            runner=runner_class.TYPE,
            stats=model_type.format_stats(stats),
        ))

    return test_assets, results

def print_matplotlib_warning():
    print "Warning: cannot import matplotlib, no picture displayed. " \
          "If you are on Mac OS and have installed matplotlib, you " \
          "possibly need to run: \nsudo pip uninstall python-dateutil \n" \
          "sudo pip install python-dateutil==2.2 \n" \
          "Refer to: http://stackoverflow.com/questions/27630114/matplotlib-issue-on-os-x-importerror-cannot-import-name-thread"


def train_test_vmaf_on_dataset(train_dataset, test_dataset,
                               feature_param, model_param,
                               train_ax, test_ax, result_store,
                               parallelize=True, logger=None, fifo_mode=True,
                               output_model_filepath=None,
                               aggregate_method=np.mean,
                               **kwargs):

    train_assets = read_dataset(train_dataset, **kwargs)
    train_raw_assets = None
    try:
        for train_asset in train_assets:
            assert train_asset.groundtruth is not None
    except AssertionError:
        # no groundtruth, try do subjective modeling
        subj_model_class = kwargs['subj_model_class'] if 'subj_model_class' in kwargs and kwargs['subj_model_class'] is not None else DmosModel
        subjective_model = subj_model_class(RawDatasetReader(train_dataset))
        subjective_model.run_modeling(**kwargs)
        train_dataset_aggregate = subjective_model.to_aggregated_dataset(**kwargs)
        train_raw_assets = train_assets
        train_assets = read_dataset(train_dataset_aggregate, **kwargs)

    train_fassembler = FeatureAssembler(
        feature_dict = feature_param.feature_dict,
        feature_option_dict = None,
        assets = train_assets,
        logger=logger,
        fifo_mode=fifo_mode,
        delete_workdir=True,
        result_store=result_store,
        optional_dict=None,
        optional_dict2=None,
        parallelize=parallelize,
    )
    train_fassembler.run()
    train_features = train_fassembler.results

    for result in train_features:
        result.set_score_aggregate_method(aggregate_method)

    model_type = model_param.model_type
    model_param_dict = model_param.model_param_dict

    model_class = TrainTestModel.find_subclass(model_type)

    train_xys = model_class.get_xys_from_results(train_features)
    train_xs = model_class.get_xs_from_results(train_features)
    train_ys = model_class.get_ys_from_results(train_features)

    model = model_class(model_param_dict, logger)

    model.train(train_xys)

    # append additional information to model before saving, so that
    # VmafQualityRunner can read and process
    model.append_info('feature_dict', feature_param.feature_dict)
    if 'score_clip' in model_param_dict:
        VmafQualityRunner.set_clip_score(model, model_param_dict['score_clip'])
    if 'score_transform' in model_param_dict:
        VmafQualityRunner.set_transform_score(model, model_param_dict['score_transform'])

    train_ys_pred = VmafQualityRunner.predict_with_model(model, train_xs, **kwargs)

    raw_groundtruths = None if train_raw_assets is None else \
        map(lambda asset: asset.raw_groundtruth, train_raw_assets)

    train_stats = model.get_stats(train_ys['label'], train_ys_pred,
                                  ys_label_raw=raw_groundtruths)

    log = 'Stats on training data: {}'.format(model.format_stats(train_stats))
    if logger:
        logger.info(log)
    else:
        print log

    # save model
    if output_model_filepath is not None:
        model.to_file(output_model_filepath)

    if train_ax is not None:
        train_content_ids = map(lambda asset: asset.content_id, train_assets)
        model_class.plot_scatter(train_ax, train_stats, train_content_ids)
        train_ax.set_xlabel('True Score')
        train_ax.set_ylabel("Predicted Score")
        train_ax.grid()
        train_ax.set_title( "Dataset: {dataset}, Model: {model}\n{stats}".format(
            dataset=train_dataset.dataset_name,
            model=model.model_id,
            stats=model_class.format_stats(train_stats)
        ))

    # === test model on test dataset ===

    if test_dataset is None:
        test_assets = None
        test_stats = None
        test_fassembler = None
    else:
        test_assets = read_dataset(test_dataset, **kwargs)
        test_raw_assets = None
        try:
            for test_asset in test_assets:
                assert test_asset.groundtruth is not None
        except AssertionError:
            # no groundtruth, try do subjective modeling
            subj_model_class = kwargs['subj_model_class'] if 'subj_model_class' in kwargs and kwargs['subj_model_class'] is not None else DmosModel
            subjective_model = subj_model_class(RawDatasetReader(test_dataset))
            subjective_model.run_modeling(**kwargs)
            test_dataset_aggregate = subjective_model.to_aggregated_dataset(**kwargs)
            test_raw_assets = test_assets
            test_assets = read_dataset(test_dataset_aggregate, **kwargs)

        test_fassembler = FeatureAssembler(
            feature_dict = feature_param.feature_dict,
            feature_option_dict = None,
            assets = test_assets,
            logger=logger,
            fifo_mode=fifo_mode,
            delete_workdir=True,
            result_store=result_store,
            optional_dict=None,
            optional_dict2=None,
            parallelize=True,
        )
        test_fassembler.run()
        test_features = test_fassembler.results

        for result in test_features:
            result.set_score_aggregate_method(aggregate_method)

        test_xs = model_class.get_xs_from_results(test_features)
        test_ys = model_class.get_ys_from_results(test_features)

        test_ys_pred = VmafQualityRunner.predict_with_model(model, test_xs, **kwargs)

        raw_groundtruths = None if test_raw_assets is None else \
            map(lambda asset: asset.raw_groundtruth, test_raw_assets)

        test_stats = model_class.get_stats(test_ys['label'], test_ys_pred,
                                           ys_label_raw=raw_groundtruths)

        log = 'Stats on testing data: {}'.format(model_class.format_stats(test_stats))
        if logger:
            logger.info(log)
        else:
            print log

        if test_ax is not None:
            test_content_ids = map(lambda asset: asset.content_id, test_assets)
            model_class.plot_scatter(test_ax, test_stats, test_content_ids)
            test_ax.set_xlabel('True Score')
            test_ax.set_ylabel("Predicted Score")
            test_ax.grid()
            test_ax.set_title( "Dataset: {dataset}, Model: {model}\n{stats}".format(
                dataset=test_dataset.dataset_name,
                model=model.model_id,
                stats=model_class.format_stats(test_stats)
            ))

    return train_fassembler, train_assets, train_stats, \
           test_fassembler, test_assets, test_stats, model


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
        optional_dict=None,
        optional_dict2=None,
        parallelize=True, fifo_mode=True,
        # parallelize=False, fifo_mode=False, # VQM
    )
    fassembler.run()
    results = fassembler.results

    for result in results:
        result.set_score_aggregate_method(aggregate_method)

    model_class = TrainTestModel.find_subclass(model_param.model_type)
    # run nested kfold cv for each combintation
    cv_output = ModelCrossValidation.run_kfold_cross_validation(
        model_class,
        model_param.model_param_dict,
        results,
        kfold,
        logger=logger,
    )

    print 'Feature parameters: {}'.format(feature_param.feature_dict)
    print 'Model type: {}'.format(model_param.model_type)
    print 'Model parameters: {}'.format(model_param.model_param_dict)
    print 'Stats: {}'.format(model_class.format_stats(cv_output['aggr_stats']))

    if ax is not None:
        model_class.plot_scatter(ax, cv_output['aggr_stats'], cv_output['contentids'])
        ax.set_xlabel('True Score')
        ax.set_ylabel("Predicted Score")
        ax.grid()
        ax.set_title( "Dataset: {dataset}, Model: {model},\n{stats}".format(
            dataset=dataset.dataset_name,
            model=model_param.model_type,
            stats=model_class.format_stats(cv_output['aggr_stats'])
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
                output_model_filepath=None,
                **kwargs):

    logger = get_stdout_logger()
    result_store = FileSystemResultStore()

    train_dataset = import_python_file(train_dataset_filepath)
    test_dataset = import_python_file(test_dataset_filepath) if test_dataset_filepath is not None else None

    param = import_python_file(param_filepath)

    # === plot scatter ===

    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(figsize=(5*ncols, 5*nrows), nrows=nrows, ncols=ncols)

    train_test_vmaf_on_dataset(train_dataset, test_dataset, param, param, axs[0], axs[1],
                               result_store, parallelize=True, logger=None,
                               output_model_filepath=output_model_filepath,
                               **kwargs)

    if 'xlim' in kwargs:
        axs[0].set_xlim(kwargs['xlim'])
        axs[1].set_xlim(kwargs['xlim'])

    if 'ylim' in kwargs:
        axs[0].set_ylim(kwargs['ylim'])
        axs[1].set_ylim(kwargs['ylim'])

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


def explain_model_on_dataset(model, test_assets_selected_indexs,
                             test_dataset_filepath):

    def print_assets(test_assets):
        print '\n'.join(map(
            lambda (i, asset): "Asset {i}: {name}".format(
                i=i, name=get_file_name_without_extension(asset.dis_path)),
            enumerate(test_assets)
        ))

    test_dataset = import_python_file(test_dataset_filepath)
    test_assets = read_dataset(test_dataset)
    print_assets(test_assets)
    print "Assets selected for local explanation: {}".format(
        test_assets_selected_indexs)
    result_store = FileSystemResultStore()
    test_assets = [test_assets[i] for i in test_assets_selected_indexs]
    test_fassembler = FeatureAssembler(
        feature_dict=model.model_dict['feature_dict'],
        feature_option_dict=None,
        assets=test_assets,
        logger=None,
        fifo_mode=True,
        delete_workdir=True,
        result_store=result_store,
        optional_dict=None,
        optional_dict2=None,
        parallelize=True,
    )
    test_fassembler.run()
    test_feature_results = test_fassembler.results
    test_xs = model.get_xs_from_results(test_feature_results)
    test_ys = model.get_ys_from_results(test_feature_results)
    test_ys_pred = model.predict(test_xs)
    explainer = LocalExplainer(neighbor_samples=1000)
    test_exps = explainer.explain(model, test_xs)

    explainer.print_explanations(test_exps, assets=test_assets, ys=test_ys, ys_pred=test_ys_pred)
    explainer.plot_explanations(test_exps, assets=test_assets, ys=test_ys, ys_pred=test_ys_pred)
    plt.show()


def run_subjective_models(dataset_filepath, subjective_model_classes, do_plot=None, **kwargs):

    if do_plot is None:
        do_plot = []

    if 'dataset_reader_class' in kwargs:
        dataset_reader_class = kwargs['dataset_reader_class']
    else:
        dataset_reader_class = RawDatasetReader

    if 'dataset_reader_info_dict' in kwargs:
        dataset_reader_info_dict = kwargs['dataset_reader_info_dict']
    else:
        dataset_reader_info_dict = {}

    dataset = import_python_file(dataset_filepath)
    dataset_reader = dataset_reader_class(dataset, input_dict=dataset_reader_info_dict)

    subjective_models = map(
        lambda subjective_model_class: subjective_model_class(dataset_reader),
        subjective_model_classes
    )

    results = map(
        lambda subjective_model: subjective_model.run_modeling(**kwargs),
        subjective_models
    )

    if do_plot == 'all' or 'raw_scores' in do_plot:
        # ===== plot raw scores
        plt.figure(figsize=(5, 2.5))
        mtx = dataset_reader.opinion_score_2darray.T
        S, E = mtx.shape
        plt.imshow(mtx, interpolation='nearest')
        # xs = np.array(range(S)) + 1
        # my_xticks = map(lambda x: "#{}".format(x), xs)
        # plt.yticks(np.array(xs), my_xticks, rotation=0)
        plt.title(r'Raw Opinion Scores ($x_{es}$)')
        plt.xlabel(r'Impaired Video Encodes ($e$)')
        plt.ylabel(r'Test Subjects ($s$)')
        plt.set_cmap('gray')
        plt.tight_layout()

    if do_plot == 'all' or 'quality_scores' in do_plot:
        # ===== plot quality scores =====
        bar_width = 0.4
        fig, ax_quality = plt.subplots(figsize=(10, 2.5), nrows=1)
        xs = None
        shift_count = 0
        colors = ['black', 'white', 'blue', 'red']
        my_xticks = None
        for subjective_model, result in zip(subjective_models, results):
            if 'quality_scores' in result:
                quality = result['quality_scores']
                xs = range(len(quality))
                # plt.plot(result['quality_scores'], label=subjective_model.TYPE)
                ax_quality.bar(np.array(xs)+shift_count*bar_width, quality,
                            width=bar_width,
                            color=colors[shift_count],
                            label=subjective_model.TYPE)
                ax_quality.set_xlabel(r'Impaired Video Encodes ($e$)')
                ax_quality.set_title(r'Recovered Quality Score ($x_e$)')
                ax_quality.set_xlim([min(xs), max(xs)+1])
                shift_count += 1
        ax_quality.legend(loc=4, ncol=2, frameon=True)
        plt.tight_layout()

    if do_plot == 'all' or 'subject_scores' in do_plot:

        # ===== plot subject bias and inconsistency =====
        bar_width = 0.4
        fig, (ax_bias, ax_inconsty) = plt.subplots(figsize=(5, 3.5), nrows=2, sharex=True)
        xs = None
        shift_count = 0
        colors = ['black', 'white', 'blue', 'red']
        my_xticks = None
        for subjective_model, result in zip(subjective_models, results):

            if 'observer_bias' in result:
                bias = result['observer_bias']
                xs = range(len(bias))
                ax_bias.bar(np.array(xs)+shift_count*bar_width, bias,
                            width=bar_width,
                            color=colors[shift_count],
                            label=subjective_model.TYPE)
                ax_inconsty.set_xlim([min(xs), max(xs)+1])
                ax_bias.set_title(r'Subject Bias ($b_s$)')

                if 'observers' in result:
                    observers = result['observers']
                    assert len(bias) == len(observers)
                    my_xticks = observers
                    plt.xticks(np.array(xs) + 0.5, my_xticks, rotation=90)

            if 'observer_inconsistency' in result:
                inconsty = result['observer_inconsistency']
                xs = range(len(inconsty))
                ax_inconsty.bar(np.array(xs)+shift_count*bar_width, inconsty,
                                width=bar_width,
                                color=colors[shift_count],
                                label=subjective_model.TYPE)
                ax_inconsty.set_xlim([min(xs), max(xs)+1])
                ax_inconsty.set_title(r'Subject Inconsisency ($v_s$)')

            if 'observer_bias' in result:
                shift_count += 1

        if xs and my_xticks is None:
            my_xticks = map(lambda x: "#{}".format(x+1), xs)
            plt.xticks(np.array(xs) + 0.3, my_xticks, rotation=90)
        # ax_bias.legend(loc=1, ncol=2, frameon=False, prop={'size':12})
        plt.tight_layout()

    if do_plot == 'all' or 'content_scores' in do_plot:

        # ===== plot content ambiguity =====
        bar_width = 0.7
        fig, ax_ambgty = plt.subplots(figsize=(5, 3.5), nrows=1)
        xs = None
        shift_count = 0
        colors = ['black', 'white', 'blue', 'red']
        for subjective_model, result in zip(subjective_models, results):
            if 'content_ambiguity' in result:
                ambgty = result['content_ambiguity']
                xs = range(len(ambgty))
                ax_ambgty.bar(xs, ambgty,
                              width=bar_width,
                              color=colors[shift_count],
                              label=subjective_model.TYPE)
                shift_count += 1
                ax_ambgty.set_title(r'Content Ambiguity ($a_c$)')
        if xs:
            my_xticks = map(lambda ref_video: ref_video['content_name'],
                            dataset_reader.dataset.ref_videos)
            plt.xticks(np.array(xs) + 0.5, my_xticks, rotation=75)
        # ax_ambgty.legend(loc=1, ncol=2, frameon=False, prop={'size':12})
        plt.tight_layout()

    return dataset, subjective_models, results