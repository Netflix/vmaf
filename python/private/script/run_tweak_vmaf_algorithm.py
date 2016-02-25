from cross_validation import FeatureCrossValidation

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
import matplotlib.pyplot as plt
from run_validate_dataset import validate_dataset, read_dataset
from feature_assembler import FeatureAssembler
from result import FileSystemResultStore
from train_test_model import TrainTestModel, LibsvmnusvrTrainTestModel
from tools import get_stdout_logger, close_logger, indices
import numpy as np

def run_joe_vmaf():

    sys.path.append(config.ROOT + '/python/private/script')

    import NFLX_dataset as dataset

    from quality_runner import VmafQualityRunner as runner_class

    result_store = FileSystemResultStore()

    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(figsize=(5*ncols, 5*nrows), nrows=nrows, ncols=ncols)

    validate_dataset(dataset, runner_class, axs[0], result_store, 'train')
    validate_dataset(dataset, runner_class, axs[1], result_store, 'test')

    bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
    axs[0].text(80, 10, "Training Set", bbox=bbox)
    axs[1].text(80, 10, "Testing Set", bbox=bbox)

    plt.tight_layout()

def run_vmaf_train_test():

    logger = get_stdout_logger()
    result_store = FileSystemResultStore()
    sys.path.append(config.ROOT + '/python/private/script')
    import NFLX_dataset as dataset

    libsvmnusr_param_dict = {

        # 'norm_type':'none', # default
        'norm_type':'clip_0to1', # Joe's - select
        # 'norm_type':'clip_minus1to1',
        # 'norm_type':'normalize',

        # 'gamma':0.0, # default
        'gamma':0.85, # Joe's - select

        'C':1.0, # default - select

        'nu':0.5, # default - select

        'cache_size':200 # default - select
    }

    # === train model on training dataset, also test on training dataset ===
    train_assets = read_dataset(dataset, train_or_test='train')
    train_fassembler = FeatureAssembler(
        feature_dict = {'VMAF_feature':'all'},
        feature_option_dict = None,
        assets = train_assets,
        logger=logger,
        log_file_dir=config.ROOT + "/workspace/log_file_dir",
        fifo_mode=True,
        delete_workdir=True,
        result_store=result_store)
    train_fassembler.run()
    train_features = train_fassembler.results
    train_xys = TrainTestModel.get_xys_from_results(train_features)
    train_xs = TrainTestModel.get_xs_from_results(train_features)
    train_ys = TrainTestModel.get_ys_from_results(train_features)
    model = LibsvmnusvrTrainTestModel(libsvmnusr_param_dict, logger)
    model.train(train_xys)

    train_ys_pred = model.predict(train_xs)
    train_ys_pred = np.clip(train_ys_pred, 0.0, 100.0)
    train_stats = TrainTestModel.get_stats(train_ys['label'], train_ys_pred)

    # === test model on test dataset ===
    test_assets = read_dataset(dataset, train_or_test='test')
    test_fassembler = FeatureAssembler(
        feature_dict = {'VMAF_feature':'all'},
        feature_option_dict = None,
        assets = test_assets,
        logger=logger,
        log_file_dir=config.ROOT + "/workspace/log_file_dir",
        fifo_mode=True,
        delete_workdir=True,
        result_store=result_store)
    test_fassembler.run()
    test_features = test_fassembler.results
    test_xs = TrainTestModel.get_xs_from_results(test_features)
    test_ys = TrainTestModel.get_ys_from_results(test_features)

    test_ys_pred = model.predict(test_xs)
    test_ys_pred = np.clip(test_ys_pred, 0.0, 100.0)
    test_stats = TrainTestModel.get_stats(test_ys['label'], test_ys_pred)

    # === plot scatter ===
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(figsize=(5*ncols, 5*nrows), nrows=nrows, ncols=ncols)

    train_content_ids = map(lambda asset: asset.content_id, train_assets)
    TrainTestModel.plot_scatter(axs[0], train_stats, train_content_ids)
    axs[0].set_xlabel('Groundtruth (DMOS)')
    axs[0].set_ylabel("Prediction")
    axs[0].grid()
    axs[0].set_xlim([0, 120])
    axs[0].set_ylim([0, 120])
    axs[0].set_title( "Dataset: {dataset}, Runner: {runner}\n{stats}".format(
        dataset=dataset.dataset_name,
        runner="VMAF (retrained)",
        stats=TrainTestModel.format_stats(train_stats)
    ))

    test_content_ids = map(lambda asset: asset.content_id, test_assets)
    TrainTestModel.plot_scatter(axs[1], test_stats, test_content_ids)
    axs[1].set_xlabel('Groundtruth (DMOS)')
    axs[1].set_ylabel("Prediction")
    axs[1].grid()
    axs[1].set_xlim([0, 120])
    axs[1].set_ylim([0, 120])
    axs[1].set_title( "Dataset: {dataset}, Runner: {runner}\n{stats}".format(
        dataset=dataset.dataset_name,
        runner="VMAF (retrained)",
        stats=TrainTestModel.format_stats(test_stats)
    ))

    bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
    axs[0].text(80, 10, "Training Set", bbox=bbox)
    axs[1].text(80, 10, "Testing Set", bbox=bbox)
    plt.tight_layout()

    # save model
    retrained_vmaf_model_path = config.ROOT + '/workspace/model/model_v9.model'
    model.to_file(retrained_vmaf_model_path)

    # === clean up ===
    close_logger(logger)

def run_vmaf_tough_test():

    logger = get_stdout_logger()
    result_store = FileSystemResultStore()
    sys.path.append(config.ROOT + '/python/private/script')
    import NFLX_dataset as dataset

    assets = read_dataset(dataset, train_or_test='all')
    fassembler = FeatureAssembler(
        feature_dict = {'VMAF_feature':'all'},
        feature_option_dict = None,
        assets = assets,
        logger=logger,
        log_file_dir=config.ROOT + "/workspace/log_file_dir",
        fifo_mode=True,
        delete_workdir=True,
        result_store=result_store)
    fassembler.run()
    feature_results = fassembler.results

    content_groups=[
        [3, 4, 5, 30, 15], # cartoon
        [0, 1, 2, 6, 7, 8, 9, 10], # film-grain
        [11, 12, 13, 14, 25], # synthetic
        [19, 20, 21, 22, 23, 24], # action
        [27, 28, 29, 16, 17, 18, 31, 32, 33, 26], # other
        # [-1, -2, -3, -4] # injected blackframes
    ]

    content_ids = map(lambda asset: asset.content_id, assets)

    # construct kfold
    kfold = []
    for curr_content_group in content_groups:
        curr_indices = indices(content_ids, lambda x: x in curr_content_group)
        kfold.append(curr_indices)

    modelparam_searchrange = {
        'nu':[1],
        'C':[20],
        'gamma':[0.0],
    }

    # run nested kfold cv for each combintation
    output_vmaf_libsvmnusvr = FeatureCrossValidation.run_nested_kfold_cross_validation(
        LibsvmnusvrTrainTestModel,
        modelparam_searchrange,
        feature_results,
        kfold
    )

    # TODO: finish this...

if __name__ == '__main__':

    # Run Joe's VMAF on NFLX dataset
    # run_joe_vmaf()

    # Retrain and test VMAF using NFLX dataset
    run_vmaf_train_test()

    # Run cross validation across genres (tough test)
    # run_vmaf_tough_test()

    plt.show()

    print 'Done.'