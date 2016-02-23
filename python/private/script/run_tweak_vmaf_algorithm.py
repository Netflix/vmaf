__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
import matplotlib.pyplot as plt
from run_validate_dataset import validate_dataset, read_dataset
from feature_assembler import FeatureAssembler
from result import FileSystemResultStore
from train_test_model import TrainTestModel, LibsvmnusvrTrainTestModel
from tools import get_stdout_logger, close_logger

def run_joe_vmaf():

    sys.path.append(config.ROOT + '/python/private/script')

    import NFLX_dataset as dataset

    from quality_runner import VmafQualityRunner as runner_class

    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(figsize=(5*ncols, 5*nrows), nrows=nrows, ncols=ncols)

    validate_dataset(dataset, runner_class, axs[0], train_or_test='train')
    validate_dataset(dataset, runner_class, axs[1], train_or_test='test')

    bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
    axs[0].text(80, 10, "Training Set", bbox=bbox)
    axs[1].text(80, 10, "Testing Set", bbox=bbox)

    plt.tight_layout()

def run_vmaf_train_test():

    logger = get_stdout_logger()
    result_store = FileSystemResultStore()
    sys.path.append(config.ROOT + '/python/private/script')
    import NFLX_dataset as dataset

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
    model = LibsvmnusvrTrainTestModel({'norm_type':'normalize'}, logger)
    model.train(train_xys)
    train_stats = model.evaluate(train_xs, train_ys)

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
    test_stats = model.evaluate(test_xs, test_ys)

    # === plot scatter ===
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(figsize=(5*ncols, 5*nrows), nrows=nrows, ncols=ncols)

    train_content_ids = map(lambda asset: asset.content_id, train_assets)
    TrainTestModel.plot_scatter(axs[0], train_stats, train_content_ids)
    axs[0].set_xlabel('Groundtruth (DMOS)')
    axs[0].set_ylabel("Prediction")
    axs[0].grid()
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
    axs[1].set_title( "Dataset: {dataset}, Runner: {runner}\n{stats}".format(
        dataset=dataset.dataset_name,
        runner="VMAF (retrained)",
        stats=TrainTestModel.format_stats(test_stats)
    ))

    bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
    axs[0].text(80, 10, "Training Set", bbox=bbox)
    axs[1].text(80, 10, "Testing Set", bbox=bbox)
    plt.tight_layout()

    # === clean up ===
    close_logger(logger)

if __name__ == '__main__':

    # Run Joe's VMAF on NFLX dataset
    run_joe_vmaf()

    # Retrain and test VMAF using NFLX dataset
    run_vmaf_train_test()

    plt.show()

    print 'Done.'