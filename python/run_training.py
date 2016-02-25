#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys
from tools import import_python_file, get_stdout_logger, close_logger
from result import FileSystemResultStore
import matplotlib.pyplot as plt
from run_testing import read_dataset, plot_scatter
import config
from feature_assembler import FeatureAssembler
from train_test_model import TrainTestModel

def train_on_dataset(train_dataset, feature_param, model_param,
                     ax, result_store, output_model_filepath):

    logger = get_stdout_logger()

    train_assets = read_dataset(train_dataset)
    train_fassembler = FeatureAssembler(
        feature_dict = feature_param.feature_dict,
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

    model_class = TrainTestModel.find_subclass(model_param.model_type)
    model = model_class(model_param.model_param_dict, logger)
    model.train(train_xys)

    train_ys_pred = model.predict(train_xs)
    train_stats = TrainTestModel.get_stats(train_ys['label'], train_ys_pred)

    # plot
    train_content_ids = map(lambda asset: asset.content_id, train_assets)
    TrainTestModel.plot_scatter(ax, train_stats, train_content_ids)
    ax.set_xlabel('DMOS')
    ax.set_ylabel("Predicted Score")
    ax.grid()
    ax.set_title( "Dataset: {dataset}, Model: {model}\n{stats}".format(
        dataset=train_dataset.dataset_name,
        model=model.model_id,
        stats=TrainTestModel.format_stats(train_stats)
    ))

    # save model
    model.to_file(output_model_filepath)

    # clean up
    close_logger(logger)


def print_usage():
    cache_result = ['yes', 'no']
    print "usage: " + os.path.basename(sys.argv[0]) + \
        " cache_result train_dataset_file feature_param_file model_param_file output_model_file\n"
    print "cache_result:\n\t" + "\n\t".join(cache_result) +"\n"

if __name__ == '__main__':

    if len(sys.argv) < 6:
        print_usage()
        exit(2)

    try:
        cache_result = sys.argv[1]
        train_dataset_filepath = sys.argv[2]
        feature_param_filepath = sys.argv[3]
        model_param_filepath = sys.argv[4]
        output_model_filepath = sys.argv[5]
    except ValueError:
        print_usage()
        exit(2)

    try:
        train_dataset = import_python_file(train_dataset_filepath)
        feature_param = import_python_file(feature_param_filepath)
        model_param = import_python_file(model_param_filepath)
    except Exception as e:
        print "Error: " + str(e)
        exit(1)

    if cache_result == 'yes':
        result_store = FileSystemResultStore()
    elif cache_result == 'no':
        result_store = None
    else:
        print_usage()
        exit(2)

    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    train_on_dataset(train_dataset, feature_param, model_param, ax,
                     result_store, output_model_filepath)
    plt.tight_layout()
    plt.show()

    print 'Done.'

    exit(0)