#!/usr/bin/env python

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys
from tools import import_python_file
from result import FileSystemResultStore
from run_testing import read_dataset, print_matplotlib_warning
import config
from feature_assembler import FeatureAssembler
from train_test_model import TrainTestModel
from quality_runner import VmafQualityRunner


def train_on_dataset(train_dataset, feature_param, model_param,
                     ax, result_store, output_model_filepath,
                     parallelize=False, fifo_mode=True, logger=None):

    train_assets = read_dataset(train_dataset)
    train_fassembler = FeatureAssembler(
        feature_dict = feature_param.feature_dict,
        feature_option_dict = None,
        assets = train_assets,
        logger=logger,
        log_file_dir=config.ROOT + "/workspace/log_file_dir",
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
    if 'dis1st_thr' in model_param_dict:
        VmafQualityRunner.set_warp_score(model, model_param_dict['dis1st_thr'])

    train_ys_pred = model.predict(train_xs)

    # apply instructions indicated in the appended info
    train_ys_pred = VmafQualityRunner.clip_score(model, train_ys_pred)
    train_ys_pred = VmafQualityRunner.warp_score(model, train_xs, train_ys_pred)

    train_stats = TrainTestModel.get_stats(train_ys['label'], train_ys_pred)

    print 'Stats on training data: {}'.format(TrainTestModel.format_stats(train_stats))

    # plot
    train_content_ids = map(lambda asset: asset.content_id, train_assets)
    if ax is not None:
        TrainTestModel.plot_scatter(ax, train_stats, train_content_ids)
        ax.set_xlabel('DMOS')
        ax.set_ylabel("Predicted Score")
        ax.grid()
        ax.set_title( "Dataset: {dataset}, Model: {model}\n{stats}".format(
            dataset=train_dataset.dataset_name,
            model=model.model_id,
            stats=TrainTestModel.format_stats(train_stats)
        ))

        bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
        ax.annotate('Training Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

    # save model
    model.to_file(output_model_filepath)

    return train_fassembler, train_ys_pred

def print_usage():
    cache_result = ['yes', 'no']
    parallelize = ['yes', 'no']
    print "usage: " + os.path.basename(sys.argv[0]) + \
        " cache_result parallelize train_dataset_file feature_param_file model_param_file output_model_file\n"
    print "cache_result:\n\t" + "\n\t".join(cache_result) +"\n"
    print "parallelize:\n\t" + "\n\t".join(parallelize) +"\n"

if __name__ == '__main__':

    if len(sys.argv) < 7:
        print_usage()
        exit(2)

    try:
        cache_result = sys.argv[1]
        do_parallelize = sys.argv[2]
        train_dataset_filepath = sys.argv[3]
        feature_param_filepath = sys.argv[4]
        model_param_filepath = sys.argv[5]
        output_model_filepath = sys.argv[6]
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

    if do_parallelize == 'yes':
        parallelize = True
    elif do_parallelize == 'no':
        parallelize = False
    else:
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
        train_on_dataset(train_dataset, feature_param, model_param, ax,
                         result_store, output_model_filepath, parallelize)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print_matplotlib_warning()
        train_on_dataset(train_dataset, feature_param, model_param, None,
                         result_store, output_model_filepath, parallelize)

    print 'Done.'

    exit(0)