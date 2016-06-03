__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import config
from routine import test_on_dataset, train_test_on_dataset, cv_on_dataset
from core.result_store import FileSystemResultStore
from tools.misc import get_stdout_logger, close_logger, import_python_file
from tools.stats import ListStats

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
                          result_store, parallelize=False, logger=None,
                          output_model_filepath=output_model_filepath)

    axs[0].set_xlim([0, 120])
    axs[0].set_ylim([0, 120])

    axs[1].set_xlim([0, 120])
    axs[1].set_ylim([0, 120])

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


if __name__ == '__main__':

    # ==== Run simple cross validation: one training and one testing dataset ====

    run_vmaf_cv(
        train_dataset_filepath=config.ROOT + '/resource/dataset/NFLX_dataset_public.py',
        test_dataset_filepath=config.ROOT + '/resource/dataset/VQEGHD3_dataset.py',
        param_filepath=config.ROOT + '/resource/param/vmaf_v3.py',
        output_model_filepath=config.ROOT + '/workspace/model/test_model1.pkl'
    )

    # ==== Run cross validation across genres (tough test) ====

    nflx_dataset_path = config.ROOT + '/resource/dataset/NFLX_dataset_public.py'
    contentid_groups = [
        [0, 5], # cartoon: BigBuckBunny, FoxBird
        [1], # CG: BirdsInCage
        [2, 6, 7], # complex: CrowdRun, OldTownCross, Seeking
        [3, 4], # ElFuente: ElFuente1, ElFuente2
        [8], # sports: Tennis
    ]
    param_filepath = config.ROOT + '/resource/param/vmaf_v3.py'

    aggregate_method = np.mean
    # aggregate_method = ListStats.harmonic_mean
    # aggregate_method = partial(ListStats.lp_norm, p=2.0)

    run_vmaf_kfold_cv(
        dataset_filepath=nflx_dataset_path,
        contentid_groups=contentid_groups,
        param_filepath=param_filepath,
        aggregate_method=aggregate_method,
    )

    plt.show()

    print 'Done.'