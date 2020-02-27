from time import time

import numpy as np

from vmaf import plt
from vmaf.config import VmafConfig, DisplayConfig
from vmaf.core.quality_runner import VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.core.train_test_model import RegressorMixin
from vmaf.routine import run_test_on_dataset
from vmaf.tools.decorator import persist_to_dir
from vmaf.tools.misc import import_python_file

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


@persist_to_dir(VmafConfig.workdir_path("run_vmafossexec_subsampling"))
def run_vmafossexec_with_subsample(dataset_filepath, subsample):

    result_store = FileSystemResultStore()
    runner_class = VmafossExecQualityRunner

    dataset = import_python_file(dataset_filepath)

    start_time = time()
    assets, results = run_test_on_dataset(dataset, runner_class, None, result_store,
                                          model_filepath=None, parallelize=False,
                                          fifo_mode=True, subsample=subsample)
    elapsed_time = time() - start_time

    groundtruths = list(map(lambda asset: asset.groundtruth, assets))
    predictions = list(map(lambda result: result[runner_class.get_score_key()], results))
    stats = RegressorMixin.get_stats(groundtruths, predictions)
    srcc = stats['SRCC']
    pcc = stats['PCC']
    rmse = stats['RMSE']

    return elapsed_time, srcc, pcc, rmse


def main():

    subsamples = [1, 2, 4, 8, 16, 32, 64, 128]
    elapsed_times = []
    pccs = []
    for subsample in subsamples:
        elapsed_time, srcc, pcc, rmse = run_vmafossexec_with_subsample(
            VmafConfig.resource_path('dataset', 'NFLX_dataset_public.py'), subsample)
        elapsed_times.append(elapsed_time)
        pccs.append(pcc)
        print("SRCC: {}, PCC: {}, RMSE: {}, time: {}".format(srcc, pcc, rmse, elapsed_time))

    fig, ax = plt.subplots(1, 1, figsize=[8, 5])
    ax.plot(subsamples, 6*24*79 / np.array(elapsed_times), 'x-')
    ax.set_xlabel("Subsample")
    ax.set_ylabel("Processing Speed (Frms/Sec)")
    ax.grid(True)

    plt.tight_layout()

    DisplayConfig.show()


if __name__ == '__main__':
    main()
    print('Done.')
