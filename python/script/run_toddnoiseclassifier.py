__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import os

import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.executor import run_executors_in_parallel
from vmaf.core.raw_extractor import DisYUVRawVideoExtractor
from vmaf.core.nn_train_test_model import ToddNoiseClassifierTrainTestModel
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file


def main():
    # parameters
    num_train = 500
    num_test = 50
    n_epochs = 30
    seed = 0 # None

    # read input dataset
    dataset_path = VmafConfig.resource_path('dataset', 'BSDS500_noisy_dataset.py')
    dataset = import_python_file(dataset_path)
    assets = read_dataset(dataset)

    # shuffle assets
    np.random.seed(seed)
    np.random.shuffle(assets)
    assets = assets[:(num_train + num_test)]

    raw_video_h5py_filepath = VmafConfig.workdir_path('rawvideo.hdf5')
    raw_video_h5py_file = DisYUVRawVideoExtractor.open_h5py_file(raw_video_h5py_filepath)

    print('======================== Extract raw YUVs ==============================')

    _, raw_yuvs = run_executors_in_parallel(
        DisYUVRawVideoExtractor,
        assets,
        fifo_mode=True,
        delete_workdir=True,
        parallelize=False, # CAN ONLY USE SERIAL MODE FOR DisYRawVideoExtractor
        result_store=None,
        optional_dict=None,
        optional_dict2={'h5py_file': raw_video_h5py_file})

    patch_h5py_filepath = VmafConfig.workdir_path('patch.hdf5')
    patch_h5py_file = ToddNoiseClassifierTrainTestModel.open_h5py_file(patch_h5py_filepath)
    model = ToddNoiseClassifierTrainTestModel(
        param_dict={
            'seed': seed,
            'n_epochs': n_epochs,
        },
        logger=None,
        optional_dict2={ # for options that won't impact the result
            # 'checkpoints_dir': VmafConfig.workspace_path('checkpoints_dir'),
            'h5py_file': patch_h5py_file,
        })

    print('============================ Train model ===============================')
    xys = ToddNoiseClassifierTrainTestModel.get_xys_from_results(raw_yuvs[:num_train])
    model.train(xys)

    print('=========================== Evaluate model =============================')
    xs = ToddNoiseClassifierTrainTestModel.get_xs_from_results(raw_yuvs[num_train:])
    ys = ToddNoiseClassifierTrainTestModel.get_ys_from_results(raw_yuvs[num_train:])
    result = model.evaluate(xs, ys)

    print("")
    print("f1 test %g, errorrate test %g" % (result['f1'], result['errorrate']))

    # tear down
    DisYUVRawVideoExtractor.close_h5py_file(raw_video_h5py_file)
    ToddNoiseClassifierTrainTestModel.close_h5py_file(patch_h5py_file)
    os.remove(raw_video_h5py_filepath)
    os.remove(patch_h5py_filepath)

    print('Done.')


if __name__ == "__main__":
    main()
