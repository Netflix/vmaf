__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import numpy as np

from vmaf.config import VmafConfig, DisplayConfig
from vmaf.routine import run_vmaf_cv, run_vmaf_kfold_cv

if __name__ == '__main__':

    # ==== Run simple cross validation: one training and one testing dataset ====

    run_vmaf_cv(
        train_dataset_filepath=VmafConfig.resource_path('dataset', 'NFLX_dataset_public.py'),
        test_dataset_filepath=VmafConfig.resource_path('dataset', 'VQEGHD3_dataset.py'),
        param_filepath=VmafConfig.resource_path('param', 'vmaf_v3.py'),
        output_model_filepath=VmafConfig.workspace_path('model', 'test_model1.pkl'),
    )

    # ==== Run cross validation across genres (tough test) ====

    nflx_dataset_path = VmafConfig.resource_path('dataset', 'NFLX_dataset_public.py')
    contentid_groups = [
        [0, 5],  # cartoon: BigBuckBunny, FoxBird
        [1],  # CG: BirdsInCage
        [2, 6, 7],  # complex: CrowdRun, OldTownCross, Seeking
        [3, 4],  # ElFuente: ElFuente1, ElFuente2
        [8],  # sports: Tennis
    ]
    param_filepath = VmafConfig.resource_path('param', 'vmaf_v3.py')

    aggregate_method = np.mean
    # aggregate_method = ListStats.harmonic_mean
    # aggregate_method = partial(ListStats.lp_norm, p=2.0)

    run_vmaf_kfold_cv(
        dataset_filepath=nflx_dataset_path,
        contentid_groups=contentid_groups,
        param_filepath=param_filepath,
        aggregate_method=aggregate_method,
    )

    DisplayConfig.show()

    print('Done.')
