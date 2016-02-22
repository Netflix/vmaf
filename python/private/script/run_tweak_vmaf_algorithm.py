__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
import matplotlib.pyplot as plt
from run_validate_dataset import validate_dataset

if __name__ == '__main__':

    sys.path.append(config.ROOT + '/python/private/script')

    # import example_dataset as dataset
    # import NFLX_dataset_public as dataset
    import NFLX_dataset as dataset

    from quality_runner import VmafQualityRunner as runner_class

    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(figsize=(5*ncols, 5*nrows), nrows=nrows, ncols=ncols)

    # validate_dataset(dataset, runner_class, axs[0], train_or_test='all')
    validate_dataset(dataset, runner_class, axs[0], train_or_test='train')
    validate_dataset(dataset, runner_class, axs[1], train_or_test='test')

    bbox = {'facecolor':'white', 'alpha':1, 'pad':20}
    axs[0].text(80, 10, "Training Set", bbox=bbox)
    axs[1].text(80, 10, "Testing Set", bbox=bbox)

    plt.tight_layout()
    plt.show()

    print 'Done.'