import scipy

from routine import run_subjective_models
from tools.decorator import persist_to_file

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import config
from tools.misc import import_python_file
from mos.dataset_reader import RawDatasetReader, SyntheticRawDatasetReader, \
    MissingDataRawDatasetReader, SelectSubjectRawDatasetReader, \
    CorruptSubjectRawDatasetReader, CorruptDataRawDatasetReader
from mos.subjective_model import MosModel, DmosModel, \
    MaximumLikelihoodEstimationModelReduced, MaximumLikelihoodEstimationModel, \
    MaximumLikelihoodEstimationDmosModel, SubjrejMosModel, ZscoringSubjrejMosModel

color_dict = {
    'Subject/Content-Aware': 'red',
    'Subject-Aware': 'blue',
    'DMOS': 'green',
    'MOS': 'cyan',
    'LIVE DMOS': 'magenta',
}

def _validate_with_synthetic_dataset(subjective_model_classes, dataset_filepath, synthetic_result):

        dataset = import_python_file(dataset_filepath)

        dataset_reader = SyntheticRawDatasetReader(dataset, input_dict=synthetic_result)

        subjective_models = map(
            lambda subjective_model_class: subjective_model_class(dataset_reader),
            subjective_model_classes
        )

        results = map(
            lambda subjective_model: subjective_model.run_modeling(),
            subjective_models
        )

        # ===== plot scatter =====
        fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2)
        for subjective_model_class, result, idx in zip(
                subjective_model_classes, results, range(len(results))):

            model_name = subjective_model_class.TYPE

            ax = axs.item(0)
            if 'quality_scores' in result and 'quality_scores' in synthetic_result:
                color = color_dict[model_name] if model_name in color_dict else 'black'
                x = synthetic_result['quality_scores']
                y = result['quality_scores']
                ax.scatter(x, y, color=color,
                           label='{sm}'.format(sm=model_name))

            ax = axs.item(1)
            if 'observer_bias' in result and 'observer_bias' in synthetic_result:
                color = color_dict[model_name] if model_name in color_dict else 'black'
                x = synthetic_result['observer_bias']
                y = result['observer_bias']
                min_xy = np.min([len(x),len(y)])
                x = x[:min_xy]
                y = y[:min_xy]
                ax.scatter(x, y, color=color,
                           label='{sm}'.format(sm=model_name))

            ax = axs.item(2)
            if 'observer_inconsistency' in result and 'observer_inconsistency' in synthetic_result:
                color = color_dict[model_name] if model_name in color_dict else 'black'
                x = synthetic_result['observer_inconsistency']
                y = result['observer_inconsistency']
                min_xy = np.min([len(x),len(y)])
                x = x[:min_xy]
                y = y[:min_xy]
                ax.scatter(x, y, color=color,
                           label='{sm}'.format(sm=model_name))

            ax = axs.item(3)
            if 'content_ambiguity' in result and 'content_ambiguity' in synthetic_result:
                color = color_dict[model_name] if model_name in color_dict else 'black'
                x = synthetic_result['content_ambiguity']
                y = result['content_ambiguity']
                ax.scatter(x, y, color=color,
                           label='{sm}'.format(sm=model_name))

        axs.item(0).set_title(r'Quality Score ($x_e$)')
        axs.item(1).set_title(r'Subject Bias ($b_s$)')
        axs.item(2).set_title(r'Subject Inconsisency ($v_s$)')
        # axs.item(3).set_title(r'Content Bias ($\mu_c$)')
        axs.item(3).set_title(r'Content Ambiguity ($a_c$)')

        for i in range(4):
            ax = axs.item(i)
            ax.set_xlabel('Synthetic')
            ax.set_ylabel('Recovered')
            ax.grid()

        plt.tight_layout()

def validate_with_synthetic_dataset():

    # use the dataset_filepath only for its dimensions and reference video mapping
    dataset_filepath = config.ROOT + '/resource/dataset/NFLX_dataset_public_raw_last4outliers.py'
    np.random.seed(0)
    _validate_with_synthetic_dataset(
        subjective_model_classes=[
            MaximumLikelihoodEstimationModel
        ],
        dataset_filepath=dataset_filepath,
        synthetic_result={
            'quality_scores': np.random.uniform(1, 5, 79),
            'observer_bias': np.random.normal(0, 1, 30),
            'observer_inconsistency': np.abs(np.random.uniform(0.0, 0.4, 30)),
            'content_bias': np.random.normal(0, 0.00001, 9),
            'content_ambiguity': np.abs(np.random.uniform(0.4, 0.6, 9)),
        }
    )

def run_datasize_growth(dataset_filepaths):

    @persist_to_file(config.ROOT + '/workspace/workdir/_run_subject_nums.json')
    def _run_subject_nums(dataset, subject_nums, model_class, seed,
                          perf_type='rmse'):
        def run_one_num_subject(num_subject, dataset, seed):
            np.random.seed(seed)
            total_subject = len(dataset.dis_videos[0]['os'])
            info_dict = {
                'selected_subjects': np.random.permutation(total_subject)[:num_subject]
            }
            dataset_reader = SelectSubjectRawDatasetReader(dataset, input_dict=info_dict)
            subjective_model = model_class(dataset_reader)
            result = subjective_model.run_modeling(normalize_final=False)
            return dataset_reader, result

        inputs = []
        for subject_num in subject_nums:
            input = [subject_num, dataset, seed]
            inputs.append(input)
        outputs = map(lambda input: run_one_num_subject(*input), inputs)

        result0 = model_class(RawDatasetReader(dataset)).run_modeling(normalize_final=False)

        result0_qs = np.array(result0['quality_scores'])
        result0_qs_mean = np.mean(result0_qs)
        result0_qs_std = np.std(result0_qs)
        result0_qs = (result0_qs - result0_qs_mean) / result0_qs_std

        perfs = []
        datasizes = []
        for output in outputs:
            reader, result = output

            result_qs = np.array(result['quality_scores'])
            result_qs = (result_qs - result0_qs_mean) / result0_qs_std

            if perf_type == 'pcc':
                perf, _ = scipy.stats.pearsonr(result_qs, result0_qs)
            elif perf_type == 'rmse':
                perf = np.sqrt(np.mean(np.power(result_qs - result0_qs, 2.0)))
            else:
                assert False
            datasize = reader.opinion_score_2darray.shape[1]
            perfs.append(perf)
            datasizes.append(datasize)
        return datasizes, perfs

    for dataset_filepath in dataset_filepaths:

        subject_nums = np.arange(30, 10, -1)

        model_classes = [
            MosModel,
            SubjrejMosModel,
            ZscoringSubjrejMosModel,
            MaximumLikelihoodEstimationModel,
        ]

        seedss = [
            range(10),
            range(10),
            range(10),
            range(1),
        ]

        linestyles = ['--', '-.', ':', '-']
        linewidths = [1.5, 1.5, 3, 1.5]

        dataset = import_python_file(dataset_filepath)

        datasizesss = [[] for _ in model_classes]
        perfsss = [[] for _ in model_classes]

        i = 0
        for model_class, seeds in zip(model_classes, seedss):
            print 'class {}...'.format(model_class.__name__)
            for seed in seeds:
                datasizes, perfs = _run_subject_nums(dataset, subject_nums, model_class, seed)
                datasizesss[i].append(datasizes)
                perfsss[i].append(perfs)
            i += 1

        plt.figure(figsize=(5, 3.5))

        for i, model_class in enumerate(model_classes):
            plt.plot(np.mean(np.asarray(datasizesss[i]), axis=0),
                     np.mean(np.asarray(perfsss[i]), axis=0),
                     label=model_class.TYPE,
                     linestyle=linestyles[i],
                     linewidth=linewidths[i],
                     color='black',
            )
        plt.xlabel('No. Subjects')
        plt.ylabel(r'RMSE of Quality Scores ($q_e$)')
        plt.legend(loc=1, ncol=2, prop={'size':12}, frameon=False)
        plt.tight_layout()

def run_subject_corruption_growth(dataset_filepaths):

    @persist_to_file(config.ROOT + '/workspace/workdir/_run_corrupt_nums.json')
    def _run_corrupt_nums(dataset, subject_nums, model_class, seed,
                          perf_type='rmse'):
        def run_one_num_subject(num_subject, dataset, seed):
            np.random.seed(seed)
            info_dict = {
                'selected_subjects': np.random.permutation(len(
                    dataset.dis_videos[0]['os']))[:num_subject]
            }
            dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
            subjective_model = model_class(dataset_reader)
            result = subjective_model.run_modeling(normalize_final=False)
            return dataset_reader, result

        inputs = []
        for subject_num in subject_nums:
            input = [subject_num, dataset, seed]
            inputs.append(input)
        outputs = map(lambda input: run_one_num_subject(*input), inputs)

        result0 = model_class(RawDatasetReader(dataset)).run_modeling(normalize_final=False)

        result0_qs = np.array(result0['quality_scores'])
        result0_qs_mean = np.mean(result0_qs)
        result0_qs_std = np.std(result0_qs)
        result0_qs = (result0_qs - result0_qs_mean) / result0_qs_std

        perfs = []
        datasizes = []
        for input, output in zip(inputs, outputs):
            subject_num, dataset, seed = input
            reader, result = output

            result_qs = np.array(result['quality_scores'])
            result_qs = (result_qs - result0_qs_mean) / result0_qs_std

            if perf_type == 'pcc':
                perf, _ = scipy.stats.pearsonr(result_qs, result0_qs)
            elif perf_type == 'rmse':
                perf = np.sqrt(np.mean(np.power(result_qs - result0_qs, 2.0)))
            else:
                assert False
            # datasize = np.prod(subject_num * len(reader.dataset.dis_videos))
            datasize = np.prod(subject_num)
            perfs.append(perf)
            datasizes.append(datasize)
        return datasizes, perfs

    for dataset_filepath in dataset_filepaths:

        subject_nums = np.arange(0, 20, 1)

        model_classes = [
            MosModel,
            SubjrejMosModel,
            ZscoringSubjrejMosModel,
            MaximumLikelihoodEstimationModel,
        ]

        seedss = [
            range(10),
            range(10),
            range(10),
            range(1),
        ]

        linestyles = ['--', '-.', ':', '-']
        linewidths = [1.5, 1.5, 3, 1.5]

        dataset = import_python_file(dataset_filepath)

        datasizesss = [[] for _ in model_classes]
        perfsss = [[] for _ in model_classes]

        i = 0
        for model_class, seeds in zip(model_classes, seedss):
            print 'class {}...'.format(model_class.__name__)
            for seed in seeds:
                datasizes, perfs = _run_corrupt_nums(
                    dataset, subject_nums, model_class, seed)
                datasizesss[i].append(datasizes)
                perfsss[i].append(perfs)
            i += 1

        plt.figure(figsize=(5, 3.5))

        for i, model_class in enumerate(model_classes):
            plt.plot(np.mean(np.asarray(datasizesss[i]), axis=0),
                     np.mean(np.asarray(perfsss[i]), axis=0),
                     label=model_class.TYPE,
                     linestyle=linestyles[i],
                     linewidth=linewidths[i],
                     color='black',
            )
        plt.xlabel('No. Corrupted Subjects')
        plt.ylabel(r'RMSE of Quality Scores ($x_e$)')
        plt.legend(loc=2, ncol=2, prop={'size':12}, frameon=False)
        plt.tight_layout()

def run_missing_growth(dataset_filepaths):

    @persist_to_file(config.ROOT + '/workspace/workdir/_run_missing_probs.json')
    def _run_missing_probs(dataset, missing_probs, model_class, seed,
                           perf_type='rmse'):
        def run_one_missing_prob(missing_prob, dataset, seed):
            np.random.seed(seed)
            info_dict = {
                'missing_probability': missing_prob,
            }
            dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)
            subjective_model = model_class(dataset_reader)
            try:
                result = subjective_model.run_modeling(normalize_final=False)
            except ValueError as e:
                print 'Warning: {}, return result None'.format(e)
                result = None
            return dataset_reader, result

        inputs = []
        for missing_prob in missing_probs:
            input = [missing_prob, dataset, seed]
            inputs.append(input)
        outputs = map(lambda input: run_one_missing_prob(*input), inputs)

        result0 = model_class(RawDatasetReader(dataset)).run_modeling(normalize_final=False)

        result0_qs = np.array(result0['quality_scores'])
        result0_qs_mean = np.mean(result0_qs)
        result0_qs_std = np.std(result0_qs)
        result0_qs = (result0_qs - result0_qs_mean) / result0_qs_std

        perfs = []
        datasizes = []
        for output in outputs:
            reader, result = output

            result_qs = np.array(result['quality_scores'])
            result_qs = (result_qs - result0_qs_mean) / result0_qs_std

            if result is None:
                perf = float('NaN')
            else:
                if perf_type == 'pcc':
                    perf, _ = scipy.stats.pearsonr(result_qs, result0_qs)
                elif perf_type == 'rmse':
                    perf = np.sqrt(np.mean(np.power(result_qs - result0_qs, 2.0)))
                else:
                    assert False
            datasize = np.prod(reader.opinion_score_2darray.shape) \
                       - np.isnan(reader.opinion_score_2darray).sum()
            perfs.append(perf)
            datasizes.append(datasize)
        return datasizes, perfs

    for dataset_filepath in dataset_filepaths:

        missing_probs = np.linspace(0.7, 0.0, num=20)

        model_classes = [
            MosModel,
            SubjrejMosModel,
            ZscoringSubjrejMosModel,
            MaximumLikelihoodEstimationModel,
        ]

        seedss = [
            range(10),
            range(10),
            range(10),
            range(1),
        ]

        linestyles = ['--', '-.', ':', '-']
        linewidths = [1.5, 1.5, 3, 1.5]

        dataset = import_python_file(dataset_filepath)

        datasizesss = [[] for _ in model_classes]
        perfsss = [[] for _ in model_classes]

        i = 0
        for model_class, seeds in zip(model_classes, seedss):
            print 'class {}...'.format(model_class.__name__)
            for seed in seeds:
                datasizes, perfs = _run_missing_probs(dataset, missing_probs, model_class, seed)
                datasizesss[i].append(datasizes)
                perfsss[i].append(perfs)
            i += 1

        plt.figure(figsize=(5, 3.5))

        for i, model_class in enumerate(model_classes):
            plt.plot(np.mean(np.asarray(datasizesss[i]), axis=0),
                     np.mean(np.asarray(perfsss[i]), axis=0),
                     label=model_class.TYPE,
                     linestyle=linestyles[i],
                     linewidth=linewidths[i],
                     color='black',
            )

        plt.xlabel('No. Scores Sampled')
        plt.ylabel(r'RMSE of Quality Scores ($x_e$)')
        plt.legend(loc=2, ncol=2, prop={'size':12}, frameon=False)
        plt.tight_layout()

def run_random_corruption_growth(dataset_filepaths):

    @persist_to_file(config.ROOT + '/workspace/workdir/_run_corrupt_probs.json')
    def _run_corrupt_probs(dataset, corrupt_probs, model_class, seed,
                           perf_type='rmse'):
        def run_one_corrput_prob(corrupt_prob, dataset, seed):
            np.random.seed(seed)
            info_dict = {
                'corrupt_probability': corrupt_prob,
            }
            dataset_reader = CorruptDataRawDatasetReader(dataset, input_dict=info_dict)
            subjective_model = model_class(dataset_reader)
            try:
                result = subjective_model.run_modeling(normalize_final=False)
            except ValueError as e:
                print 'Warning: {}, return result None'.format(e)
                result = None
            return dataset_reader, result

        inputs = []
        for corrupt_prob in corrupt_probs:
            input = [corrupt_prob, dataset, seed]
            inputs.append(input)
        outputs = map(lambda input: run_one_corrput_prob(*input), inputs)

        result0 = model_class(RawDatasetReader(dataset)).run_modeling(normalize_final=False)

        result0_qs = np.array(result0['quality_scores'])
        result0_qs_mean = np.mean(result0_qs)
        result0_qs_std = np.std(result0_qs)
        result0_qs = (result0_qs - result0_qs_mean) / result0_qs_std

        perfs = []
        datasizes = []
        for input, output in zip(inputs, outputs):
            corrupt_prob, dataset, seed = input
            reader, result = output

            result_qs = np.array(result['quality_scores'])
            result_qs = (result_qs - result0_qs_mean) / result0_qs_std

            if result is None:
                perf = float('NaN')
            else:
                if perf_type == 'pcc':
                    perf, _ = scipy.stats.pearsonr(result_qs, result0_qs)
                elif perf_type == 'rmse':
                    perf = np.sqrt(np.mean(np.power(result_qs - result0_qs, 2.0)))
                else:
                    assert False
            datasize = corrupt_prob
            perfs.append(perf)
            datasizes.append(datasize)
        return datasizes, perfs

    for dataset_filepath in dataset_filepaths:

        corrupt_probs = np.linspace(0.7, 0.0, num=20)

        model_classes = [
            MosModel,
            SubjrejMosModel,
            ZscoringSubjrejMosModel,
            MaximumLikelihoodEstimationModel,
        ]

        seedss = [
            range(10),
            range(10),
            range(10),
            range(1),
        ]

        linestyles = ['--', '-.', ':', '-']
        linewidths = [1.5, 1.5, 3, 1.5]

        dataset = import_python_file(dataset_filepath)

        datasizesss = [[] for _ in model_classes]
        perfsss = [[] for _ in model_classes]

        i = 0
        for model_class, seeds in zip(model_classes, seedss):
            print 'class {}...'.format(model_class.__name__)
            for seed in seeds:
                datasizes, perfs = _run_corrupt_probs(dataset, corrupt_probs, model_class, seed)
                datasizesss[i].append(datasizes)
                perfsss[i].append(perfs)
            i += 1

        plt.figure(figsize=(5, 3.5))

        for i, model_class in enumerate(model_classes):
            plt.plot(np.mean(np.asarray(datasizesss[i]), axis=0),
                     np.mean(np.asarray(perfsss[i]), axis=0),
                     label=model_class.TYPE,
                     linestyle=linestyles[i],
                     linewidth=linewidths[i],
                     color='black',
            )

        plt.xlabel('Data Corruption Probability')
        plt.ylabel(r'RMSE of Quality Scores ($x_e$)')
        plt.legend(loc=2, ncol=2, prop={'size':12}, frameon=False)
        plt.tight_layout()

def run_subject_partial_corruption_growth(dataset_filepaths):

    @persist_to_file(config.ROOT + '/workspace/workdir/_run_partial_corrupt_nums.json')
    def _run_partial_corrupt_nums(dataset, subject_nums, model_class, seed,
                          perf_type='rmse'):
        def run_one_num_subject(num_subject, dataset, seed):
            np.random.seed(seed)
            info_dict = {
                'selected_subjects': np.random.permutation(len(
                    dataset.dis_videos[0]['os']))[:num_subject],
                'corrupt_probability': 0.5,
            }
            dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
            subjective_model = model_class(dataset_reader)
            result = subjective_model.run_modeling(normalize_final=False)
            return dataset_reader, result

        inputs = []
        for subject_num in subject_nums:
            input = [subject_num, dataset, seed]
            inputs.append(input)
        outputs = map(lambda input: run_one_num_subject(*input), inputs)

        result0 = model_class(RawDatasetReader(dataset)).run_modeling(normalize_final=False)

        result0_qs = np.array(result0['quality_scores'])
        result0_qs_mean = np.mean(result0_qs)
        result0_qs_std = np.std(result0_qs)
        result0_qs = (result0_qs - result0_qs_mean) / result0_qs_std

        perfs = []
        datasizes = []
        for input, output in zip(inputs, outputs):
            subject_num, dataset, seed = input
            reader, result = output

            result_qs = np.array(result['quality_scores'])
            result_qs = (result_qs - result0_qs_mean) / result0_qs_std

            if perf_type == 'pcc':
                perf, _ = scipy.stats.pearsonr(result_qs, result0_qs)
            elif perf_type == 'rmse':
                perf = np.sqrt(np.mean(np.power(result_qs - result0_qs, 2.0)))
            else:
                assert False
            # datasize = np.prod(subject_num * len(reader.dataset.dis_videos))
            datasize = np.prod(subject_num)
            perfs.append(perf)
            datasizes.append(datasize)
        return datasizes, perfs

    for dataset_filepath in dataset_filepaths:

        subject_nums = np.arange(0, 20, 1)

        model_classes = [
            MosModel,
            SubjrejMosModel,
            ZscoringSubjrejMosModel,
            MaximumLikelihoodEstimationModel,
        ]

        seedss = [
            range(10),
            range(10),
            range(10),
            range(1),
        ]

        linestyles = ['--', '-.', ':', '-']
        linewidths = [1.5, 1.5, 3, 1.5]

        dataset = import_python_file(dataset_filepath)

        datasizesss = [[] for _ in model_classes]
        perfsss = [[] for _ in model_classes]

        i = 0
        for model_class, seeds in zip(model_classes, seedss):
            print 'class {}...'.format(model_class.__name__)
            for seed in seeds:
                datasizes, perfs = _run_partial_corrupt_nums(
                    dataset, subject_nums, model_class, seed)
                datasizesss[i].append(datasizes)
                perfsss[i].append(perfs)
            i += 1

        plt.figure(figsize=(5, 3.5))

        for i, model_class in enumerate(model_classes):
            plt.plot(np.mean(np.asarray(datasizesss[i]), axis=0),
                     np.mean(np.asarray(perfsss[i]), axis=0),
                     label=model_class.TYPE,
                     linestyle=linestyles[i],
                     linewidth=linewidths[i],
                     color='black',
            )
        plt.xlabel('No. Corrupted Subjects')
        plt.ylabel(r'RMSE of Quality Scores ($x_e$)')
        plt.legend(loc=2, ncol=2, prop={'size':12}, frameon=False)
        plt.tight_layout()

def plot_sample_results(dataset_filepaths, subjective_model_classes):

    for dataset_filepath in dataset_filepaths:
        run_subjective_models(
            dataset_filepath=dataset_filepath,
            subjective_model_classes = subjective_model_classes,
            normalize_final=False, # True or False
            do_plot=[
                'raw_scores',
                'quality_scores',
                'subject_scores',
                'content_scores',
            ], # 'all' or a list of types
            # dataset_reader_class=CorruptSubjectRawDatasetReader,
            # dataset_reader_info_dict={'selected_subjects': [0, 5, 10, 15]},
        )

def main():

    dataset_filepaths = [
        config.ROOT + '/resource/dataset/NFLX_dataset_public_raw_last4outliers.py',
        config.ROOT + '/resource/dataset/VQEGHD3_dataset_raw.py',
    ]

    # ============ sample results =================

    subjective_model_classes = [
        MaximumLikelihoodEstimationModel,
        MosModel,

        # MaximumLikelihoodEstimationDmosModel,
        # DmosModel,
    ]

    plot_sample_results(dataset_filepaths, subjective_model_classes)

    # ============ plot trends =================

    # ===== datasize growth =====
    run_datasize_growth(dataset_filepaths)

    # ===== corrpution growth =====
    run_subject_corruption_growth(dataset_filepaths)
    run_random_corruption_growth(dataset_filepaths)

    run_subject_partial_corruption_growth(dataset_filepaths)

    # ===== random missing growth =====
    run_missing_growth(dataset_filepaths)

    # ===== synthetic data =====
    validate_with_synthetic_dataset()

    plt.show()

if __name__ == '__main__':
    main()
    print 'Done.'
