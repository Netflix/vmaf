#!/usr/bin/env python3
import os
import sys
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file
from vmaf.core.feature_extractor import FeatureExtractor
from vmaf.core.result_store import FileSystemResultStore
from vmaf.core.perf_metric import RmsePerfMetric, SrccPerfMetric, PccPerfMetric

def print_usage():
    print("usage: " + os.path.basename(sys.argv[0]) + " dataset ref_feature_extractor test_feature_extractor [ref_feature:test_feature]*")


def main():
    if len(sys.argv) < 4:
        print_usage()
        return 2

    try:
        dataset_filepath = sys.argv[1]
        ref_feature_extractor_type = sys.argv[2]
        test_feature_extractor_type = sys.argv[3]
    except ValueError:
        print_usage()
        return 2

    test_dataset = import_python_file(dataset_filepath)

    test_assets = read_dataset(test_dataset)
    # print(test_assets)

    ref_fextractor_class = FeatureExtractor.find_subclass(ref_feature_extractor_type)
    test_fextractor_class = FeatureExtractor.find_subclass(test_feature_extractor_type)

    result_store = FileSystemResultStore();
    ref_extractor = ref_fextractor_class(test_assets, None)
    test_extractor = test_fextractor_class(test_assets, None)

    all_ref_features = ref_extractor.ATOM_FEATURES + ref_extractor.DERIVED_ATOM_FEATURES
    all_test_features = test_extractor.ATOM_FEATURES + test_extractor.DERIVED_ATOM_FEATURES

    matching = sorted(set(all_ref_features).intersection(set(all_test_features)))
    ref_features = ["%s_%s_score" % (ref_feature_extractor_type, f) for f in matching]
    test_features = ["%s_%s_score" % (test_feature_extractor_type, f) for f in matching]

    # append user supplied features
    if len(sys.argv) >= 5:
        for pair in sys.argv[4:]:
            [ref_feature, test_feature] = pair.split(':')
            ref_features.append("%s_%s_score" % (ref_feature_extractor_type, ref_feature))
            test_features.append("%s_%s_score" % (test_feature_extractor_type, test_feature))

    print("Comparing the following feature pairs:\n")
    for ref_feature, test_feature in zip(ref_features, test_features):
        print(f"    {ref_feature} => {test_feature}")

    for asset in test_assets:        
        if asset.dis_path == asset.ref_path:
            # skip self comparisons
            continue
        print("")
        print(asset.dis_path)

        ref_result = result_store.load(asset, ref_extractor.executor_id)
        test_result = result_store.load(asset, test_extractor.executor_id)

        ref_result = ref_extractor._post_process_result(ref_result)
        test_result = test_extractor._post_process_result(test_result)

        if ref_result is None or test_result is None:
            print('Missing result for asset ' + asset.asset_id);
            continue
        ref_frames = ref_result.to_dict()['frames']
        test_frames = test_result.to_dict()['frames']
        if(len(ref_frames) != len(test_frames)):
            print('Mismatched number of frames for asset ' + asset.asset_id)

        count = 0

        for ref_feature, test_feature in zip(ref_features, test_features):
            sorted_values = sorted(zip([f[ref_feature] for f in ref_frames], [f[test_feature] for f in test_frames]))
            ref_values = [x for x, _ in sorted_values]
            test_values = [y for _, y in sorted_values]
            
            rmse = RmsePerfMetric(ref_values, test_values).evaluate(enable_mapping=True)['score']
            srcc = SrccPerfMetric(ref_values, test_values).evaluate(enable_mapping=True)['score']
            pcc = PccPerfMetric(ref_values, test_values).evaluate(enable_mapping=True)['score']
            print(f"{test_feature}: PCC {pcc:0.3f} SRCC {srcc:0.3f} RMSE {rmse:0.3f}")

if __name__ == "__main__":
    main()