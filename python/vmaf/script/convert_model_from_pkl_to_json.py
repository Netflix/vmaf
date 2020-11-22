#!/usr/bin/env python3

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from vmaf.core.train_test_model import LibsvmNusvrTrainTestModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-pkl-filepath", dest="input_pkl_filepath", nargs=1, type=str,
        help="path to the input pkl file, example: model/vmaf_float_v0.6.1.pkl or model/vmaf_float_b_v0.6.3/vmaf_float_b_v0.6.3.pkl", required=True)

    parser.add_argument(
        "--output-json-filepath", dest="output_json_filepath", nargs=1, type=str,
        help="path to the output json file, example: model/vmaf_float_v0.6.1.json or model/vmaf_float_b_v0.6.3.json", required=True)

    args = parser.parse_args()

    input_pkl_filepath = args.input_pkl_filepath[0]
    output_json_filepath = args.output_json_filepath[0]

    loaded_model = LibsvmNusvrTrainTestModel.from_file(input_pkl_filepath, None, format='pkl')
    loaded_model.to_file(output_json_filepath, format='json', combined=True)

    print(f'converted {loaded_model.TYPE} model from pkl file {input_pkl_filepath} to json file {output_json_filepath}')

    exit(0)
