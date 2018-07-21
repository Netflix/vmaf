#!/usr/bin/env python

import os
import sys
import re

from collections import OrderedDict
from xml.etree import ElementTree
import json
import copy, numpy as np

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


def print_usage():
    print("usage: python " + os.path.basename(sys.argv[0]) + " input_file output_file\n")
    print("input_file contains a list of files for assembly (can be xml or json)")
    print("output_file contains the assembled file")

class FileAssembler():

    SUPPORTED_FILE_TYPES = ['xml', 'json']

    def __init__(self, to_assemble_input):
        self.aggregate_method = ''
        self.executorId = ''
        self.assembled_frame_dict_list = []
        self.scores_dict = OrderedDict()
        self.to_assemble_input = to_assemble_input
        self.to_assemble_list = []

    def _parse_files(self):
        pass

    def _write_assembled(self):
        pass

    def create_assembly_filelist(self):

        if isinstance(self.to_assemble_input, list):
            self.to_assemble_list = self.to_assemble_input
        else:
            with open(self.to_assemble_input, "rt") as input_file:
                for line in input_file.readlines():

                    # match comment
                    mo = re.match(r"^#", line)
                    if mo:
                        print("Skip commented line: {}".format(line))
                        continue

                    # match whitespace
                    mo = re.match(r"[\s]+", line)
                    if mo:
                        continue

                    mo = re.match(r"([\S]+)", line)
                    if not mo:
                        print("Invalid file: {}".format(line))
                        print_usage()
                        return 1

                    self.to_assemble_list.append(line.strip())

    def assemble(self):
        """
        Main file assembly logic
        """
        self.create_assembly_filelist()
        self._assert()
        self._parse_files()
        assembly_output = self._write_assembled()

        return assembly_output

    def create_aggregation_dict(self):

        # TODO fix: only does for aggregation
        aggregate_dict = OrderedDict()
        valid_score_keys = sorted([key for key in self.scores_dict if key.endswith("score")])
        for key in valid_score_keys:
            aggregate_dict[key] = np.mean(self.scores_dict[key])
        aggregate_dict['method'] = self.aggregate_method
        return aggregate_dict, valid_score_keys

    def _assert(self):
        """
        Perform necessary assertions before parsing any of the files.
        """

        # check that the number of files is greater than 0
        assert len(self.to_assemble_list) > 0
        # check that the file formats match
        assemble_format_list = [os.path.splitext(f)[1].split(".")[1] for f in self.to_assemble_list]
        assert len(set(assemble_format_list)) == 1, "The file formats for assembly do not much."
        # check that the file format is supported for assembly
        assert assemble_format_list[0] in self.SUPPORTED_FILE_TYPES, \
            "The assembly format is not consistent, use any of {fmts}".format(fmts = str(self.SUPPORTED_FILE_TYPES))

    def _assert_aggregation(self, aggregation_methods, executor_ids):
        # assert that aggregation method is unique
        assert len(set(aggregation_methods)) == 1, "The aggregation methods do not match."
        # assert that executors which generated the individual file formats are the same
        assert len(set(executor_ids)) == 1, "The executor ids do not match."

class XML_Assembler(FileAssembler):

    @staticmethod
    def prettify(elem):
        from xml.dom import minidom
        rough_string = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _parse_files(self):

        executor_ids = []
        aggregate_methods = []
        aggregated_list = []

        frame_count = 0

        for xml_file in self.to_assemble_list:

            top = ElementTree.parse(xml_file).getroot()
            frames = top.find('frames')
            for frame in frames:
                frame.set('frameNum', str(frame_count))
                frame_count += 1

                assembled_frame = copy.deepcopy(frame)
                assembled_frame.tail = None
                self.assembled_frame_dict_list.append(assembled_frame)
                for attribute_key, attribute_value in frame.attrib.items():
                    self.scores_dict.setdefault(attribute_key, []).append(float(attribute_value))

            executor_ids.append(top.get('executorId'))
            aggregate_methods.append(top.find('aggregate').get('method'))
            aggregated_list.append(top.find('aggregate'))

        self._assert_aggregation(aggregate_methods, executor_ids)

        self.executorId = executor_ids[0]
        self.aggregate_method = aggregate_methods[0]

    def _write_assembled(self):

        assembled_xml_top = ElementTree.Element('result')

        asset = ElementTree.SubElement(assembled_xml_top, 'asset')
        asset.set('identifier', 'common')

        assembled_frames = ElementTree.SubElement(assembled_xml_top, 'frames')

        for frame in self.assembled_frame_dict_list:
            assembled_frames.append(frame)

        aggregate_dict, valid_score_keys = self.create_aggregation_dict()

        aggregate = ElementTree.SubElement(assembled_xml_top, 'aggregate')

        for key in valid_score_keys:
            aggregate.set(key, str(aggregate_dict[key]))

        aggregate.set('method', self.aggregate_method)
        assembled_xml_top.set('executorId', self.executorId)

        return self.prettify(assembled_xml_top)

class JSON_Assembler(FileAssembler):

    def _parse_files(self):

        frame_count = 0

        executor_ids = []
        aggregate_methods = []

        for json_file in self.to_assemble_list:
            with open(json_file) as f:
                data = json.load(f)
                executor_ids.append(data['executorId'])
                aggregate_methods.append(data['aggregate']['method'])
                frames = data['frames']

                for frame in frames:

                    assembled_frame = OrderedDict()
                    sorted_keys = sorted([key for key in frame.keys()], reverse=True)
                    for attribute_key in sorted_keys:
                        assembled_frame[attribute_key] = frame[attribute_key]
                        self.scores_dict.setdefault(attribute_key, []).append(float(frame[attribute_key]))

                    assembled_frame['frameNum'] = frame_count
                    frame_count += 1

                    self.assembled_frame_dict_list.append(assembled_frame)

        self._assert_aggregation(aggregate_methods, executor_ids)

        self.executorId = executor_ids[0]
        self.aggregate_method = aggregate_methods[0]

    def _write_assembled(self):

        aggregate_dict, _ = self.create_aggregation_dict()

        assembled_data_dict = OrderedDict()
        assembled_data_dict['executorId'] = self.executorId
        assembled_data_dict['asset'] = {}
        assembled_data_dict['asset']['identifier'] = 'common'
        assembled_data_dict['frames'] = self.assembled_frame_dict_list
        assembled_data_dict['aggregate'] = aggregate_dict

        return json.dumps(assembled_data_dict, sort_keys=False, indent=4)

def main():

    if len(sys.argv) != 2:
        print_usage()
        return 2

    files_to_assemble_list = sys.argv[1]

    JSON_Assembler(files_to_assemble_list).assemble()
    # XML_Assembler(files_to_assemble_list).assemble()

if __name__ == "__main__":
    ret = main()
    exit(ret)