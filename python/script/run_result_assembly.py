#!/usr/bin/env python3

import os
import sys
import re

import json

from vmaf.core.result import Result

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


def print_usage():
    print("usage: python " + os.path.basename(sys.argv[0]) + " input_file\n")
    print("input_file contains a list of files for assembly (can be xml or json)")


class FileAssembler:

    SUPPORTED_FILE_TYPES = ['xml', 'json']

    def __init__(self, to_assemble_input):
        self.to_assemble_input = to_assemble_input

    @staticmethod
    def create_assembly_file_list(to_assemble_input):

        to_assemble_list = []
        if isinstance(to_assemble_input, list):
            to_assemble_list = to_assemble_input
        else:
            with open(to_assemble_input, "rt") as input_file:
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

                    to_assemble_list.append(line.strip())

        return to_assemble_list

    def _create_result_list(self, to_assemble_list):
        pass

    def assemble(self):
        """
        Main file assembly logic
        """
        to_assemble_list = self.create_assembly_file_list(self.to_assemble_input)
        self._assert(to_assemble_list)
        results = self._create_result_list(to_assemble_list)
        combined_result = Result.combine_result(results)

        return combined_result

    def _assert(self, to_assemble_list):
        """
        Perform necessary assertions before parsing any of the files.
        """

        # check that the number of files is greater than 0
        assert len(to_assemble_list) > 0
        # check that the file formats match
        assemble_format_list = [os.path.splitext(f)[1].split(".")[1] for f in to_assemble_list]
        assert len(set(assemble_format_list)) == 1, "The file formats for assembly do not much."
        # check that the file format is supported for assembly
        assert assemble_format_list[0] in self.SUPPORTED_FILE_TYPES, \
            "The assembly format is not consistent, use any of {fmts}".format(fmts=str(self.SUPPORTED_FILE_TYPES))


class XmlAssembler(FileAssembler):

    @staticmethod
    def _parse_files(to_assemble_list):

        to_assemble_xml_strings = []
        for to_assemble_xml in to_assemble_list:
            with open(to_assemble_xml, 'r') as f:
                to_assemble_xml_strings.append(f.read())

        return to_assemble_xml_strings

    def _create_result_list(self, to_assemble_list):

        to_assemble_xml_strings = self._parse_files(to_assemble_list)

        results = []
        for to_assemble_xml_string in to_assemble_xml_strings:
            results.append(Result.from_xml(to_assemble_xml_string))

        return results


class JsonAssembler(FileAssembler):

    @staticmethod
    def _parse_files(to_assemble_list):

        to_assemble_json_strings = []
        for json_file in to_assemble_list:
            with open(json_file, 'r') as f:
                to_assemble_json_strings.append(json.load(f))

        return to_assemble_json_strings

    def _create_result_list(self, to_assemble_list):

        to_assemble_jsons = self._parse_files(to_assemble_list)

        results = []
        for to_assemble_json in to_assemble_jsons:
            to_assemble_json_string = json.dumps(to_assemble_json)
            results.append(Result.from_json(to_assemble_json_string))

        return results


def main():

    if len(sys.argv) != 2:
        print_usage()
        return 2

    files_to_assemble_list = sys.argv[1]

    desired_file_list = FileAssembler.create_assembly_file_list(files_to_assemble_list)

    if ".xml" in desired_file_list[0]:
        xml_output = XmlAssembler(files_to_assemble_list).assemble().to_xml()
        print(xml_output)
    elif ".json" in desired_file_list[0]:
        json_output = JsonAssembler(files_to_assemble_list).assemble().to_json()
        print(json_output)
    else:
        print_usage()
        return 2


if __name__ == "__main__":
    ret = main()
    exit(ret)
