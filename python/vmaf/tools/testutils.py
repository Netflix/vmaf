import re
from typing import List, Optional

from vmaf.tools.misc import MyTestCase


def get_tidy_mock_call_args_list(mockProcessRunner_run) -> List[str]:
    l = list()
    for i in range(len(mockProcessRunner_run.call_args_list)):
        e = mockProcessRunner_run.call_args_list[i][0][0]
        if isinstance(e, str):
            l.append(e)
        else:
            l.append(' '.join(e))
    return l


def replace_uuid(command_line: str) -> str:
    """
    Replace UUIDs in a command line with pattern [UUID]

    >>> replace_uuid('/tmp/72b3a7af-204c-4455-afe5-be2d536f2fdd/dv_el_out_1.h265')
    '/tmp/[UUID]/dv_el_out_1.h265'
    """
    uuid_pattern = r'\b[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12}\b'
    return re.sub(uuid_pattern, '[UUID]', command_line)


def replace_root(command_line: str, root: str) -> str:
    """
    Replace root directory specified in input with pattern [ROOT]

    >>> replace_root('/opt/project/vmaf/libvmaf/build/tools/vmaf', root='/opt/project')
    '[ROOT]/vmaf/libvmaf/build/tools/vmaf'
    >>> replace_root('/tmp/72b3a7af-204c-4455-afe5-be2d536f2fdd', root='/opt/project')
    '/tmp/72b3a7af-204c-4455-afe5-be2d536f2fdd'
    """
    return re.sub(root, '[ROOT]', command_line)


def remove_redundant_whitespace(command_line: str) -> str:
    """
    Replaces multiple whitespace between words with a single one, and removes redundant whitespace at the start and end
    >>> remove_redundant_whitespace('  a b  c d   e     f ')
    'a b c d e f'
    >>> remove_redundant_whitespace('cat /opt/project/vmaf/workspace/workdir/9e693ccc-7706-49c5-8c8e-40f5242e81a6/dis_test_0_0_seeking_10_288_375_notyuv_lanczos_accurate_rnd_10to14_prece_FFmpegDecoder_postunsharpunsharp_q_480x360_PostDecode_tmp/pixfmt/* >  /opt/project/vmaf/workspace/workdir/9e693ccc-7706-49c5-8c8e-40f5242e81a6/dis_test_0_0_seeking_10_288_375_notyuv_lanczos_accurate_rnd_10to14_prece_FFmpegDecoder_postunsharpunsharp_q_480x360_PostPreresamplingFilter0 ')
    'cat /opt/project/vmaf/workspace/workdir/9e693ccc-7706-49c5-8c8e-40f5242e81a6/dis_test_0_0_seeking_10_288_375_notyuv_lanczos_accurate_rnd_10to14_prece_FFmpegDecoder_postunsharpunsharp_q_480x360_PostDecode_tmp/pixfmt/* > /opt/project/vmaf/workspace/workdir/9e693ccc-7706-49c5-8c8e-40f5242e81a6/dis_test_0_0_seeking_10_288_375_notyuv_lanczos_accurate_rnd_10to14_prece_FFmpegDecoder_postunsharpunsharp_q_480x360_PostPreresamplingFilter0'
    """
    return " ".join(command_line.split())


def remove_option(command_line: str, option: str) -> str:
    """
    Removes a whitespace-separated option that is prefixed by two dashes, e.g., --option_name.
    >>> remove_option('vmaf --reference REFERENCE --model MODEL', 'model')
    'vmaf --reference REFERENCE'
    >>> remove_option('vmaf --model MODEL --reference REFERENCE', 'model')
    'vmaf --reference REFERENCE'
    >>> remove_option(remove_option('vmaf --model MODEL --dist DIST --reference REFERENCE', 'model'), 'reference')
    'vmaf --dist DIST'
    >>> remove_option(remove_option('vmaf --model MODEL --dist DIST --reference REFERENCE', 'reference'), 'model')
    'vmaf --dist DIST'
    >>> remove_option('vmaf --reference REFERENCE', 'model')
    'vmaf --reference REFERENCE'
    >>> remove_option('a --model M b', 'model')
    'a b'
    >>> remove_option('a --model KLM b', 'model')
    'a b'
    >>> remove_option('abc --model K def', 'model')
    'abc def'
    >>> remove_option('abc --model KLM def', 'model')
    'abc def'
    >>> remove_option('abc --model KLM d', 'model')
    'abc d'
    >>> remove_option('a --model KLM def', 'model')
    'a def'
    >>> remove_option('a --model M b c --model M d', 'model')
    'a b c d'
    >>> remove_option('a --model M', 'model')
    'a'
    >>> remove_option('a --model M ', 'model')
    'a '
    >>> remove_option(' --model M ', 'model')
    ' '
    >>> remove_option('--model M ', 'model')
    ' '
    >>> remove_option('--model2 M ', 'model')
    '--model2 M '
    >>> remove_option('-- model M ', 'model')
    '-- model M '
    """
    if command_line.startswith('--{option}'.format(option=option)):
        return re.sub(r'--{option} [^\s]*'.format(option=option), '', command_line)
    else:
        return re.sub(r' --{option} [^\s]*'.format(option=option), '', command_line)


def remove_elements_containing_substring(command_line: str, sub_str: str) -> str:
    """
    Removes strings from the command line that contain a specific substring
    >>> remove_elements_containing_substring('cat /opt/project/vmaf/workspace/workdir/9e693ccc-7706-49c5-8c8e-40f5242e81a6/dis_test_0_0_seeking_10_288_375_notyuv_lanczos_accurate_rnd_10to14_prece_FFmpegDecoder_postunsharpunsharp_q_480x360_PostDecode_tmp/pixfmt/* >  /opt/project/vmaf/workspace/workdir/9e693ccc-7706-49c5-8c8e-40f5242e81a6/dis_test_0_0_seeking_10_288_375_notyuv_lanczos_accurate_rnd_10to14_prece_FFmpegDecoder_postunsharpunsharp_q_480x360_PostPreresamplingFilter0', 'workspace/workdir')
    'cat >'
    """
    assert isinstance(sub_str, str)
    return " ".join([x for x in command_line.split() if sub_str not in x])


def assert_equivalent_commands(self, cmds: List[str], cmds_expected: List[str], root: str, root_expected: str, do_replace_uuid: bool = True,
                               options_to_remove: Optional[List[str]] = None, substrings_to_remove: Optional[List[str]] = None):
    """
    >>> self = MyTestCase()
    >>> self.setUp()
    >>> assert_equivalent_commands(self, cmds=["/opt/project/vmaf/libvmaf/build/tools/vmaf /tmp/72b3a7af-204c-4455-afe5-be2d536f2fdd/dv_el_out_1.h265"], cmds_expected=["/opt/project/vmaf/libvmaf/build/tools/vmaf /tmp/82b3a7af-304c-5455-afe5-be2d536f2fdd/dv_el_out_1.h265"], root="/opt/project", root_expected="/opt/project")
    >>> self.tearDown()
    >>> self2 = MyTestCase()
    >>> self2.setUp()
    >>> assert_equivalent_commands(self2, cmds=["/opt/project/vmaf/libvmaf/build/tools/vmaf /tmp/72b3a7af-204c-4455-afe5-be2d536f2fdd/dv_el_out_1.h265"], cmds_expected=["/opt/project/vmaf/libvmaf/build/tools/vmaf /tmp/82b3a7af-304c-5455-afe5-be2d536f2fdd/dv_el_out_1.h265"], root="/opt/project", root_expected="/opt/project", do_replace_uuid=False)
    >>> with self.assertRaises(AssertionError): self2.tearDown()
    >>> self3 = MyTestCase()
    >>> self3.setUp()
    >>> assert_equivalent_commands(self3, cmds=["/opt/project/vmaf --reference ref.h265 --distorted dist.h265"], cmds_expected=["/opt/project/vmaf --reference ref.h265 --distorted dist.h266"], root="/opt/project", root_expected="/opt/project", options_to_remove=["distorted"])
    >>> self3.tearDown()
    >>> self4 = MyTestCase()
    >>> self4.setUp()
    >>> assert_equivalent_commands(self4, cmds=["/opt/project/vmaf --reference ref.h265 --distorted dist.h265 --output output.xml"], cmds_expected=["/opt/project/vmaf --reference ref.h266 --distorted dist.h266 --output output.xml"], root="/opt/project", root_expected="/opt/project", options_to_remove=["distorted", "reference"])
    >>> self4.tearDown()
    >>> self5 = MyTestCase()
    >>> self5.setUp()
    >>> assert_equivalent_commands(self5, cmds=["/opt/project/vmaf --reference /opt/project/vmaf/workspace/workdir/ref.h265 --distorted /opt/project/vmaf/workspace/workdir/dist.h265 --output output.xml"], cmds_expected=["/opt/project/vmaf --reference /opt/project/vmaf/workspace/workdir/ref.h266 --distorted /opt/project/vmaf/workspace/workdir/dist.h266 --output output2.xml"], root="/opt/project", root_expected="/opt/project", substrings_to_remove=["workspace/workdir", "output"])
    >>> self5.tearDown()
    """

    if options_to_remove is None:
        options_to_remove = []
    if substrings_to_remove is None:
        substrings_to_remove = []
    assert len(cmds) == len(cmds_expected), f"length of cmds and cmds_expected are not equal: {len(cmds)} vs. {len(cmds_expected)}"
    for cmd, cmd_expected in zip(cmds, cmds_expected):

        if do_replace_uuid is True:
            cmd1 = replace_uuid(cmd)
        else:
            cmd1 = cmd
        cmd2 = replace_root(cmd1, root)
        cmd3 = remove_redundant_whitespace(cmd2)
        for option_to_remove in options_to_remove:
            cmd3 = remove_option(cmd3, option_to_remove)
        for sbstr_to_remove in substrings_to_remove:
            cmd3 = remove_elements_containing_substring(cmd3, sbstr_to_remove)

        cmd_expected1 = replace_uuid(cmd_expected)
        cmd_expected2 = replace_root(cmd_expected1, root_expected)
        cmd_expected3 = remove_redundant_whitespace(cmd_expected2)
        for option_to_remove in options_to_remove:
            cmd_expected3 = remove_option(cmd_expected3, option_to_remove)
        for sbstr_to_remove in substrings_to_remove:
            cmd_expected3 = remove_elements_containing_substring(cmd_expected3, sbstr_to_remove)

        self.assertEqual(cmd3, cmd_expected3, msg=f"cmd and cmd_expected are not matched:\ncmd: {cmd}\ncmd:expected: "
                                                  f"{cmd_expected}\nprocessed cmd: {cmd3}\nprocessed cmd:expected: "
                                                  f"{cmd_expected3}")
