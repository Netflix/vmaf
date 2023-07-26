import os.path
import re
from typing import List


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


def assert_equivalent_commands(self, cmds: List[str], cmds_expected: List[str], root: str, root_expected: str):
    assert len(cmds) == len(cmds_expected), f"length of cmds and cmds_expected are not equal: {len(cmds)} vs. {len(cmds_expected)}"
    for cmd, cmd_expected in zip(cmds, cmds_expected):

        cmd1 = replace_uuid(cmd)
        cmd2 = replace_root(cmd1, root)

        cmd_expected1 = replace_uuid(cmd_expected)
        cmd_expected2 = replace_root(cmd_expected1, root_expected)

        self.assertEqual(cmd2, cmd_expected2, msg=f"cmd and cmd_expected are not matched:\ncmd: {cmd}\ncmd:expected: {cmd_expected}")
