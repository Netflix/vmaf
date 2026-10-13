from __future__ import absolute_import

import unittest
from unittest import mock

from vmaf import ExternalProgramCaller
from test.testutil import set_default_576_324_hdr_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

COLOR = {'range': 'limited', 'primaries': 'bt2020', 'trc': 'smpte2084', 'matrix': 'bt2020nc'}


class VmafexecColorArgsTest(unittest.TestCase):

    def _build_cmd(self, **color_kwargs):
        ref_path, dis_path, _, _ = set_default_576_324_hdr_videos_for_testing()
        with mock.patch('vmaf.run_process') as run_process:
            ExternalProgramCaller.call_vmafexec(
                ref_path, dis_path, 576, 324, '420', 10,
                False, False, False, False, False, False, False,
                True, None, 1, 1, False, 'out.xml', 'vmaf', None, **color_kwargs)
        return run_process.call_args[0][0]

    def test_no_color_adds_no_color_flags(self):
        self.assertNotIn('--color_', self._build_cmd())

    def test_color_ref_and_dist_become_per_input_flags(self):
        cmd = self._build_cmd(color_ref=COLOR, color_dist=COLOR)
        for suffix in ('ref', 'dist'):
            self.assertIn('--color_range_{} limited'.format(suffix), cmd)
            self.assertIn('--color_primaries_{} bt2020'.format(suffix), cmd)
            self.assertIn('--color_trc_{} smpte2084'.format(suffix), cmd)
            self.assertIn('--color_matrix_{} bt2020nc'.format(suffix), cmd)

    def test_one_input_can_be_left_unspecified(self):
        cmd = self._build_cmd(color_ref=COLOR)
        self.assertIn('--color_trc_ref smpte2084', cmd)
        self.assertNotIn('_dist', cmd)

    def test_incomplete_color_is_rejected(self):
        with self.assertRaises(AssertionError):
            self._build_cmd(color_ref={'range': 'limited'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
