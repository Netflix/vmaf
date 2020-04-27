import os
import subprocess
import unittest

from vmaf import run_process
from vmaf.config import VmafConfig, VmafExternalConfig

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


@unittest.skipIf(not VmafExternalConfig.ffmpeg_path(), "ffmpeg not installed")
class CommandLineTest(unittest.TestCase):

    def setUp(self):
        self.dataset_filename = VmafConfig.test_resource_path('example_dataset.py')
        self.raw_dataset_filename = VmafConfig.test_resource_path('example_raw_dataset.py')
        self.out_model_filepath = VmafConfig.workdir_path('tmp.pkl')
        self.param_filename = VmafConfig.test_resource_path('vmaf_v4.py')
        self.batch_filename = VmafConfig.workdir_path('test_batch_input')

    def tearDown(self):
        if os.path.exists(self.out_model_filepath):
            os.remove(self.out_model_filepath)
        if os.path.exists(self.out_model_filepath + '.model'):
            os.remove(self.out_model_filepath + '.model')
        if os.path.exists(self.batch_filename):
            os.remove(self.batch_filename)

    def test_run_ffmpeg2vmaf(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'ffmpeg2vmaf.py')
        line = '576 324 {ref} {dis}'.format(ref=VmafConfig.test_resource_path('mp4', 'Seeking_30_480_1050.mp4'),
                                            dis=VmafConfig.test_resource_path('mp4', 'Seeking_10_288_375.mp4'))
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_ffmpeg2vmaf_ci(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'ffmpeg2vmaf.py')
        line = '576 324 {ref} {dis}'.format(ref=VmafConfig.test_resource_path('mp4', 'Seeking_30_480_1050.mp4'),
                                            dis=VmafConfig.test_resource_path('mp4', 'Seeking_10_288_375.mp4'))
        cmd = "{exe} {line} --ci >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_ffmpeg2vmaf_ci_and_local_explain(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'ffmpeg2vmaf.py')
        line = '576 324 {ref} {dis}'.format(ref=VmafConfig.test_resource_path('mp4', 'Seeking_30_480_1050.mp4'),
                                            dis=VmafConfig.test_resource_path('mp4', 'Seeking_10_288_375.mp4'))
        cmd = "{exe} {line} --ci --local-explain >/dev/null 2>&1".format(line=line, exe=exe)
        ret = subprocess.call(cmd, shell=True)
        self.assertEqual(ret, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
