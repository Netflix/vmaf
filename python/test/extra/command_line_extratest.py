import os
import subprocess
import unittest

from vmaf import run_process
from vmaf.config import VmafConfig

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"

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
        exe = VmafConfig.root_path('ffmpeg2vmaf')
        line = '576 324 {root}/python/test/resource/mp4/Seeking_30_480_1050.mp4 ' \
               '{root}/python/test/resource/mp4/Seeking_10_288_375.mp4'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_ffmpeg2vmaf_ci(self):
        exe = VmafConfig.root_path('ffmpeg2vmaf')
        line = '576 324 {root}/python/test/resource/mp4/Seeking_30_480_1050.mp4 ' \
               '{root}/python/test/resource/mp4/Seeking_10_288_375.mp4'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} --ci >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_ffmpeg2vmaf_ci_and_local_explain(self):
        exe = VmafConfig.root_path('ffmpeg2vmaf')
        line = '576 324 {root}/python/test/resource/mp4/Seeking_30_480_1050.mp4 ' \
               '{root}/python/test/resource/mp4/Seeking_10_288_375.mp4'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} --ci --local-explain >/dev/null 2>&1".format(line=line, exe=exe)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 2)
